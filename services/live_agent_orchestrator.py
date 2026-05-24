import hashlib
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

from livekit.agents import Agent, AgentSession

from purpleParrotMultiCharacterVoiceAssistant.services.kami_brain import KamiBrain
from purpleParrotMultiCharacterVoiceAssistant.services.local_kokoro_tts import LocalKokoroTTS
from purpleParrotMultiCharacterVoiceAssistant.services.local_whisper_stt import LocalWhisperSTT

logger = logging.getLogger("live_agent_orchestrator")


@dataclass
class VADRuntime:
    enabled: bool
    provider: str
    instance: object | None
    error: str | None = None


class LiveAgentOrchestrator:
    def __init__(
        self,
        *,
        brain: KamiBrain,
        stt_factory: Callable[[], LocalWhisperSTT],
        tts_factory: Callable[[], LocalKokoroTTS],
    ):
        self._brain = brain
        self._stt_factory = stt_factory
        self._tts_factory = tts_factory
        self._vad_runtime: VADRuntime | None = None
        self._session: AgentSession | None = None
        self._agent: Agent | None = None
        self._last_persona_sync: dict | None = None

        self.turn_detection = os.getenv("LIVEKIT_TURN_DETECTION", "vad")
        self.min_endpointing_delay = float(os.getenv("LIVEKIT_MIN_ENDPOINTING_DELAY", "0.25"))
        self.max_endpointing_delay = float(os.getenv("LIVEKIT_MAX_ENDPOINTING_DELAY", "1.25"))
        self.min_interruption_duration = float(os.getenv("LIVEKIT_MIN_INTERRUPTION_DURATION", "0.18"))
        self.min_interruption_words = int(os.getenv("LIVEKIT_MIN_INTERRUPTION_WORDS", "0"))
        self.allow_interruptions = os.getenv("LIVEKIT_ALLOW_INTERRUPTION", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _load_vad(self) -> VADRuntime:
        if self._vad_runtime is not None:
            return self._vad_runtime

        if os.getenv("LIVEKIT_SILERO_VAD_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
            self._vad_runtime = VADRuntime(enabled=False, provider="disabled", instance=None)
            return self._vad_runtime

        try:
            from livekit.plugins import silero

            vad = silero.VAD.load(
                min_speech_duration=float(os.getenv("SILERO_MIN_SPEECH_DURATION", "0.05")),
                min_silence_duration=float(os.getenv("SILERO_MIN_SILENCE_DURATION", "0.45")),
                prefix_padding_duration=float(os.getenv("SILERO_PREFIX_PADDING_DURATION", "0.2")),
                activation_threshold=float(os.getenv("SILERO_ACTIVATION_THRESHOLD", "0.5")),
                sample_rate=int(os.getenv("SILERO_SAMPLE_RATE", "16000")),
            )
            self._vad_runtime = VADRuntime(enabled=True, provider="silero", instance=vad)
            logger.info("Silero VAD loaded for LiveKit AgentSession endpointing")
        except Exception as exc:
            self._vad_runtime = VADRuntime(
                enabled=False,
                provider="unavailable",
                instance=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.warning("Silero VAD unavailable; AgentSession will expose fallback config: %s", exc)

        return self._vad_runtime

    def _instructions_hash(self, instructions: str) -> str:
        return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:12]

    def ensure_session(self) -> AgentSession:
        if self._session is not None:
            return self._session

        vad_runtime = self._load_vad()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        turn_handling = {
            "turn_detection": self.turn_detection,
            "endpointing": {
                "min_delay": self.min_endpointing_delay,
                "max_delay": self.max_endpointing_delay,
            },
            "interruption": {
                "enabled": self.allow_interruptions,
                "mode": "vad",
                "min_duration": self.min_interruption_duration,
                "min_words": self.min_interruption_words,
                "discard_audio_if_uninterruptible": True,
            },
        }
        session_kwargs = {
            "stt": self._stt_factory(),
            "tts": self._tts_factory(),
            "turn_handling": turn_handling,
            "loop": loop,
        }
        if vad_runtime.instance is not None:
            session_kwargs["vad"] = vad_runtime.instance

        self._session = AgentSession(**session_kwargs)
        self._agent = self._build_agent()
        logger.info(
            "LiveKit AgentSession configured vad_provider=%s turn_detection=%s allow_interruptions=%s",
            vad_runtime.provider,
            self.turn_detection,
            self.allow_interruptions,
        )
        return self._session

    def _build_agent(self) -> Agent:
        instructions = self._brain.compile_system_instructions()
        return Agent(
            instructions=instructions,
            stt=self._stt_factory(),
            tts=self._tts_factory(),
            turn_handling={
                "turn_detection": self.turn_detection,
                "interruption": {
                    "enabled": self.allow_interruptions,
                    "mode": "vad",
                    "min_duration": self.min_interruption_duration,
                    "min_words": self.min_interruption_words,
                    "discard_audio_if_uninterruptible": True,
                },
            },
        )

    def apply_persona(self, persona_name: str) -> dict:
        persona = self._brain.switch_persona(persona_name)
        tts_engine = self._tts_factory()
        voice_profile = tts_engine.set_active_persona(self._brain.active_persona_name)
        instructions = self._brain.compile_system_instructions()

        self.ensure_session()
        self._agent = self._build_agent()

        agent_hot_swap_applied = False
        if self._session is not None and self._agent is not None:
            self._session.update_agent(self._agent)
            agent_hot_swap_applied = True

        self._last_persona_sync = {
            "active_persona": self._brain.active_persona_name,
            "persona_display_name": persona.name,
            "kokoro_voice": voice_profile["voice"],
            "kokoro_speed": voice_profile["speed"],
            "kokoro_acoustic_note": voice_profile["acoustic_note"],
            "instructions_hash": self._instructions_hash(instructions),
            "instructions_preview": instructions[:240],
            "agent_hot_swap_applied": agent_hot_swap_applied,
            "synced_at": time.time(),
        }
        logger.info(
            "ThemeMutationEvent applied to live agent persona=%s voice=%s speed=%.2f instructions_hash=%s",
            self._last_persona_sync["active_persona"],
            self._last_persona_sync["kokoro_voice"],
            self._last_persona_sync["kokoro_speed"],
            self._last_persona_sync["instructions_hash"],
        )
        return self._last_persona_sync

    def status(self) -> dict:
        vad_runtime = self._load_vad()
        instructions = self._brain.compile_system_instructions()
        return {
            "agent_session_configured": self._session is not None,
            "agent_configured": self._agent is not None,
            "active_persona": self._brain.active_persona_name,
            "instructions_hash": self._instructions_hash(instructions),
            "vad": {
                "enabled": vad_runtime.enabled,
                "provider": vad_runtime.provider,
                "error": vad_runtime.error,
            },
            "turn_detection": self.turn_detection,
            "endpointing": {
                "min_endpointing_delay": self.min_endpointing_delay,
                "max_endpointing_delay": self.max_endpointing_delay,
            },
            "barge_in": {
                "allow_interruptions": self.allow_interruptions,
                "min_interruption_duration": self.min_interruption_duration,
                "min_interruption_words": self.min_interruption_words,
            },
            "last_persona_sync": self._last_persona_sync,
        }

    def compose_local_reply(self, user_text: str) -> str:
        persona_name = self._brain.active_persona_name
        self._brain.append_interaction("user", user_text)
        reply = f"{persona_name} heard: {user_text.strip() or 'your voice'}. Let's keep it short and clear."
        self._brain.append_interaction("assistant", reply)
        return reply

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        self._agent = None
