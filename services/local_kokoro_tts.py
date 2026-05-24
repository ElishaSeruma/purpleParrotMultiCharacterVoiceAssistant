# services/local_kokoro_tts.py
import asyncio
import os
import logging
import time
import numpy as np
import uuid
from collections import OrderedDict
from concurrent.futures import Executor
from threading import Lock
from typing import Optional, TypedDict
from huggingface_hub import hf_hub_download
from livekit.agents import APIConnectOptions, tts

logger = logging.getLogger("local_kokoro_tts")
logger.setLevel(logging.INFO)

class KokoroPersonaVoiceProfile(TypedDict):
    voice: str
    speed: float
    acoustic_note: str


KOKORO_PERSONA_VOICE_PROFILES: dict[str, KokoroPersonaVoiceProfile] = {
    "kami": {
        "voice": "af_bella",
        "speed": 1.0,
        "acoustic_note": "Warm, balanced, clear neutral narrator.",
    },
    "patty": {
        "voice": "af_sarah",
        "speed": 1.18,
        "acoustic_note": "Bright, quick, expressive teen-energy profile.",
    },
    "bram": {
        "voice": "am_puck",
        "speed": 1.12,
        "acoustic_note": "Bouncy casual young male profile.",
    },
    "atlas": {
        "voice": "am_michael",
        "speed": 0.92,
        "acoustic_note": "Calm, polished, deeper adult male profile.",
    },
    "vela": {
        "voice": "af_nova",
        "speed": 0.98,
        "acoustic_note": "Elegant, musical, theatrical female profile.",
    },
    "suki": {
        "voice": "bf_lily",
        "speed": 0.88,
        "acoustic_note": "Soft, slow, quiet British female profile.",
    },
    "kiko": {
        "voice": "am_eric",
        "speed": 1.2,
        "acoustic_note": "Fast, jumpy, bright young male profile.",
    },
    "nori": {
        "voice": "af_kore",
        "speed": 0.96,
        "acoustic_note": "Cool, dry, confident female profile.",
    },
    "miso": {
        "voice": "af_jessica",
        "speed": 1.08,
        "acoustic_note": "Small, precise, bright teen profile.",
    },
    "rune": {
        "voice": "bf_emma",
        "speed": 0.82,
        "acoustic_note": "Soft, sleepy, old-soul British female profile.",
    },
    "tavo": {
        "voice": "am_adam",
        "speed": 0.84,
        "acoustic_note": "Lower, slower, gruffer adult male profile.",
    },
    "zeni": {
        "voice": "af_sky",
        "speed": 1.0,
        "acoustic_note": "Crisp, clean, controlled adult female profile.",
    },
}
KOKORO_PERSONA_VOICE_MATRIX = {
    persona: profile["voice"] for persona, profile in KOKORO_PERSONA_VOICE_PROFILES.items()
}

class LocalKokoroTTS(tts.TTS):
    def __init__(self, executor: Executor | None = None, cache_size: int | None = None):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=24000,   # Kokoro's native generation frequency
            num_channels=1       # Mono channel audio output
        )
        self._executor = executor
        self._cache_size = cache_size or int(os.getenv("KOKORO_AUDIO_CACHE_SIZE", "64"))
        self._pcm_block_ms = int(os.getenv("KOKORO_PCM_BLOCK_MS", "100"))
        self._audio_cache: OrderedDict[tuple[str, str, float], tuple[bytes, int, int]] = OrderedDict()
        self._cache_lock = Lock()
        # Import dynamically to isolate resource extraction paths
        from kokoro_onnx import Kokoro

        logger.info("Resolving Kokoro ONNX model assets...")
        model_path, voices_path = self._resolve_model_assets()

        logger.info(f"Successfully retrieved model: {model_path}")
        logger.info(f"Successfully retrieved voice vectors: {voices_path}")

        logger.info("Initializing Localized Kokoro ONNX Instance from cache directory...")
        self._kokoro = Kokoro(model_path, voices_path)
        logger.info("Local Kokoro-82M Vocal Synthesis framework successfully mounted.")

        # Map our System Persona Names to native high-quality Kokoro voice prints
        self._voice_print_matrix = KOKORO_PERSONA_VOICE_MATRIX
        self._voice_profiles = KOKORO_PERSONA_VOICE_PROFILES
        self._active_persona_name = "kami"

    def _resolve_model_assets(self) -> tuple[str, str]:
        model_path = os.getenv("KOKORO_MODEL_PATH")
        voices_path = os.getenv("KOKORO_VOICES_PATH")

        if model_path and voices_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"KOKORO_MODEL_PATH does not exist: {model_path}")
            if not os.path.exists(voices_path):
                raise FileNotFoundError(f"KOKORO_VOICES_PATH does not exist: {voices_path}")
            return model_path, voices_path

        repo_id = os.getenv("KOKORO_HF_REPO", "fastrtc/kokoro-onnx")
        model_file = os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
        voices_file = os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin")

        try:
            return (
                hf_hub_download(repo_id=repo_id, filename=model_file),
                hf_hub_download(repo_id=repo_id, filename=voices_file),
            )
        except Exception:
            logger.exception(
                "Failed to download Kokoro assets from repo=%s model=%s voices=%s. "
                "Set KOKORO_MODEL_PATH and KOKORO_VOICES_PATH to use local files.",
                repo_id,
                model_file,
                voices_file,
            )
            raise

    def _get_voice_for_persona(self, persona_name: str) -> str:
        voice = self.get_voice_profile_for_persona(persona_name)["voice"]
        logger.info("Resolved persona voice persona=%s kokoro_voice=%s", persona_name, voice)
        return voice

    def get_voice_profile_for_persona(self, persona_name: str) -> KokoroPersonaVoiceProfile:
        return self._voice_profiles.get(persona_name.lower(), self._voice_profiles["kami"])

    def set_active_persona(self, persona_name: str) -> KokoroPersonaVoiceProfile:
        self._active_persona_name = persona_name.lower()
        profile = self.get_voice_profile_for_persona(self._active_persona_name)
        logger.info(
            "Active Kokoro persona updated persona=%s kokoro_voice=%s speed=%.2f",
            self._active_persona_name,
            profile["voice"],
            profile["speed"],
        )
        return profile

    def create_pcm(self, text: str, voice: str, speed: float = 1.0) -> tuple[bytes, int, int, bool]:
        if text == "__force_kokoro_error__":
            raise RuntimeError("Forced Kokoro synthesis error for resiliency testing")

        cache_key = (voice, text, speed)
        with self._cache_lock:
            cached = self._audio_cache.get(cache_key)
            if cached:
                self._audio_cache.move_to_end(cache_key)
                pcm_data, sample_rate, sample_count = cached
                logger.info(
                    "tts_audio_cache_hit voice=%s text_length=%d sample_rate=%s samples=%d",
                    voice,
                    len(text),
                    sample_rate,
                    sample_count,
                )
                return pcm_data, sample_rate, sample_count, True

        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
        pcm_data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        sample_count = len(samples)

        with self._cache_lock:
            self._audio_cache[cache_key] = (pcm_data, sample_rate, sample_count)
            self._audio_cache.move_to_end(cache_key)
            while len(self._audio_cache) > self._cache_size:
                self._audio_cache.popitem(last=False)

        return pcm_data, sample_rate, sample_count, False

    def prewarm_text(self, text: str, voice: str, speed: float = 1.0) -> None:
        self.create_pcm(text=text, voice=voice, speed=speed)

    def synthesize(
        self, 
        text: str, 
        *,
        conn_options: Optional[APIConnectOptions] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> "LocalKokoroChunkedStream":
        """
        Synthesizes a standalone, static text block into a complete media frame return object.
        """
        active_profile = self.get_voice_profile_for_persona(self._active_persona_name)
        selected_voice = voice or active_profile["voice"]
        selected_speed = speed if voice else active_profile["speed"]
        return LocalKokoroChunkedStream(
            tts=self,
            create_pcm=self.create_pcm,
            input_text=text,
            voice=selected_voice,
            speed=selected_speed,
            pcm_block_ms=self._pcm_block_ms,
            conn_options=conn_options or APIConnectOptions(),
            executor=self._executor,
        )


class LocalKokoroChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: LocalKokoroTTS,
        create_pcm,
        input_text: str,
        voice: str,
        speed: float,
        pcm_block_ms: int,
        conn_options: APIConnectOptions,
        executor: Executor | None,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._create_pcm = create_pcm
        self._text = input_text
        self._voice = voice
        self._speed = speed
        self._pcm_block_ms = pcm_block_ms
        self._executor = executor

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Generates audio and hands raw PCM to LiveKit's current ChunkedStream API.
        Transforms Kokoro's native
        24kHz float32 arrays into standard LiveKit 24kHz 16-bit linear PCM arrays.
        """
        loop = asyncio.get_running_loop()

        started_at = time.perf_counter()
        try:
            # Offload high-density audio wave synthesis to executor threads
            pcm_data, sample_rate, sample_count, cache_hit = await loop.run_in_executor(
                self._executor,
                lambda: self._create_pcm(self._text, self._voice, self._speed)
            )
            synth_duration_ms = (time.perf_counter() - started_at) * 1000

            audio_duration_ms = (sample_count / max(sample_rate or 24000, 1)) * 1000
            logger.info(
                "tts_synthesis_duration_ms=%.2f audio_duration_ms=%.2f voice=%s speed=%.2f text_length=%d sample_rate=%s samples=%d cache_hit=%s fallback=%s",
                synth_duration_ms,
                audio_duration_ms,
                self._voice,
                self._speed,
                len(self._text),
                sample_rate,
                sample_count,
                cache_hit,
                False,
            )
        except Exception:
            synth_duration_ms = (time.perf_counter() - started_at) * 1000
            sample_rate = 24000
            sample_count = sample_rate // 10
            pcm_data = b"\0\0" * sample_count
            logger.exception(
                "tts_synthesis_failed duration_ms=%.2f voice=%s text_length=%d fallback=silence sample_rate=%d samples=%d",
                synth_duration_ms,
                self._voice,
                len(self._text),
                sample_rate,
                sample_count,
            )

        output_emitter.initialize(
            request_id=f"kokoro-{uuid.uuid4().hex}",
            sample_rate=sample_rate or 24000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        block_size = max(int((sample_rate or 24000) * 2 * self._pcm_block_ms / 1000), 2)
        block_count = 0
        for offset in range(0, len(pcm_data), block_size):
            output_emitter.push(pcm_data[offset:offset + block_size])
            block_count += 1
        logger.info(
            "tts_pcm_blocks_emitted voice=%s sample_rate=%s block_ms=%d blocks=%d bytes=%d",
            self._voice,
            sample_rate,
            self._pcm_block_ms,
            block_count,
            len(pcm_data),
        )
