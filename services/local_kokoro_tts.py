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
from typing import Optional
from huggingface_hub import hf_hub_download
from livekit.agents import APIConnectOptions, tts

logger = logging.getLogger("local_kokoro_tts")
logger.setLevel(logging.INFO)

KOKORO_PERSONA_VOICE_MATRIX = {
    "kami": "af_bella",       # Warm, balanced, clear female narrator
    "patty": "af_sarah",      # High energy, faster pace young profile
    "tavo": "am_adam",        # Deep, gravelly, mature tone profiling
    "zeni": "af_sky"          # Analytical, crisp, clean speech tone
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
        voice = self._voice_print_matrix.get(persona_name.lower(), KOKORO_PERSONA_VOICE_MATRIX["kami"])
        logger.info("Resolved persona voice persona=%s kokoro_voice=%s", persona_name, voice)
        return voice

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
    ) -> "LocalKokoroChunkedStream":
        """
        Synthesizes a standalone, static text block into a complete media frame return object.
        """
        selected_voice = voice or "af_bella"
        return LocalKokoroChunkedStream(
            tts=self,
            create_pcm=self.create_pcm,
            input_text=text,
            voice=selected_voice,
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
        conn_options: APIConnectOptions,
        executor: Executor | None,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._create_pcm = create_pcm
        self._text = input_text
        self._voice = voice
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
                lambda: self._create_pcm(self._text, self._voice, 1.0)
            )
            synth_duration_ms = (time.perf_counter() - started_at) * 1000

            audio_duration_ms = (sample_count / max(sample_rate or 24000, 1)) * 1000
            logger.info(
                "tts_synthesis_duration_ms=%.2f audio_duration_ms=%.2f voice=%s text_length=%d sample_rate=%s samples=%d cache_hit=%s fallback=%s",
                synth_duration_ms,
                audio_duration_ms,
                self._voice,
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
        output_emitter.push(pcm_data)
