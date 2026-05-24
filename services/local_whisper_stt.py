# services/local_whisper_stt.py
import asyncio
import logging
import os
import time
import numpy as np
from concurrent.futures import Executor
from typing import Optional
from livekit import rtc
from livekit.agents import APIConnectOptions, stt

logger = logging.getLogger("local_whisper_stt")
logger.setLevel(logging.INFO)

class LocalWhisperSTT(stt.STT):
    def __init__(self, executor: Executor | None = None):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False
            )
        )
        self._executor = executor
        self._force_next_error = False
        # Load the tiny optimized faster-whisper model into local RAM/VRAM
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL", "tiny.en")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "0"))
        num_workers = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
        logger.info(
            "Initializing Localized Faster-Whisper Model matrix model=%s device=%s compute_type=%s cpu_threads=%d num_workers=%d",
            model_name,
            device,
            compute_type,
            cpu_threads,
            num_workers,
        )
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        logger.info("Local Faster-Whisper Engine primed successfully.")

    def stream(
        self, 
        *, 
        language: Optional[str] = "en", 
        conn_options: Optional[APIConnectOptions] = None
    ) -> "LocalWhisperRecognizeStream":
        """
        Complies explicitly with LiveKit's required streaming factory signature.
        """
        return LocalWhisperRecognizeStream(
            stt=self,
            model=self._model,
            language=language or "en",
            conn_options=conn_options or APIConnectOptions(),
            executor=self._executor,
            should_force_error=self._consume_force_next_error,
        )

    def force_next_error(self) -> None:
        self._force_next_error = True

    def _consume_force_next_error(self) -> bool:
        if self._force_next_error:
            self._force_next_error = False
            return True
        return os.getenv("WHISPER_FORCE_TRANSCRIPTION_ERROR", "").lower() in {"1", "true", "yes", "on"}

    async def _recognize_impl(
        self, 
        *, 
        buffer: rtc.AudioFrame, 
        language: Optional[str] = "en", 
        conn_options: Optional[APIConnectOptions] = None
    ) -> stt.SpeechEvent:
        """
        Required single-frame abstract implementation fallback for non-streaming audio chunks.
        Transforms an isolated AudioFrame directly into a finished SpeechEvent.
        """
        audio_np = _frame_to_float32(buffer)
        logger.info("single_frame_pcm_summary=%s", _pcm_summary(audio_np, buffer.sample_rate, buffer.num_channels))
        
        # Offload the blocking transcription block to an executor thread
        loop = asyncio.get_running_loop()
        started_at = time.perf_counter()
        segments, info = await loop.run_in_executor(
            self._executor,
            lambda: self._model.transcribe(audio_np, beam_size=1, language=language or "en")
        )
        duration_ms = (time.perf_counter() - started_at) * 1000
        
        full_text = "".join([seg.text for seg in segments]).strip()
        logger.info("single_frame_transcription_duration_ms=%.2f text_length=%d", duration_ms, len(full_text))
        
        # Return a finalized final transcript payload event
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=full_text, language=language or "en")]
        )


class LocalWhisperRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: LocalWhisperSTT,
        model,
        language: str,
        conn_options: APIConnectOptions,
        executor: Executor | None,
        should_force_error,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._model = model
        self._language = language
        self._executor = executor
        self._should_force_error = should_force_error

    async def _run(self) -> None:
        """
        Sliding-window evaluation loops pushing chunks to the local model thread pool.
        """
        audio_buffer = bytearray()

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if audio_buffer:
                    await self._transcribe_buffer(bytes(audio_buffer), stt.SpeechEventType.FINAL_TRANSCRIPT)
                    audio_buffer.clear()
                await self._event_ch.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
                continue

            # Dynamically append raw linear 16-bit PCM buffer array allocations
            audio_buffer.extend(item.data)
            logger.debug(
                "stt_frame_received sample_rate=%s channels=%s samples_per_channel=%s buffer_bytes=%s",
                item.sample_rate,
                item.num_channels,
                item.samples_per_channel,
                len(audio_buffer),
            )

            # Process rolling 500ms windows at 16kHz, then clear to avoid repeated transcripts.
            if len(audio_buffer) >= 16000:
                await self._transcribe_buffer(bytes(audio_buffer), stt.SpeechEventType.INTERIM_TRANSCRIPT)
                audio_buffer.clear()

        if audio_buffer:
            await self._transcribe_buffer(bytes(audio_buffer), stt.SpeechEventType.FINAL_TRANSCRIPT)

    async def _transcribe_buffer(self, raw_bytes: bytes, event_type: stt.SpeechEventType) -> None:
        if not raw_bytes:
            return

        # Transform standard 16-bit PCM allocations to float32 dimensions
        audio_np = _pcm_bytes_to_float32(raw_bytes)
        summary = _pcm_summary(audio_np, sample_rate=16000, num_channels=1)
        logger.info("stream_pcm_summary event_type=%s %s", event_type.value, summary)

        # Offload to executor to keep the main event thread non-blocking
        loop = asyncio.get_running_loop()
        started_at = time.perf_counter()
        try:
            if self._should_force_error():
                raise RuntimeError("Forced Whisper transcription error for resiliency testing")

            segments, info = await loop.run_in_executor(
                self._executor,
                lambda: self._model.transcribe(audio_np, beam_size=1, language=self._language)
            )
            duration_ms = (time.perf_counter() - started_at) * 1000

            segments = list(segments)
            full_text = "".join([seg.text for seg in segments]).strip()
            logger.info(
                "stream_transcription_duration_ms=%.2f event_type=%s text_length=%d fallback=%s",
                duration_ms,
                event_type.value,
                len(full_text),
                False,
            )
            if segments:
                if full_text:
                    # Construct standard compliant structural events
                    await self._event_ch.send(
                        stt.SpeechEvent(
                            type=event_type,
                            alternatives=[stt.SpeechData(text=full_text, language=self._language)]
                        )
                    )
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "stream_transcription_failed duration_ms=%.2f event_type=%s audio_samples=%d fallback=empty_transcript",
                duration_ms,
                event_type.value,
                int(audio_np.size),
            )


def _frame_to_float32(frame: rtc.AudioFrame) -> np.ndarray:
    return _pcm_bytes_to_float32(frame.data)


def _pcm_bytes_to_float32(raw_pcm) -> np.ndarray:
    pcm_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
    return pcm_i16.astype(np.float32) / 32768.0


def _pcm_summary(audio_np: np.ndarray, sample_rate: int, num_channels: int) -> dict:
    if audio_np.size == 0:
        return {
            "sample_rate": sample_rate,
            "channels": num_channels,
            "samples": 0,
            "duration_ms": 0.0,
        }

    return {
        "sample_rate": sample_rate,
        "channels": num_channels,
        "samples": int(audio_np.size),
        "duration_ms": round((audio_np.size / max(sample_rate * num_channels, 1)) * 1000, 2),
        "float32_min": round(float(audio_np.min()), 6),
        "float32_max": round(float(audio_np.max()), 6),
        "float32_rms": round(float(np.sqrt(np.mean(np.square(audio_np)))), 6),
        "clipped": bool(np.any(audio_np <= -1.0) or np.any(audio_np >= 0.999969)),
    }
