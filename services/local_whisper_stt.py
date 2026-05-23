# services/local_whisper_stt.py
import asyncio
import logging
import numpy as np
from typing import Optional
from livekit import rtc
from livekit.agents import APIConnectOptions, stt

logger = logging.getLogger("local_whisper_stt")
logger.setLevel(logging.INFO)

class LocalWhisperSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False
            )
        )
        # Load the tiny optimized faster-whisper model into local RAM/VRAM
        from faster_whisper import WhisperModel
        logger.info("Initializing Localized Faster-Whisper Model matrix (cpu/tiny)...")
        self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
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
        )

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
        # Convert incoming audio buffer frame directly to float32 dimensions
        audio_np = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Offload the blocking transcription block to an executor thread
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None, 
            lambda: self._model.transcribe(audio_np, beam_size=1, language=language or "en")
        )
        
        full_text = "".join([seg.text for seg in segments]).strip()
        
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
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._model = model
        self._language = language

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
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Offload to executor to keep the main event thread non-blocking
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(audio_np, beam_size=1, language=self._language)
        )

        segments = list(segments)
        if segments:
            full_text = "".join([seg.text for seg in segments]).strip()
            if full_text:
                # Construct standard compliant structural events
                await self._event_ch.send(
                    stt.SpeechEvent(
                        type=event_type,
                        alternatives=[stt.SpeechData(text=full_text, language=self._language)]
                    )
                )
