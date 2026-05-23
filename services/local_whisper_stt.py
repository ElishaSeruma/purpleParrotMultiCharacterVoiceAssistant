# services/local_whisper_stt.py
import asyncio
import logging
import numpy as np
from typing import Optional
from livekit import rtc
from livekit.agents import stt

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
        conn_options: Optional[stt.APIConnectOptions] = None
    ) -> "LocalWhisperRecognizeStream":
        """
        Complies explicitly with LiveKit's required streaming factory signature.
        """
        return LocalWhisperRecognizeStream(self._model, language or "en")

    async def _recognize_impl(
        self, 
        *, 
        buffer: rtc.AudioFrame, 
        language: Optional[str] = "en", 
        conn_options: Optional[stt.APIConnectOptions] = None
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
    def __init__(self, model, language: str):
        super().__init__()
        self._model = model
        self._language = language
        self._audio_buffer = bytearray()
        self._event_queue = asyncio.Queue()
        self._closed = False
        
        # Fire up our asynchronous worker pipeline
        self._process_task = asyncio.create_task(self._transcription_worker_loop())

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Receives raw WebRTC block signals instantly out of LiveKit's track loop.
        """
        if self._closed:
            return
        # Dynamically append raw linear 16-bit PCM buffer array allocations
        self._audio_buffer.extend(frame.data)

    def end_input(self) -> None:
        """Indicates the incoming live track stream has concluded."""
        self._closed = True

    async def aclose(self) -> None:
        """Safely cleans up stream tasks and queues."""
        self._closed = True
        self._process_task.cancel()
        try:
            await self._process_task
        except asyncio.CancelledError:
            pass

    async def _transcription_worker_loop(self):
        """
        Sliding-window evaluation loops pushing chunks to the local model thread pool.
        """
        try:
            while not self._closed:
                # Continuous tumbling window step every 500ms
                await asyncio.sleep(0.5)
                
                # Ensure we have captured enough chunk volume to model speech (minimum 0.5s at 16kHz)
                if len(self._audio_buffer) < 16000:
                    continue
                
                # Slice our buffer securely for processing snapshot
                raw_bytes = bytes(self._audio_buffer)
                
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
                        event = stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=full_text, language=self._language)]
                        )
                        await self._event_queue.put(event)
                        
        except asyncio.CancelledError:
            pass
        finally:
            # Emit formal completion boundaries back to the pipeline observer loops
            final_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
            await self._event_queue.put(final_event)

    async def __anext__(self) -> stt.SpeechEvent:
        """Allows streaming loops to simply consume events via 'async for' patterns."""
        event = await self._event_queue.get()
        self._event_queue.task_done()
        return event