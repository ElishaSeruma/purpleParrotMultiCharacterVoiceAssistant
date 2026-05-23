# services/local_kokoro_tts.py
import asyncio
import os
import logging
import numpy as np
from typing import Optional, AsyncIterator
from huggingface_hub import hf_hub_download
from livekit import rtc
from livekit.agents import tts

logger = logging.getLogger("local_kokoro_tts")
logger.setLevel(logging.INFO)

class LocalKokoroTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=24000,   # Kokoro's native generation frequency
            num_channels=1       # Mono channel audio output
        )
        # Import dynamically to isolate resource extraction paths
        from kokoro_onnx import Kokoro
        
        logger.info("Resolving official Kokoro-82M ONNX model assets via authenticated HuggingFace Hub client...")
        
        # Primary official repository identifier
        repo_id = "hexgrad/Kokoro-82M"
        
        try:
            # hexgrad stores the main model file as "model.onnx" and weights matrices as "voices.bin"
            model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
            voices_path = hf_hub_download(repo_id=repo_id, filename="voices.bin")
        except Exception as e:
            logger.error(f"Primary repository target resolution failed: {e}. Trying fallback model layout...")
            # Fallback repository layout mapping if structure shifts
            repo_id = "onnx-community/Kokoro-82M-ONNX"
            model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")
            voices_path = hf_hub_download(repo_id=repo_id, filename="voices.bin")

        logger.info(f"Successfully retrieved model: {model_path}")
        logger.info(f"Successfully retrieved voice vectors: {voices_path}")

        logger.info("Initializing Localized Kokoro ONNX Instance from cache directory...")
        self._kokoro = Kokoro(model_path, voices_path)
        logger.info("Local Kokoro-82M Vocal Synthesis framework successfully mounted.")

        # Map our System Persona Names to native high-quality Kokoro voice prints
        self._voice_print_matrix = {
            "kami": "af_bella",       # Warm, balanced, clear female narrator
            "patty": "af_sarah",      # High energy, faster pace young profile
            "tavo": "am_adam",        # Deep, gravelly, mature tone profiling
            "zeni": "af_sky"          # Analytical, crisp, clean speech tone
        }

    def _get_voice_for_persona(self, persona_name: str) -> str:
        return self._voice_print_matrix.get(persona_name.lower(), "af_bella")

    def synthesize(
        self, 
        text: str, 
        *, 
        voice: Optional[str] = None
    ) -> "LocalKokoroSynthesizedAudio":
        """
        Synthesizes a standalone, static text block into a complete media frame return object.
        """
        selected_voice = voice or "af_bella"
        return LocalKokoroSynthesizedAudio(self._kokoro, text, selected_voice)


class LocalKokoroSynthesizedAudio(tts.SynthesizedAudio):
    def __init__(self, kokoro, text: str, voice: str):
        self._kokoro = kokoro
        self._text = text
        self._voice = voice

    async def __aiter__(self) -> AsyncIterator[tts.SynthesizedChunks]:
        """
        Iterates over generated audio chunks. Transforms Kokoro's native 
        24kHz float32 arrays into standard LiveKit 24kHz 16-bit linear PCM arrays.
        """
        loop = asyncio.get_running_loop()
        
        # Offload high-density audio wave synthesis to executor threads
        samples, sample_rate = await loop.run_in_executor(
            None, 
            lambda: self._kokoro.create(self._text, voice=self._voice, speed=1.0)
        )
        
        # Convert raw float array outputs directly back to standard int16 linear PCM allocations
        pcm_data = (samples * 32767).astype(np.int16).tobytes()
        
        # Pack the raw data into an official compliant frame allocation block
        frame = rtc.AudioFrame(
            data=pcm_data,
            sample_rate=24000, 
            num_channels=1,
            samples_per_channel=len(pcm_data) // 2
        )
        
        yield tts.SynthesizedChunks(text=self._text, frame=frame)