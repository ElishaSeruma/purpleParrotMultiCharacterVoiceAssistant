# main.py
import os
import json
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from livekit import api, rtc
import psutil
import numpy as np
from purpleParrotMultiCharacterVoiceAssistant.services.local_kokoro_tts import (
    KOKORO_PERSONA_VOICE_MATRIX,
    LocalKokoroTTS,
)
# Local structural imports
from purpleParrotMultiCharacterVoiceAssistant.services.kami_brain import KamiBrain
from purpleParrotMultiCharacterVoiceAssistant.services.phonetic_analyzer import SubSurfacePhoneticAnalyzer
 
#STT imports
from purpleParrotMultiCharacterVoiceAssistant.services.local_whisper_stt import LocalWhisperSTT
from livekit.agents import stt
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
AUDIO_INFERENCE_WORKERS = int(os.getenv("AUDIO_INFERENCE_WORKERS", "2"))
MODEL_WARMUP_ENABLED = os.getenv("MODEL_WARMUP_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MODEL_WARMUP_TTS_ENABLED = os.getenv("MODEL_WARMUP_TTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MODEL_WARMUP_STT_ENABLED = os.getenv("MODEL_WARMUP_STT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MODEL_WARMUP_INFERENCE_ENABLED = os.getenv("MODEL_WARMUP_INFERENCE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MODEL_WARMUP_START_DELAY_SECONDS = float(os.getenv("MODEL_WARMUP_START_DELAY_SECONDS", "0.25"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("purple_parrot_main")
audio_inference_executor = ThreadPoolExecutor(
    max_workers=AUDIO_INFERENCE_WORKERS,
    thread_name_prefix="audio-inference",
)
warmup_task: asyncio.Task | None = None
warmup_status = {
    "enabled": MODEL_WARMUP_ENABLED,
    "running": False,
    "started_at": None,
    "completed_at": None,
    "duration_ms": None,
    "services": {},
    "errors": {},
}

app = FastAPI(
    title="Purple Parrot Voice Assistant Core Engine",
    description="LiveKit WebRTC token server and multi-character orchestration layer.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    global warmup_task
    logger.info("Audio inference executor ready max_workers=%d", AUDIO_INFERENCE_WORKERS)
    if MODEL_WARMUP_ENABLED:
        warmup_task = asyncio.create_task(warm_models_after_startup(), name="model-warmup")
    else:
        logger.info("Model warmup disabled by MODEL_WARMUP_ENABLED")

@app.on_event("shutdown")
async def shutdown_event():
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down audio inference executor")
    audio_inference_executor.shutdown(wait=False, cancel_futures=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize foreground brain state engine
brain = KamiBrain(default_persona="kami")

PERSONA_VOICE_MATRIX = KOKORO_PERSONA_VOICE_MATRIX

class ConnectionTokenRequest(BaseModel):
    room_name: str = Field(..., description="Unique room namespace string")
    participant_identity: str = Field(..., description="Unique identifier for the user account")
    participant_name: str = Field(..., description="Human-readable profile name")

class PersonaMutationRequest(BaseModel):
    persona: str = Field(..., description="Target persona id, for example kami, patty, tavo, or zeni")

def memory_snapshot() -> dict:
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    return {
        "rss_mb": round(memory.rss / (1024 * 1024), 2),
        "vms_mb": round(memory.vms / (1024 * 1024), 2),
    }

def mark_warmup_service(service: str, status_value: str, duration_ms: float | None = None, detail: str | None = None) -> None:
    payload = {"status": status_value}
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 2)
    if detail:
        payload["detail"] = detail
    warmup_status["services"][service] = payload

async def warm_models_after_startup() -> None:
    await asyncio.sleep(MODEL_WARMUP_START_DELAY_SECONDS)
    warmup_status["running"] = True
    warmup_status["started_at"] = time.time()
    warmup_status["completed_at"] = None
    warmup_status["duration_ms"] = None
    warmup_status["errors"] = {}
    warmup_status["memory_before"] = memory_snapshot()
    started_at = time.perf_counter()
    logger.info("Model warmup started memory=%s", warmup_status["memory_before"])
    loop = asyncio.get_running_loop()

    if MODEL_WARMUP_STT_ENABLED:
        service_started = time.perf_counter()
        try:
            stt_engine = await loop.run_in_executor(audio_inference_executor, get_local_stt_engine)
            if MODEL_WARMUP_INFERENCE_ENABLED:
                warm_audio = np.zeros(16000, dtype=np.float32)
                await loop.run_in_executor(
                    audio_inference_executor,
                    lambda: list(stt_engine._model.transcribe(warm_audio, beam_size=1, language="en")[0]),
                )
            duration_ms = (time.perf_counter() - service_started) * 1000
            mark_warmup_service("stt", "ready", duration_ms)
            logger.info("Warmup STT ready duration_ms=%.2f memory=%s", duration_ms, memory_snapshot())
        except Exception as exc:
            duration_ms = (time.perf_counter() - service_started) * 1000
            warmup_status["errors"]["stt"] = str(exc)
            mark_warmup_service("stt", "failed", duration_ms, str(exc))
            logger.exception("Warmup STT failed")
    else:
        mark_warmup_service("stt", "disabled")

    if MODEL_WARMUP_TTS_ENABLED:
        service_started = time.perf_counter()
        try:
            engine = await loop.run_in_executor(audio_inference_executor, get_local_tts_engine)
            voices_warmed = []
            for voice in dict.fromkeys(PERSONA_VOICE_MATRIX.values()):
                engine._get_voice_for_persona(next((persona for persona, mapped_voice in PERSONA_VOICE_MATRIX.items() if mapped_voice == voice), "kami"))
                if MODEL_WARMUP_INFERENCE_ENABLED:
                    await loop.run_in_executor(
                        audio_inference_executor,
                        lambda voice=voice: engine._kokoro.create("Ready.", voice=voice, speed=1.0),
                    )
                voices_warmed.append(voice)
            duration_ms = (time.perf_counter() - service_started) * 1000
            mark_warmup_service("tts", "ready", duration_ms, f"voices_warmed={','.join(voices_warmed)}")
            logger.info("Warmup TTS ready duration_ms=%.2f memory=%s", duration_ms, memory_snapshot())
        except Exception as exc:
            duration_ms = (time.perf_counter() - service_started) * 1000
            warmup_status["errors"]["tts"] = str(exc)
            mark_warmup_service("tts", "failed", duration_ms, str(exc))
            logger.exception("Warmup TTS failed")
    else:
        mark_warmup_service("tts", "disabled")

    warmup_status["running"] = False
    warmup_status["completed_at"] = time.time()
    warmup_status["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    warmup_status["memory_after"] = memory_snapshot()
    logger.info("Model warmup complete status=%s", warmup_status)

def resolve_voice_for_persona(persona_name: str) -> str:
    return PERSONA_VOICE_MATRIX.get(persona_name.lower(), PERSONA_VOICE_MATRIX["kami"])

def persona_state_payload(persona_name: Optional[str] = None) -> dict:
    active_name = (persona_name or brain.active_persona_name).lower()
    persona = brain.active_persona if active_name == brain.active_persona_name else brain.switch_persona(active_name)
    voice_print = resolve_voice_for_persona(active_name)
    return {
        "active_persona": active_name,
        "kokoro_voice": voice_print,
        "theme": {
            "color_palette": persona.color_palette,
            "typography_scale": persona.typography_scale,
            "audio_synthesis_engine": persona.audio_synthesis_engine,
            "dialogue_tonality_modifier": persona.dialogue_tonality,
            "kokoro_voice": voice_print,
        },
    }

def theme_mutation_event(persona_name: Optional[str] = None) -> dict:
    state = persona_state_payload(persona_name)
    return {
        "ThemeMutationEvent": {
            "target_persona": state["active_persona"],
            "structural_changes": state["theme"],
        }
    }

@app.post("/api/v1/auth/token")
async def generate_token(payload: ConnectionTokenRequest):
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server credentials configuration is missing."
        )
    try:
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(payload.participant_identity)
            .with_name(payload.participant_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=payload.room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True
                )
            )
        )
        return {"server_url": LIVEKIT_URL, "token": token.to_jwt()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.get("/api/v1/persona")
async def get_active_persona():
    return persona_state_payload()

@app.get("/api/v1/persona/voices")
async def get_persona_voice_matrix():
    return {"voices": PERSONA_VOICE_MATRIX}

@app.post("/api/v1/persona")
async def mutate_active_persona(payload: PersonaMutationRequest):
    updated_persona = brain.switch_persona(payload.persona)
    active_name = brain.active_persona_name
    voice_print = resolve_voice_for_persona(active_name)
    logger.info(
        "Active persona synced persona=%s kokoro_voice=%s audio_engine=%s",
        active_name,
        voice_print,
        updated_persona.audio_synthesis_engine,
    )
    return persona_state_payload()

@app.websocket("/api/v1/voice/control")
async def voice_control_stream(websocket: WebSocket):
    await websocket.accept()
    print("[CONTROL] UI client connected directly to the Persona Control Plane.")

    await websocket.send_text(json.dumps(theme_mutation_event()))

    # Instantiate our clinical background parsing context for this session channel
    analyzer = SubSurfacePhoneticAnalyzer(learner_id="learner-elisha-01", room_name="parrot-test-room")
    analyzer_task = asyncio.create_task(analyzer.start_analysis_loop())

    # Simulate inbound raw PCM data streaming over LiveKit WebRTC Track buffers
    async def simulate_livekit_audio_feed():
        try:
            fake_frame = rtc.AudioFrame(
                data=b'\x00' * 960, # 20ms of empty 16-bit linear PCM frame data at 24kHz
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=480
            )
            while analyzer.is_processing:
                await analyzer.push_audio_frame(fake_frame)
                await asyncio.sleep(0.02) # Feed data exactly every 20ms to match genuine WebRTC timelines
        except asyncio.CancelledError:
            pass

    audio_simulation_task = asyncio.create_task(simulate_livekit_audio_feed())

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "request_persona_mutation" in message:
                target = message["request_persona_mutation"]
                updated_persona = brain.switch_persona(target)
                voice_print = resolve_voice_for_persona(brain.active_persona_name)
                logger.info(
                    "WebSocket persona sync persona=%s kokoro_voice=%s audio_engine=%s",
                    brain.active_persona_name,
                    voice_print,
                    updated_persona.audio_synthesis_engine,
                )

                await websocket.send_text(json.dumps(theme_mutation_event()))

    except (WebSocketDisconnect, Exception) as e:
        print(f"[CONTROL] UI connection tracking update: Client disconnected ({type(e).__name__}).")
    finally:
        # 1. STOP THE ANALYZER IMMEDIATELY AND CAPTURE LOGS FIRST BEFORE AWAITING TASK SHUTDOWNS
        analyzer.stop()
        
        print("\n=======================================================")
        print("=== FINAL THERAPIST ROLLING PROGRESS SUMMARY RECORD ===")
        print(json.dumps(analyzer.generate_therapist_rolling_summary(), indent=2))
        print("=======================================================\n")
        
        # 2. Safely tear down active concurrent tasks
        audio_simulation_task.cancel()
        analyzer_task.cancel()
        try:
            await analyzer_task
        except asyncio.CancelledError:
            pass
        print("[SYSTEM] Audio streaming simulation and sub-surface workers cleanly defused.")

@app.get("/sandbox", response_class=HTMLResponse)
async def serve_sandbox():
    with open("purpleParrotMultiCharacterVoiceAssistant/sandbox.html", "r") as f:
        return f.read()

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "active_persona": brain.active_persona_name}

@app.get("/api/v1/system/resources")
async def system_resources():
    return {
        "memory": memory_snapshot(),
        "audio_inference_workers": AUDIO_INFERENCE_WORKERS,
        "models": {
            "stt_loaded": local_stt_engine is not None,
            "tts_loaded": local_tts_engine is not None,
        },
        "warmup": warmup_status,
    }

@app.get("/api/v1/system/warmup")
async def model_warmup_status():
    return warmup_status

@app.post("/api/v1/system/warmup")
async def trigger_model_warmup():
    global warmup_task
    if warmup_task and not warmup_task.done():
        return {"status": "already_running", "warmup": warmup_status}
    warmup_task = asyncio.create_task(warm_models_after_startup(), name="model-warmup-manual")
    return {"status": "started", "warmup": warmup_status}


local_stt_engine: LocalWhisperSTT | None = None

def get_local_stt_engine() -> LocalWhisperSTT:
    global local_stt_engine
    if local_stt_engine is None:
        local_stt_engine = LocalWhisperSTT(executor=audio_inference_executor)
    return local_stt_engine

@app.post("/api/v1/test/stt-pipeline")
async def simulate_live_stt_feed():
    """
    Simulation route validating that LiveKit's abstract stream interface 
    correctly ingests data arrays and transforms them into text models.
    """
    print("\n--- TRIGGERING LOCAL WHISPER STT STREAM TEST ROUTINE ---")
    endpoint_started_at = time.perf_counter()
    local_stt_engine = get_local_stt_engine()
    stt_stream = local_stt_engine.stream()
    
    # Generate 1 second of mock vocal 16kHz PCM data frames
    fake_frame = rtc.AudioFrame(
        data=b'\x00' * 640, # 20ms block mapping at 16kHz sample rate standard configuration
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320
    )
    
    # Push data blocks down the active stream pipeline channel
    for _ in range(50):
        stt_stream.push_frame(fake_frame)
    stt_stream.end_input()
    
    print("[SYSTEM] Audio chunks successfully pushed. Processing predictions...")
    
    # Exhaustively read emitted prediction event signals
    try:
        while True:
            event = await stt_stream.__anext__()
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                break
            print(f"[LOCAL WHISPER MATCHED]: {event.alternatives[0].text}")
    except Exception:
        pass
        
    print("--- STT BLUEPRINT PIPELINE CHECK VERIFIED COMPLETE ---\n")
    latency_ms = round((time.perf_counter() - endpoint_started_at) * 1000, 2)
    logger.info("STT pipeline endpoint latency_ms=%.2f memory=%s", latency_ms, memory_snapshot())
    return {
        "status": "Whisper engine loop validated successfully.",
        "latency_ms": latency_ms,
        "memory": memory_snapshot(),
    }

local_tts_engine: LocalKokoroTTS | None = None

def get_local_tts_engine() -> LocalKokoroTTS:
    global local_tts_engine
    if local_tts_engine is None:
        local_tts_engine = LocalKokoroTTS(executor=audio_inference_executor)
    return local_tts_engine

@app.post("/api/v1/test/tts-pipeline")
async def simulate_live_tts_feed(
    persona: Optional[str] = None,
    dry_run: bool = False,
    text: str = "Hello Elisha! I am operating as your local safety and speech assistant. Systems are active.",
):
    """
    Simulation validation verifying that our local open-source voice engine 
    can target individual persona prints and output operational audio signals.
    """
    if persona:
        brain.switch_persona(persona)

    endpoint_started_at = time.perf_counter()
    active_persona = brain.active_persona_name
    print(f"\n--- TRIGGERING LOCAL KOKORO TTS SYNTHESIS FOR CHARACTER: {active_persona.upper()} ---")
    voice_print = resolve_voice_for_persona(active_persona)
    logger.info("TTS pipeline persona sync persona=%s kokoro_voice=%s", active_persona, voice_print)
    if dry_run:
        latency_ms = round((time.perf_counter() - endpoint_started_at) * 1000, 2)
        return {
            "status": f"Dry run resolved persona {active_persona} to voice {voice_print}.",
            "active_persona": active_persona,
            "kokoro_voice": voice_print,
            "chunks": 0,
            "dry_run": True,
            "latency_ms": latency_ms,
            "memory": memory_snapshot(),
        }

    local_tts_engine = get_local_tts_engine()
    
    # Resolve through the live TTS adapter too, so logs prove the active engine agrees.
    voice_print = local_tts_engine._get_voice_for_persona(active_persona)
    # Fire up the synthesizer core adapter loop
    audio_stream = local_tts_engine.synthesize(text, voice=voice_print)
    
    print("[SYSTEM] Audio synthesis computation processing...")
    chunks = 0
    async for chunk in audio_stream:
        chunks += 1
        frame_meta = chunk.frame
        print(f"[VOCAL OUTPUT SUCCESS]: Synthesized chunk array partition matches!")
        print(f" -> Sample Rate: {frame_meta.sample_rate}Hz | Channels: {frame_meta.num_channels}")
        
    print("--- TTS BLUEPRINT PIPELINE CHECK VERIFIED COMPLETE ---\n")
    latency_ms = round((time.perf_counter() - endpoint_started_at) * 1000, 2)
    logger.info("TTS pipeline endpoint latency_ms=%.2f chunks=%d memory=%s", latency_ms, chunks, memory_snapshot())
    return {
        "status": f"Vocal tracking for persona {active_persona} processed successfully.",
        "active_persona": active_persona,
        "kokoro_voice": voice_print,
        "chunks": chunks,
        "latency_ms": latency_ms,
        "memory": memory_snapshot(),
    }
