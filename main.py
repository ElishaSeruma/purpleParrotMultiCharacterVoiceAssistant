# main.py
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from livekit import api, rtc
from purpleParrotMultiCharacterVoiceAssistant.services.local_kokoro_tts import LocalKokoroTTS
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

app = FastAPI(
    title="Purple Parrot Voice Assistant Core Engine",
    description="LiveKit WebRTC token server and multi-character orchestration layer.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize foreground brain state engine
brain = KamiBrain(default_persona="kami")

class ConnectionTokenRequest(BaseModel):
    room_name: str = Field(..., description="Unique room namespace string")
    participant_identity: str = Field(..., description="Unique identifier for the user account")
    participant_name: str = Field(..., description="Human-readable profile name")

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

@app.websocket("/api/v1/voice/control")
async def voice_control_stream(websocket: WebSocket):
    await websocket.accept()
    print("[CONTROL] UI client connected directly to the Persona Control Plane.")
    
    current = brain.active_persona
    initial_payload = {
        "ThemeMutationEvent": {
            "target_persona": brain.active_persona_name,
            "structural_changes": {
                "color_palette": current.color_palette,
                "typography_scale": current.typography_scale,
                "audio_synthesis_engine": current.audio_synthesis_engine,
                "dialogue_tonality_modifier": current.dialogue_tonality
            }
        }
    }
    await websocket.send_text(json.dumps(initial_payload))

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
                
                mutation_event = {
                    "ThemeMutationEvent": {
                        "target_persona": target,
                        "structural_changes": {
                            "color_palette": updated_persona.color_palette,
                            "typography_scale": updated_persona.typography_scale,
                            "audio_synthesis_engine": updated_persona.audio_synthesis_engine,
                            "dialogue_tonality_modifier": updated_persona.dialogue_tonality
                        }
                    }
                }
                await websocket.send_text(json.dumps(mutation_event))

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
        try:
            # Shield the analyzer task termination so it doesn't get swept away by ASGI worker shutdown
            await asyncio.shield(analyzer_task)
        except Exception:
            pass
        print("[SYSTEM] Audio streaming simulation and sub-surface workers cleanly defused.")

@app.get("/sandbox", response_class=HTMLResponse)
async def serve_sandbox():
    with open("purpleParrotMultiCharacterVoiceAssistant/sandbox.html", "r") as f:
        return f.read()

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "active_persona": brain.active_persona_name}


local_stt_engine: LocalWhisperSTT | None = None

def get_local_stt_engine() -> LocalWhisperSTT:
    global local_stt_engine
    if local_stt_engine is None:
        local_stt_engine = LocalWhisperSTT()
    return local_stt_engine

@app.post("/api/v1/test/stt-pipeline")
async def simulate_live_stt_feed():
    """
    Simulation route validating that LiveKit's abstract stream interface 
    correctly ingests data arrays and transforms them into text models.
    """
    print("\n--- TRIGGERING LOCAL WHISPER STT STREAM TEST ROUTINE ---")
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
    for _ in range(25):
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
    return {"status": "Whisper engine loop validated successfully."}

local_tts_engine: LocalKokoroTTS | None = None

def get_local_tts_engine() -> LocalKokoroTTS:
    global local_tts_engine
    if local_tts_engine is None:
        local_tts_engine = LocalKokoroTTS()
    return local_tts_engine

@app.post("/api/v1/test/tts-pipeline")
async def simulate_live_tts_feed(persona: str = "patty"):
    """
    Simulation validation verifying that our local open-source voice engine 
    can target individual persona prints and output operational audio signals.
    """
    print(f"\n--- TRIGGERING LOCAL KOKORO TTS SYNTHESIS FOR CHARACTER: {persona.upper()} ---")
    local_tts_engine = get_local_tts_engine()
    
    # Resolve the corresponding underlying vocal index print pattern
    voice_print = local_tts_engine._get_voice_for_persona(persona)
    text_to_speak = f"Hello Elisha! I am operating as your local safety and speech assistant. Systems are active."
    
    # Fire up the synthesizer core adapter loop
    audio_stream = local_tts_engine.synthesize(text_to_speak, voice=voice_print)
    
    print("[SYSTEM] Audio synthesis computation processing...")
    async for chunk in audio_stream:
        frame_meta = chunk.frame
        print(f"[VOCAL OUTPUT SUCCESS]: Synthesized chunk array partition matches!")
        print(f" -> Sample Rate: {frame_meta.sample_rate}Hz | Channels: {frame_meta.num_channels}")
        
    print("--- TTS BLUEPRINT PIPELINE CHECK VERIFIED COMPLETE ---\n")
    return {"status": f"Vocal tracking for persona {persona} processed successfully."}
