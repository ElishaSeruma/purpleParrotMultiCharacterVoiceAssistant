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

# Local structural imports
from purpleParrotMultiCharacterVoiceAssistant.services.kami_brain import KamiBrain
from purpleParrotMultiCharacterVoiceAssistant.services.phonetic_analyzer import SubSurfacePhoneticAnalyzer

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