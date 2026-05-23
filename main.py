# main.py
import os
import json
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from livekit import api

# Local structural imports
from purpleParrotMultiCharacterVoiceAssistant.services.kami_brain import KamiBrain

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

# Initialize our core state engine
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
    """
    Persistent real-time bidirectional messaging link handling active persona hot-swaps
    and broadcasting structural UI mutation payloads.
    """
    await websocket.accept()
    print("UI client connected directly to the Persona Control Plane.")
    
    # Broadcast current default state immediately upon connecting
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

    try:
        while True:
            # Await input execution commands from client interface toggles
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "request_persona_mutation" in message:
                target = message["request_persona_mutation"]
                # Mutate the server-side brain configuration
                updated_persona = brain.switch_persona(target)
                
                # Format complete architectural System 4.2 payload response
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
                
                # Emit system instruction update visualization logs to terminal
                print(f"=== UPDATED SYSTEM PROMPT COMPILATION ({target}) ===")
                print(brain.compile_system_instructions())
                print("=====================================================")
                
                # Sync mutation data instantly straight back to UI client
                await websocket.send_text(json.dumps(mutation_event))

    except WebSocketDisconnect:
        print("UI client severed the control connection pipeline.")

@app.get("/sandbox", response_class=HTMLResponse)
async def serve_sandbox():
    with open("purpleParrotMultiCharacterVoiceAssistant/sandbox.html", "r") as f:
        return f.read()

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "active_persona": brain.active_persona_name}