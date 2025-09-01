import os
import logging
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import aiohttp
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MURF_API_KEY = os.getenv("MURF_API_KEY", "your-murf-api-key")
MURF_API_URL = "https://api.murf.ai/v1/speech/generate"
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "your-assemblyai-key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key")

# Mock implementation for google.generativeai
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Using real google.generativeai module")
except ImportError:
    logger.warning("google.generativeai not found, using mock implementation")
    
    # Create a mock module
    class MockGenerativeModel:
        def generate_content(self, prompt, stream=False):
            logger.info(f"Mock generate_content called with: {prompt}")
            
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            
            # Return a response about capitals
            if "uttar pradesh" in prompt.lower() or "up" in prompt.lower():
                return [MockResponse("The capital of Uttar Pradesh is Lucknow.")]
            elif "india" in prompt.lower():
                return [MockResponse("The capital of India is New Delhi.")]
            else:
                return [MockResponse("I understand you're asking about capitals. I know about Uttar Pradesh (Lucknow) and India (New Delhi).")]

    # Create mock functions
    class MockGoogleGenerativeAI:
        def GenerativeModel(self, model_name):
            logger.info(f"Creating mock GenerativeModel: {model_name}")
            return MockGenerativeModel()
    
    # Create mock module
    import sys
    from types import ModuleType
    
    genai_module = ModuleType('google.generativeai')
    genai_module.GenerativeModel = MockGoogleGenerativeAI().GenerativeModel
    sys.modules['google.generativeai'] = genai_module
    genai = genai_module

# WebSocket connections tracking
active_connections = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo the message back to the client
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")

@app.websocket("/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("New audio WebSocket connection established")
    
    try:
        while True:
            # Wait for any message from client
            data = await websocket.receive()
            
            # Handle text messages (JSON)
            if "text" in data:
                message = json.loads(data["text"])
                logger.info(f"Received message: {message}")
                
                if message.get("type") == "start":
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "status",
                        "message": "Recording started"
                    })
                    
                    # Simulate processing
                    await asyncio.sleep(1)
                    
                    # Send a mock response
                    await websocket.send_json({
                        "type": "transcript",
                        "text": "I heard you ask about capitals",
                        "is_final": False
                    })
                    
                    await asyncio.sleep(1)
                    
                    # Send final response
                    await websocket.send_json({
                        "type": "transcript",
                        "text": "The capital of Uttar Pradesh is Lucknow.",
                        "is_final": True
                    })
            
            # Handle binary data (audio)
            elif "bytes" in data:
                logger.info(f"Received binary data: {len(data['bytes'])} bytes")
                # Simulate processing audio
                await asyncio.sleep(0.5)
                
                # Send processing status
                await websocket.send_json({
                    "type": "status", 
                    "message": "Processing audio..."
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.remove(websocket)

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connections": len(active_connections)}

@app.post("/generate-audio")
async def generate_audio(text: str):
    """Generate audio using Murf TTS API"""
    try:
        if not text:
            return {"error": "Please enter some text to convert to speech."}
        
        # Prepare Murf API request
        headers = {
            'api-key': MURF_API_KEY,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'text': text,
            'voiceId': 'en-US-natalie',  # Valid Murf voice ID
            'format': 'MP3'
        }
        
        # Make request to Murf API
        async with aiohttp.ClientSession() as session:
            async with session.post(MURF_API_URL, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    audio_data = await response.json()
                    audio_url = audio_data.get('audioFile', '')
                    
                    if audio_url:
                        return {"success": True, "audio_url": audio_url}
                    else:
                        return {"error": "Audio generation failed: No audio URL returned."}
                else:
                    error_text = await response.text()
                    logger.error(f"Murf API error: {response.status} - {error_text}")
                    return {"error": f"Audio generation failed: API returned status {response.status}"}
            
    except asyncio.TimeoutError:
        return {"error": "Audio generation timed out. Please try again."}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)