# import os
# import logging
# import requests
# from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Create the app
# app = Flask(__name__)
# app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key")

# # Murf API configuration
# MURF_API_KEY = os.getenv("MURF_API_KEY", "default_murf_key")
# MURF_API_URL = "https://api.murf.ai/v1/speech/generate"

# @app.route('/')
# def index():
#     """Main page with TTS interface"""
#     return render_template('index.html')

# @app.route('/generate-audio', methods=['POST'])
# def generate_audio():
#     """Generate audio using Murf TTS API"""
#     try:
#         # Get text from form
#         text = request.form.get('text', '').strip()
        
#         if not text:
#             flash('Please enter some text to convert to speech.', 'error')
#             return redirect(url_for('index'))
        
#         # Prepare Murf API request
#         headers = {
#             'api-key': MURF_API_KEY,
#             'Content-Type': 'application/json'
#         }
        
#         payload = {
#             'text': text,
#             'voiceId': 'en-US-natalie',  # Valid Murf voice ID
#             'format': 'MP3'
#         }
        
#         # Make request to Murf API
#         app.logger.debug(f"Making request to Murf API with text: {text}")
#         response = requests.post(MURF_API_URL, json=payload, headers=headers, timeout=30)
        
#         if response.status_code == 200:
#             audio_data = response.json()
#             audio_url = audio_data.get('audioFile', '')
            
#             if audio_url:
#                 flash('Audio generated successfully!', 'success')
#                 return render_template('index.html', audio_url=audio_url, text=text)
#             else:
#                 flash('Audio generation failed: No audio URL returned.', 'error')
#                 return redirect(url_for('index'))
#         else:
#             app.logger.error(f"Murf API error: {response.status_code} - {response.text}")
#             flash(f'Audio generation failed: API returned status {response.status_code}', 'error')
#             return redirect(url_for('index'))
            
#     except requests.exceptions.Timeout:
#         flash('Audio generation timed out. Please try again.', 'error')
#         return redirect(url_for('index'))
#     except requests.exceptions.RequestException as e:
#         app.logger.error(f"Request error: {str(e)}")
#         flash('Network error occurred. Please check your connection and try again.', 'error')
#         return redirect(url_for('index'))
#     except Exception as e:
#         app.logger.error(f"Unexpected error: {str(e)}")
#         flash('An unexpected error occurred. Please try again.', 'error')
#         return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)     


import asyncio
import websockets
import json
import base64
import uuid
from typing import AsyncGenerator

class MurfTTSClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://api.murf.ai/v1/speech/stream"
        self.context_id = str(uuid.uuid4())  # Static context ID
        self.voice_id = "en_us_001"  # Default voice, change as needed
        self.sample_rate = 24000
        self.audio_format = "mp3"
        
    async def get_llm_response_stream(self) -> AsyncGenerator[str, None]:
        """
        Replace this with your actual LLM streaming implementation.
        This is a mock function that simulates an LLM response.
        """
        # Simulated LLM response chunks
        response_chunks = [
            "Hello, welcome to our service!",
            "How can I assist you today?",
            "Please let me know if you need any help."
        ]
        
        for chunk in response_chunks:
            yield chunk
            await asyncio.sleep(0.5)  # Simulate delay between chunks
            
    async def connect_to_murf(self):
        """Establish WebSocket connection to Murf and handle communication"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with websockets.connect(self.ws_url, extra_headers=headers) as websocket:
                print(f"Connected to Murf TTS with context_id: {self.context_id}")
                
                # Send configuration message
                config_msg = {
                    "action": "configure",
                    "voice_id": self.voice_id,
                    "sample_rate": self.sample_rate,
                    "format": self.audio_format,
                    "context_id": self.context_id
                }
                
                await websocket.send(json.dumps(config_msg))
                print("Sent configuration to Murf")
                
                # Process LLM stream
                async for text_chunk in self.get_llm_response_stream():
                    print(f"Sending text to Murf: {text_chunk}")
                    
                    # Send text synthesis request
                    synth_msg = {
                        "action": "synthesize",
                        "text": text_chunk,
                        "context_id": self.context_id
                    }
                    
                    await websocket.send(json.dumps(synth_msg))
                    
                    # Wait for audio response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        response_data = json.loads(response)
                        
                        if response_data.get("type") == "audio" and "data" in response_data:
                            audio_base64 = response_data["data"]
                            print(f"Received audio (base64): {audio_base64[:100]}...")
                            
                            # Optional: decode and save audio
                            # audio_data = base64.b64decode(audio_base64)
                            # with open(f"output_{int(time.time())}.mp3", "wb") as f:
                            #     f.write(audio_data)
                            
                    except asyncio.TimeoutError:
                        print("Timeout waiting for audio response")
                        continue
                    except json.JSONDecodeError:
                        print("Failed to parse JSON response")
                        continue
                
                # End the stream
                end_msg = {
                    "action": "end",
                    "context_id": self.context_id
                }
                await websocket.send(json.dumps(end_msg))
                print("Sent end of stream message")
                
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error in WebSocket communication: {e}")

async def main():
    # Replace with your actual Murf API key
    api_key = "your_murf_api_key_here"
    
    if api_key == "your_murf_api_key_here":
        print("Please set your Murf API key in the code")
        return
        
    murf_client = MurfTTSClient(api_key)
    await murf_client.connect_to_murf()

if __name__ == "__main__":
    # Install required packages: pip install websockets asyncio
    asyncio.run(main())