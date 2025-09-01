import os
import google.generativeai as genai
from tavily import TavilyClient
import pygame
import time
from gtts import gTTS
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Voice Agent with Web Search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------
# 1. Configure API Clients (SECURE VERSION)
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY and TAVILY_API_KEY environment variables")

genai.configure(api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize pygame for audio playback
pygame.mixer.init()

# ----------------------------
# 2. Text-to-Speech Function
# ----------------------------
def text_to_speech(text, lang='en'):
    """Convert text to speech and play it"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# ----------------------------
# 3. Define the Special Skill Function
# ----------------------------
def search_web(query):
    """
    Performs a web search for a given query and returns relevant results.
    This is the special skill we are adding to our agent!
    """
    print(f"\n🤖 Agent is using its special skill: Searching the web for '{query}'...")
    
    try:
        # Use the Tavily client to search
        search_result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=3
        )
        return search_result.get('results', [])
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ----------------------------
# 4. Define the Function for Gemini to Call
# ----------------------------
web_search_tool = {
    "function_declarations": [
        {
            "name": "search_web",
            "description": "Searches the web for recent information on a given query. Use this when asked about recent events, news, weather, or any topic where real-time data is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up, optimized for search engines."
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

# ----------------------------
# 5. Create the Agent with the Tool
# ----------------------------
def initialize_agent():
    """Initialize the Gemini model with web search capability"""
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        tools=[web_search_tool]
    )
    
    # Start a chat session with the tool enabled
    chat = model.start_chat(
        enable_automatic_function_calling=True,
    )
    
    return chat

# Initialize the agent
chat = initialize_agent()

# Pydantic model for request body
class ChatRequest(BaseModel):
    message: str
    use_voice: bool = True

# ----------------------------
# 6. API Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {"message": "AI Voice Agent with Web Search API"}

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        response = chat.send_message(request.message)
        
        # If voice output is requested, play it
        if request.use_voice:
            # Run TTS in a separate thread to avoid blocking
            threading.Thread(target=text_to_speech, args=(response.text,)).start()
        
        return {
            "response": response.text,
            "success": True
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Voice Agent"}

# ----------------------------
# 7. Main function for standalone execution
# ----------------------------
def main():
    print("🌟 Day 25: AI Voice Agent with Web Search Skill 🌟")
    print("Your agent is now smarter! It can search the web for live information.")
    print("Type 'exit' to quit.\n")
    
    # Greeting
    greeting = "Hello! I'm your AI assistant with web search capabilities. How can I help you today?"
    print(f"Agent: {greeting}")
    text_to_speech(greeting)
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            farewell = "Goodbye! It was great assisting you."
            print(f"Agent: {farewell}")
            text_to_speech(farewell)
            break
        
        # Send the message to the model
        try:
            response = chat.send_message(user_input)
            
            # Print and speak the final response from the agent
            print(f"\nAgent: {response.text}\n")
            text_to_speech(response.text)
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            print(f"Agent: {error_msg}")
            text_to_speech("I'm sorry, I encountered an error processing your request.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)