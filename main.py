from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

html = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WebSocket Chat - Dark Edition</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                background: #000000;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                min-height: 100vh;
                overflow-x: hidden;
                position: relative;
            }

            /* Animated background particles */
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                pointer-events: none;
            }

            .particle {
                position: absolute;
                width: 2px;
                height: 2px;
                background: #00ff88;
                border-radius: 50%;
                animation: float 6s ease-in-out infinite;
                opacity: 0.7;
            }

            @keyframes float {
                0%, 100% {
                    transform: translateY(0px) rotate(0deg);
                    opacity: 0.7;
                }
                50% {
                    transform: translateY(-20px) rotate(180deg);
                    opacity: 1;
                }
            }

            /* Glowing gradient background */
            .bg-gradient {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at 20% 50%, rgba(0, 255, 136, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 80% 20%, rgba(0, 136, 255, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 40% 80%, rgba(255, 0, 136, 0.1) 0%, transparent 50%);
                z-index: -2;
                animation: gradientShift 8s ease-in-out infinite;
            }

            @keyframes gradientShift {
                0%, 100% {
                    opacity: 0.3;
                    transform: scale(1);
                }
                50% {
                    opacity: 0.6;
                    transform: scale(1.1);
                }
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
                position: relative;
                z-index: 1;
            }

            h1 {
                text-align: center;
                font-size: 3rem;
                margin-bottom: 40px;
                background: linear-gradient(45deg, #00ff88, #0088ff, #ff0088);
                background-size: 200% 200%;
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientText 3s ease-in-out infinite;
                text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
            }

            @keyframes gradientText {
                0%, 100% {
                    background-position: 0% 50%;
                }
                50% {
                    background-position: 100% 50%;
                }
            }

            .chat-container {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                animation: slideUp 1s ease-out;
            }

            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .input-container {
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
                align-items: center;
            }

            #messageText {
                flex: 1;
                padding: 15px 20px;
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(0, 255, 136, 0.3);
                border-radius: 25px;
                color: #ffffff;
                font-size: 16px;
                outline: none;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
            }

            #messageText:focus {
                border-color: #00ff88;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
                transform: scale(1.02);
            }

            #messageText::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }

            .send-btn {
                padding: 15px 30px;
                background: linear-gradient(45deg, #00ff88, #0088ff);
                border: none;
                border-radius: 25px;
                color: #000000;
                font-weight: bold;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .send-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(0, 255, 136, 0.4);
            }

            .send-btn:active {
                transform: translateY(0);
            }

            .send-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                transition: left 0.5s;
            }

            .send-btn:hover::before {
                left: 100%;
            }

            .messages-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 20px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .messages-container::-webkit-scrollbar {
                width: 8px;
            }

            .messages-container::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }

            .messages-container::-webkit-scrollbar-thumb {
                background: linear-gradient(45deg, #00ff88, #0088ff);
                border-radius: 10px;
            }

            #messages {
                list-style: none;
            }

            .message-item {
                padding: 12px 18px;
                margin: 10px 0;
                background: rgba(255, 255, 255, 0.05);
                border-left: 4px solid #00ff88;
                border-radius: 10px;
                animation: messageSlide 0.5s ease-out;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .message-item:hover {
                background: rgba(255, 255, 255, 0.1);
                transform: translateX(5px);
            }

            .message-item::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: linear-gradient(45deg, #00ff88, #0088ff, #ff0088);
                animation: borderGlow 2s ease-in-out infinite;
            }

            @keyframes borderGlow {
                0%, 100% {
                    opacity: 0.7;
                }
                50% {
                    opacity: 1;
                    box-shadow: 0 0 10px #00ff88;
                }
            }

            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateX(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }

            .connection-status {
                text-align: center;
                margin-bottom: 20px;
                padding: 10px;
                border-radius: 10px;
                font-weight: bold;
                transition: all 0.3s ease;
            }

            .connected {
                background: rgba(0, 255, 136, 0.2);
                color: #00ff88;
                border: 1px solid rgba(0, 255, 136, 0.3);
            }

            .disconnected {
                background: rgba(255, 0, 136, 0.2);
                color: #ff0088;
                border: 1px solid rgba(255, 0, 136, 0.3);
            }

            .pulse {
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(0, 255, 136, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(0, 255, 136, 0);
                }
            }

            /* Responsive design */
            @media (max-width: 768px) {
                .container {
                    padding: 20px 15px;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .input-container {
                    flex-direction: column;
                    gap: 10px;
                }
                
                .send-btn {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="bg-gradient"></div>
        <div class="particles" id="particles"></div>
        
        <div class="container">
            <h1>WebSocket Chat</h1>
            
            <div class="chat-container">
                <div class="connection-status disconnected" id="connectionStatus">
                    🔴 Connecting...
                </div>
                
                <form action="" onsubmit="sendMessage(event)">
                    <div class="input-container">
                        <input type="text" id="messageText" placeholder="Type your message here..." autocomplete="off"/>
                        <button type="submit" class="send-btn">Send ✨</button>
                    </div>
                </form>
                
                <div class="messages-container">
                    <ul id='messages'></ul>
                </div>
            </div>
        </div>

        <script>
            // Create floating particles
            function createParticles() {
                const particlesContainer = document.getElementById('particles');
                const particleCount = 50;
                
                for (let i = 0; i < particleCount; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    particle.style.animationDelay = Math.random() * 6 + 's';
                    particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
                    particlesContainer.appendChild(particle);
                }
            }

            // WebSocket connection
            const connectionStatus = document.getElementById('connectionStatus');
            let ws;
            
            function connectWebSocket() {
                ws = new WebSocket("ws://localhost:8000/ws");
                
                ws.onopen = function(event) {
                    connectionStatus.textContent = '🟢 Connected';
                    connectionStatus.className = 'connection-status connected pulse';
                };
                
                ws.onmessage = function(event) {
                    const messages = document.getElementById('messages');
                    const message = document.createElement('li');
                    message.className = 'message-item';
                    message.textContent = event.data;
                    messages.appendChild(message);
                    
                    // Auto scroll to bottom
                    const container = document.querySelector('.messages-container');
                    container.scrollTop = container.scrollHeight;
                };
                
                ws.onclose = function(event) {
                    connectionStatus.textContent = '🔴 Disconnected';
                    connectionStatus.className = 'connection-status disconnected';
                    
                    // Try to reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = function(error) {
                    connectionStatus.textContent = '⚠️ Connection Error';
                    connectionStatus.className = 'connection-status disconnected';
                };
            }
            
            function sendMessage(event) {
                event.preventDefault();
                const input = document.getElementById("messageText");
                
                if (input.value.trim() && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(input.value);
                    input.value = '';
                    
                    // Add send animation
                    const sendBtn = document.querySelector('.send-btn');
                    sendBtn.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        sendBtn.style.transform = '';
                    }, 150);
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                createParticles();
                connectWebSocket();
                
                // Add enter key support
                document.getElementById('messageText').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage(e);
                    }
                });
            });
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo the message back to the client
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)