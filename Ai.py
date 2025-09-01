# import sys
# import time
# import threading
# from flask import Flask, render_template_string, Response
# import queue

# app = Flask(__name__)

# # Simulated transcript from AssemblyAI
# FINAL_TRANSCRIPT = """
# Hello! I'm calling to inquire about the status of my recent order, number 45678. 
# I placed it about five days ago and was wondering when I can expect it to be delivered. 
# Also, I wanted to ask if it's possible to change the delivery address since I'll be traveling next week. 
# Thank you for your assistance!
# """

# # Simulated LLM response chunks
# LLM_RESPONSE_CHUNKS = [
#     "Thank you for reaching out about your order.",
#     " I've looked up order #45678 in our system.",
#     " It's currently being processed and is scheduled for shipment tomorrow.",
#     " Based on our standard delivery times, you should receive it within 2-3 business days.",
#     " Regarding the delivery address change, yes, we can update that for you.",
#     " Please provide your new address details and I'll make the necessary changes.",
#     " Is there anything else I can help you with today?"
# ]

# # Global queue for streaming data
# response_queue = queue.Queue()

# def simulate_llm_streaming():
#     """Simulate receiving chunks from an LLM API"""
#     for chunk in LLM_RESPONSE_CHUNKS:
#         time.sleep(0.5)  # Simulate network delay
#         response_queue.put(chunk)
#     response_queue.put(None)  # Signal end of stream

# @app.route('/')
# def index():
#     return render_template_string('''
# <!DOCTYPE html>
# <html>
# <head>
#     <title>LLM Streaming Demo</title>
#     <style>
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             max-width: 800px;
#             margin: 0 auto;
#             padding: 20px;
#             background-color: #f5f7f9;
#             color: #333;
#         }
#         .container {
#             background: white;
#             border-radius: 10px;
#             padding: 20px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         }
#         h1 {
#             color: #2c3e50;
#             text-align: center;
#         }
#         .transcript, .response {
#             background: #f8f9fa;
#             border-left: 4px solid #3498db;
#             padding: 15px;
#             margin: 15px 0;
#             border-radius: 0 5px 5px 0;
#         }
#         .response {
#             border-left-color: #2ecc71;
#             min-height: 100px;
#         }
#         .btn {
#             background: #3498db;
#             color: white;
#             border: none;
#             padding: 10px 15px;
#             border-radius: 5px;
#             cursor: pointer;
#             font-size: 16px;
#             display: block;
#             margin: 20px auto;
#             transition: background 0.3s;
#         }
#         .btn:hover {
#             background: #2980b9;
#         }
#         .streaming-indicator {
#             display: inline-block;
#             width: 10px;
#             height: 10px;
#             border-radius: 50%;
#             background: #e74c3c;
#             margin-right: 5px;
#             animation: pulse 1.5s infinite;
#         }
#         @keyframes pulse {
#             0% { opacity: 1; }
#             50% { opacity: 0.4; }
#             100% { opacity: 1; }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>LLM Streaming Response Demo</h1>
        
#         <div class="transcript">
#             <h3>Final Transcript from AssemblyAI:</h3>
#             <p>{{ transcript }}</p>
#         </div>
        
#         <button class="btn" onclick="startStreaming()">Send to LLM and Stream Response</button>
        
#         <div class="response">
#             <h3>LLM Response: <span id="status"></span></h3>
#             <div id="llm-response"></div>
#         </div>
#     </div>

#     <script>
#         function startStreaming() {
#             document.getElementById('status').innerHTML = '<span class="streaming-indicator"></span>Streaming...';
#             document.getElementById('llm-response').innerHTML = '';
            
#             const eventSource = new EventSource('/stream');
#             let accumulatedText = '';
            
#             eventSource.onmessage = function(event) {
#                 if (event.data === 'END_OF_STREAM') {
#                     eventSource.close();
#                     document.getElementById('status').innerHTML = 'Complete';
#                     return;
#                 }
                
#                 accumulatedText += event.data;
#                 document.getElementById('llm-response').innerHTML = accumulatedText;
                
#                 // Auto-scroll to bottom
#                 const responseDiv = document.getElementById('llm-response');
#                 responseDiv.scrollTop = responseDiv.scrollHeight;
#             };
            
#             eventSource.onerror = function() {
#                 eventSource.close();
#                 document.getElementById('status').innerHTML = 'Stream ended';
#             };
#         }
#     </script>
# </body>
# </html>
#     ''', transcript=FINAL_TRANSCRIPT)

# @app.route('/stream')
# def stream():
#     def generate():
#         # Start the LLM simulation in a separate thread
#         thread = threading.Thread(target=simulate_llm_streaming)
#         thread.start()
        
#         # Stream the chunks as they become available
#         while True:
#             chunk = response_queue.get()
#             if chunk is None:
#                 yield "data: END_OF_STREAM\n\n"
#                 break
#             yield f"data: {chunk}\n\n"
    
#     return Response(generate(), mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True) 