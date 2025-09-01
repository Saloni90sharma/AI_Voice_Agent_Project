# import sys
# import random
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
#                              QHBoxLayout, QLabel, QTextEdit, QPushButton, 
#                              QFrame, QSplitter, QListWidget, QTabWidget)
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# class VoiceAgentApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("AI Voice Agent")
#         self.setGeometry(100, 100, 1000, 700)
        
#         # Central widget
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
        
#         # Main layout
#         main_layout = QHBoxLayout(central_widget)
        
#         # Left panel
#         left_panel = QFrame()
#         left_panel.setFrameStyle(QFrame.Box)
#         left_panel.setLineWidth(1)
#         left_panel.setMaximumWidth(250)
#         left_layout = QVBoxLayout(left_panel)
        
#         # Title
#         title = QLabel("AI Voice Agent")
#         title.setFont(QFont("Arial", 16, QFont.Bold))
#         title.setAlignment(Qt.AlignCenter)
#         left_layout.addWidget(title)
        
#         subtitle = QLabel("Your personal voice activated AI assistant.")
#         subtitle.setWordWrap(True)
#         subtitle.setAlignment(Qt.AlignCenter)
#         left_layout.addWidget(subtitle)
        
#         # Separator
#         separator = QFrame()
#         separator.setFrameShape(QFrame.HLine)
#         separator.setFrameShadow(QFrame.Sunken)
#         left_layout.addWidget(separator)
        
#         # Status
#         status_label = QLabel("You are a the capital of mind!")
#         status_label.setWordWrap(True)
#         left_layout.addWidget(status_label)
        
#         agent_label = QLabel("AI Agent")
#         agent_label.setFont(QFont("Arial", 12, QFont.Bold))
#         left_layout.addWidget(agent_label)
        
#         capital_label = QLabel('The capital of India is "Show Data".')
#         capital_label.setWordWrap(True)
#         left_layout.addWidget(capital_label)
        
#         # Record button
#         record_btn = QPushButton("CRA to start recording")
#         record_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; }")
#         left_layout.addWidget(record_btn)
        
#         # Session info
#         session_label = QLabel("Session: Workshop ... Magazine (My Name)")
#         session_label.setWordWrap(True)
#         left_layout.addWidget(session_label)
        
#         # Add stretch to push everything to the top
#         left_layout.addStretch()
        
#         # Right panel with tabs
#         right_panel = QTabWidget()
        
#         # Editor tab
#         editor_tab = QWidget()
#         editor_layout = QVBoxLayout(editor_tab)
        
#         editor_title = QLabel("main.py > @ agent.chat")
#         editor_title.setFont(QFont("Courier New", 10))
#         editor_layout.addWidget(editor_title)
        
#         # Line numbers and code
#         code_widget = QWidget()
#         code_layout = QHBoxLayout(code_widget)
        
#         # Line numbers
#         line_numbers = QTextEdit()
#         line_numbers.setMaximumWidth(40)
#         line_numbers.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         line_numbers.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         line_numbers.setText("\n".join(str(i) for i in range(90, 103)))
#         line_numbers.setReadOnly(True)
#         code_layout.addWidget(line_numbers)
        
#         # Code content
#         code_editor = QTextEdit()
#         code_editor.setText('''model = gemail.GenerativeModel('gemini-1.5-flash')

# logger.info("Starting LLM stream generation...")
# responses = model.generate_content (history, stream=True)

# accumulated_response = ""
# for response in responses:
#     if hasattr(response, 'text') and response.text:
#     logger.info(f"LLM chunk: (response.text)")
#     accumulated_response += response.text
#     logger.info("LLM stream generation complete.")
#     return accumulated_response''')
#         code_editor.setFont(QFont("Courier New", 10))
#         code_layout.addWidget(code_editor)
        
#         editor_layout.addWidget(code_widget)
        
#         # Add editor tab
#         right_panel.addTab(editor_tab, "Editor")
        
#         # Problems tab
#         problems_tab = QWidget()
#         problems_layout = QVBoxLayout(problems_tab)
#         problems_list = QListWidget()
#         problems_list.addItems(["No problems detected", "Code analysis complete"])
#         problems_layout.addWidget(problems_list)
#         right_panel.addTab(problems_tab, "PROBLEMS")
        
#         # Console tab
#         console_tab = QWidget()
#         console_layout = QVBoxLayout(console_tab)
#         console_output = QTextEdit()
#         console_output.setFont(QFont("Courier New", 9))
#         console_output.setText('''INFO.main:LLM stream generation complete.
# INFO.main:LLM response generated: "The capital of India is *"New Delhi"".

# INFO.main:Generating speech audio...
# INFO:https://api.amrf.ai/v1/speech/generate?origin=python_sdk:2.0.2 "HTTP/1.1 200 OK"
# INFO.main:Speech audio generated successfully
# INFO:    127.00-1.51915 - "POST /agent/chat/300be62a-308b-4316-8ca1-fd12760498ce HTTP/1.1" 200 OK"
# INFO:    127.00-1.51915 - "GET /favicon.ico HTTP/1.1" 404 Not Found''')
#         console_output.setReadOnly(True)
#         console_layout.addWidget(console_output)
#         right_panel.addTab(console_tab, "CONSOLE")
        
#         # Terminal tab
#         terminal_tab = QWidget()
#         terminal_layout = QVBoxLayout(terminal_tab)
#         terminal_output = QTextEdit()
#         terminal_output.setFont(QFont("Courier New", 9))
#         terminal_output.setText("$ python main.py\nAI Voice Agent started on port 8000...")
#         terminal_output.setReadOnly(True)
#         terminal_layout.addWidget(terminal_output)
#         right_panel.addTab(terminal_tab, "TERMINAL")
        
#         # Output tab
#         output_tab = QWidget()
#         output_layout = QVBoxLayout(output_tab)
#         output_text = QTextEdit()
#         output_text.setFont(QFont("Courier New", 9))
#         output_text.setText("Speech generated successfully. Playing audio...")
#         output_text.setReadOnly(True)
#         output_layout.addWidget(output_text)
#         right_panel.addTab(output_tab, "OUTPUT")
        
#         # Add tabs to main layout
#         main_layout.addWidget(left_panel)
#         main_layout.addWidget(right_panel)
        
#         # Set style
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #2b2b2b;
#                 color: #ffffff;
#             }
#             QTabWidget::pane {
#                 border: 1px solid #444;
#                 background: #2b2b2b;
#             }
#             QTabBar::tab {
#                 background: #333;
#                 color: #ccc;
#                 padding: 8px;
#                 border-top-left-radius: 4px;
#                 border-top-right-radius: 4px;
#             }
#             QTabBar::tab:selected {
#                 background: #2b2b2b;
#                 color: #fff;
#                 border-bottom: 2px solid #4CAF50;
#             }
#             QTextEdit, QListWidget {
#                 background-color: #1e1e1e;
#                 color: #d4d4d4;
#                 border: 1px solid #444;
#             }
#             QFrame {
#                 background-color: #2b2b2b;
#                 color: #ffffff;
#             }
#             QLabel {
#                 color: #ffffff;
#             }
#         """)
        
#         # Simulate activity
#         self.simulate_activity()

#     def simulate_activity(self):
#         """Simulate console activity"""
#         self.console_updates = [
#             "INFO: New voice input detected",
#             "INFO: Processing audio...",
#             "INFO: Speech recognition complete",
#             "INFO: Query: 'What is the capital of India?'",
#             "INFO: Generating response...",
#             "INFO: Response: 'The capital of India is New Delhi'",
#             "INFO: Converting text to speech...",
#             "INFO: Audio playback started"
#         ]
#         self.update_index = 0
        
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.add_console_message)
#         self.timer.start(2000)

#     def add_console_message(self):
#         if self.update_index < len(self.console_updates):
#             console_widget = self.findChild(QTabWidget).widget(2).findChild(QTextEdit)
#             current_text = console_widget.toPlainText()
#             console_widget.setText(current_text + "\n" + self.console_updates[self.update_index])
#             self.update_index += 1
#         else:
#             self.timer.stop()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = VoiceAgentApp()
#     window.show()
#     sys.exit(app.exec_())