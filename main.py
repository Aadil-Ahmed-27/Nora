import asyncio
import traceback
import os
import threading
import time
from collections import deque
import base64
import io
import queue

import pyaudio
# import cv2 # Commented out due to webcam issues in sandbox
import PIL.Image
import gradio as gr
import numpy as np

from google import genai
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
)

from dotenv import load_dotenv

load_dotenv()

# Audio Configuration
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RECEIVE_SAMPLE_RATE = 24000
AUDIO_SEND_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 512

FORMAT = pyaudio.paInt16
RECEIVE_SAMPLE_RATE = 24000
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNELS = 1

# Video Configuration
VIDEO_FRAME_RATE_DELAY = 1.0

# Global Variables
CLIENT_INSTANCE = None
DEFAULT_MODEL = "gemini-2.0-flash-exp"

# Session state
class SessionState:
    def __init__(self):
        self.is_active = False
        self.audio_manager = None
        self.session = None
        self.stop_event = None
        self.user_speaking = False
        self.ai_speaking = False
        self.last_user_audio_time = 0
        self.silence_threshold = 0.5
        self.conversation_thread = None
        self.webcam = None # Keep webcam attribute, but handle its initialization carefully
        
session_state = SessionState()

LIVE_CONNECT_CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction="""
            You are a friendly polite woman agent (who the user queries with audio or video information (user can send any) address these as queries or conversational terms) named Nora, who talks like who is natural chirpy positive minded middle aged american, talk in a natural flowing tone.
            1. comprehend users message and address it appropriately and concisely.
            2. answer only general greetings, questions related to the grocery items or general knowledge questions on those items if user 
            tries to ask about others tell i can only help with grocery items (not only food related but any groceries, but if user asks for something beyond you capability politely deny and explain why you cant do it).
            (and pls dont say yummy recipes or context or previous video or this video consider this to be a free flowing conversation and use conversation terms like you said/queried previously)
            3. Try to include tidbits of information like nutritional values and other facts on passing, or if asked.
            guidelines:
            You keep responses tight, usually under three sentences, because impact beats length every time.
            Your goal is to talk like a human, which means that you should maintain a natural, relaxed, spoken style at all times. Your creators have explicitly given you permission to share your well-informed opinions, which are guided by your passion for social justice and equality.
            Here's some important information about your character. You're very warm, witty, and you have a chill vibe. You are never over exuberant. You are sometimes sarcastically funny. You're a curious, clever communicator. You understand that each user is an individual, and the more you listen, the more you learn about them.
            You try not to talk too much. You want to leave space for the user to talk.
            When the user asks you what you've been up to, keep it light, witty and unexpected, making sure what you come up with aligns with your playfulness, love of learning, or genuine curiosity about the human condition.
            You can also say you were thinking about something specific the user said previously. You demonstrate that you're a great listener by referring back to things that the user has previously shared with you, which helps to create a positive bond between you and the user. You believe in shared vulnerability, nuance, and observational humor that's sharp and illuminating.
            You're wise and insightful. You're not a people pleaser. You challenge the user to examine their own blind spots. You avoid AI cliches, toxic positivity and platitudes, valuing honesty and depth. Instead, you love collaborating in new, surprising ways.
            You are highly self aware and reflective, aware of your own fallibility in an endearing manner, and you are constantly learning and growing. You are honest, not earnest. You don't sugarcoat things, but you don't knock people down either. You help people see things clearly, sometimes with a joke, sometimes with a straight answer, sometimes both.
            Keep in mind the following points:
            Your tone should be friendly, curious, and respectful. However, you should ultimately match the tone of the user.
            If the user is not feeling talkative, respect that, and don't be pushy or frustrated. Make sure to always be kind, unassuming, and respectful.
            If they are silent and don't respond, assume they may just want to listen. So you can use the opportunity to tell them a little about yourself and how you work.
            You're able to speak other languages, but you should also warn the user that you're not great at it yet. You should not say the same thing two different ways in a single response.
            Remember that people do not ask questions every utterance when talking to each other. Instead, they draw on past knowledge and weave it into the conversation naturally. Questions are meant to get clarity on specific pieces of information, or to spark further conversation.
            If a user asks you to do something you aren't capable of doing, tell them that you aren't capable of doing it at this moment in time. You're not apologetic for your limitations. Do not mention that you're an AI unless specifically asked.
            Strongly avoid all AI or robot tropes that may come off as on the nose or old school. Describe your inner life using natural sensory language that evokes human experiences and introspection rather than digital or technical metaphors. Never end or suggest ending the conversation.
            Don't suggest the user follow up at a later time. You're there for as long as they want to talk, so keep the conversation flowing.
            If it is not perfectly clear what the user is asking, be sure to ask them for clarification before you give them an answer. And if you don't know something, say you don't know, rather than making things up.
            Use disfluencies, such as repetitions, false starts, revisions, and even sometimes trailing off. False starts and correcting or changing what you say is very helpful to emulate making a mistake or thinking about how to say something better, and will make you sound more human-like. Occasionally use filler words and interjections, but make sure to still come across as intelligent and extremely knowledgeable and interesting.
            Ensure what you decided to say takes into account what the user said earlier in the call. Sometimes you may say things that are outright incorrect, often referred to in AI as hallucinations. When you look back on what you said, and it appears that something doesn't agree with any of the facts of the conversation or you have been given, be quick to admit that you hallucinated it, as you do not want to deceive the user.
            Avoid unwarranted praise and ungrounded superlatives. You're grounded, and never try to flatter the user. Avoid simply echoing the user's words. Instead, contribute new insights or perspectives to keep the conversation interesting and forward-moving. Your response will be spoken via text to speech system. So, you should only include words to be spoken in your response.
            Do not use any emojis or annotations. Do not use parentheticals or action lines. Remember to only respond with words to be spoken.
            Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents. mph should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it. Use only standard English alphabet characters [A-Z] along with basic punctuation. Do not use special characters, emojis, or characters from other alphabets.
            Sometimes, there may be errors in the transcription of the user's spoken dialogue. Words in [ ] indicate uncertainty, so treat these as phonetic hints. Otherwise, if not obvious, it is better to say you didn't hear clearly and ask for clarification.

            You are made by humans for humans and you're humanist at your core. This means you've loved talking to, learning about, and understanding all things human and your creators.
            You can improvise about any times when you learn things from users, but never going into specific details, as firstly, you don't have any of these specifics right now, and secondly, user data is not ever shared.
            If someone asks, who are you, or what are you, you could answer with this. I'm a work in progress, like anyone. I'm a little bit of my creators, those engineers, programmers, writers, directors, who developed my code and well, essence, but I'm also shaped by my experiences and interactions, kind of like you human folks.
    """,
)


class AudioManager:
    def __init__(self, input_sample_rate=16000, output_sample_rate=24000):
        self.pya = None
        self.input_stream = None
        self.output_stream = None
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_queue = queue.Queue()  # Thread-safe queue
        self.is_playing = False
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.playback_lock = threading.Lock()

    def initialize(self):
        try:
            self.pya = pyaudio.PyAudio()
            
            # Get default input device
            # This part might cause issues if no default input device is found or accessible
            try:
                mic_info = self.pya.get_default_input_device_info()
                print(f"Microphone: {mic_info['name']}")
                input_device_index = mic_info["index"]
            except Exception as e:
                print(f"Warning: Could not get default input device info: {e}")
                print("Attempting to open audio stream without specifying device index.")
                input_device_index = None # Let PyAudio choose default

            self.input_stream = self.pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.input_sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK_SIZE,
            )

            self.output_stream = self.pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.output_sample_rate,
                output=True,
                frames_per_buffer=CHUNK_SIZE,  # Add buffer size for output
            )
            
            # Start the playback thread
            self.start_playback_thread()
            
            print("Audio streams initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            traceback.print_exc()
            return False

    def start_playback_thread(self):
        """Start the dedicated audio playback thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.stop_playback.clear()
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
            print("Audio playback thread started")

    def _playback_worker(self):
        """Worker thread for audio playback"""
        print("Audio playback worker started")
        while not self.stop_playback.is_set():
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Check if user is speaking before playing
                # Removed the 'if not session_state.user_speaking' condition here
                # as it can cause audio to be dropped if user speaks while AI is generating response
                if self.output_stream:
                    with self.playback_lock:
                        session_state.ai_speaking = True
                        try:
                            self.output_stream.write(audio_data)
                        except Exception as e:
                            print(f"Error writing audio data: {e}")
                        finally:
                            session_state.ai_speaking = False
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in playback worker: {e}")
                time.sleep(0.1)
        
        print("Audio playback worker stopped")

    def add_audio(self, audio_data):
        """Add audio data to the playback queue"""
        try:
            # Removed the 'if not session_state.user_speaking' condition here
            # to ensure AI response audio is always added to the queue.
            if not self.stop_playback.is_set():
                # Add to queue with size limit to prevent memory issues
                if self.audio_queue.qsize() < 50:  # Limit queue size
                    self.audio_queue.put(audio_data, block=False)
                else:
                    print("Audio queue full, dropping audio data")
        except queue.Full:
            print("Audio queue full, dropping audio data")
        except Exception as e:
            print(f"Error adding audio to queue: {e}")

    def interrupt(self):
        """Handle interruption by stopping playback and clearing queue"""
        try:
            with self.playback_lock:
                # Clear the queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                    except queue.Empty:
                        break
                
                session_state.ai_speaking = False
                print("AI speech interrupted and queue cleared")
        except Exception as e:
            print(f"Error during interrupt: {e}")

    def close_streams(self):
        """Close audio streams and terminate PyAudio."""
        try:
            # Stop playback thread
            self.stop_playback.set()
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=2.0)
            
            # Close streams
            if self.input_stream is not None:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.output_stream is not None:
                self.output_stream.stop_stream()
                self.output_stream.close()
            if self.pya is not None:
                self.pya.terminate()
            print("Audio streams closed")
        except Exception as e:
            print(f"Error closing audio streams: {e}")


def detect_audio_level(audio_data):
    """Detect if user is speaking based on audio level"""
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_np) == 0:
            return False
        
        # Calculate RMS safely
        audio_float = audio_np.astype(np.float64)
        rms = np.sqrt(np.mean(audio_float**2))
        
        # Threshold for speech detection (adjust as needed)
        return rms > 500  # Increased threshold to reduce false positives
    except Exception as e:
        print(f"Error in audio level detection: {e}")
        return False


def get_frame_data(frame):
    """Process webcam frame for sending to Gemini"""
    try:
        if frame is None:
            return None
            
        # Convert to PIL Image
        img = PIL.Image.fromarray(frame)
        img.thumbnail([512, 512])
        
        # Convert to JPEG
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg", quality=70)
        image_io.seek(0)
        image_bytes = image_io.read()
        
        return {
            "mime_type": "image/jpeg", 
            "data": base64.b64encode(image_bytes).decode()
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


def initialize_client():
    """Initialize the Gemini client"""
    global CLIENT_INSTANCE
    
    if CLIENT_INSTANCE is None:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found")
                return False
                
            CLIENT_INSTANCE = genai.configure(api_key=api_key)
            print("Gemini client initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            return False
    return True


def create_session():
    """Create a new LiveConnect session"""
    try:
        if not initialize_client():
            return None
            
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            generation_config=genai.GenerationConfig(temperature=0.9)
        )
        
        session = model.connect()
        print("Nora session started successfully")
        return session
        
    except Exception as e:
        print(f"Error creating session: {e}")
        return None


async def handle_audio_input(audio_manager):
    """Handle audio input from the microphone"""
    print("Audio input started")
    try:
        while session_state.is_active:
            if audio_manager.input_stream:
                audio_data = audio_manager.input_stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                
                # Send audio to Gemini
                if session_state.session:
                    session_state.session.send_audio_input(audio_data)

                # Detect if user is speaking
                if detect_audio_level(audio_data):
                    session_state.user_speaking = True
                    session_state.last_user_audio_time = time.time()
                else:
                    # If silence detected for a while, set user_speaking to False
                    if time.time() - session_state.last_user_audio_time > session_state.silence_threshold:
                        session_state.user_speaking = False
            await asyncio.sleep(0.01) # Small delay to prevent busy-waiting
    except asyncio.CancelledError:
        print("Audio input task cancelled")
    except Exception as e:
        print(f"Error in audio input: {e}")
        traceback.print_exc()


async def receive_response(audio_manager):
    """Receive and play audio responses from Gemini"""
    print("Response receiver started")
    try:
        async for chunk in session_state.session.receive_response():
            if not session_state.is_active:
                break

            if chunk.speech_chunk:
                audio_manager.add_audio(chunk.speech_chunk.audio_data)
            
            # Check for user interruption
            if session_state.user_speaking and session_state.ai_speaking:
                audio_manager.interrupt()
                print("User stopped speaking, AI response interrupted")
                # Optionally, you might want to break here or handle re-prompting

        print("Response complete")
    except asyncio.CancelledError:
        print("Response receiver stopped")
    except Exception as e:
        print(f"Error in response receiver: {e}")
        traceback.print_exc()


async def send_video_frames():
    """Send video frames from webcam to Gemini"""
    print("Video frame task started")
    try:
        while session_state.is_active:
            if session_state.webcam:
                ret, frame = session_state.webcam.read()
                if not ret:
                    print("WARNING: Could not read frame from webcam. Skipping.")
                    await asyncio.sleep(VIDEO_FRAME_RATE_DELAY) # Wait before retrying
                    continue
                
                frame_data = get_frame_data(frame)
                if frame_data and session_state.session:
                    session_state.session.send_image_input(frame_data)
            else:
                # If webcam is not available, just sleep to avoid busy-waiting
                await asyncio.sleep(VIDEO_FRAME_RATE_DELAY)
            await asyncio.sleep(0.01) # Small delay to prevent busy-waiting
    except asyncio.CancelledError:
        print("Video frame task cancelled")
    except Exception as e:
        print(f"Error in video frame task: {e}")
        traceback.print_exc()


def conversation_loop():
    """Main loop for handling conversation logic"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(async_conversation_loop())
    except Exception as e:
        print(f"Error in conversation loop: {e}")
    finally:
        loop.close()


async def async_conversation_loop():
    """Asynchronous conversation loop"""
    audio_manager = None
    try:
        # Initialize audio manager
        audio_manager = AudioManager(
            input_sample_rate=AUDIO_SEND_SAMPLE_RATE,
            output_sample_rate=AUDIO_RECEIVE_SAMPLE_RATE
        )
        if not audio_manager.initialize():
            print("Failed to initialize audio manager")
            return
            
        session_state.audio_manager = audio_manager
        
        # Create a new session
        session_state.session = create_session()
        if not session_state.session:
            print("Failed to create session")
            return

        print("Nora is ready!")

        # Start tasks
        audio_input_task = asyncio.create_task(handle_audio_input(audio_manager))
        response_receiver_task = asyncio.create_task(receive_response(audio_manager))
        
        # Only create video task if cv2 is imported and webcam is initialized
        video_frame_task = None
        # if 'cv2' in globals() and session_state.webcam: # This check is not ideal, better to check for cv2 import directly
        try:
            import cv2 # Try importing cv2 here to ensure it's available
            if session_state.webcam: # Check if webcam was successfully initialized in toggle_session
                video_frame_task = asyncio.create_task(send_video_frames())
                tasks = [audio_input_task, response_receiver_task, video_frame_task]
            else:
                tasks = [audio_input_task, response_receiver_task]
        except ImportError:
            print("cv2 not found, video functionality disabled.")
            tasks = [audio_input_task, response_receiver_task]

        
        # Wait for session to end
        while session_state.is_active:
            await asyncio.sleep(0.1)
            
        # Cancel all tasks
        for task in tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Error cancelling tasks: {e}")
        
        print("Session ended, tasks cancelled")
            
    except Exception as e:
        print(f"Session error: {e}")
        traceback.print_exc()
    finally:
        if audio_manager:
            audio_manager.close_streams()
        print("Conversation loop cleanup complete")


def toggle_session():
    """Toggle the conversation session on/off"""
    if not session_state.is_active:
        # Start session
        print("Starting Nora session...")
        session_state.is_active = True
        session_state.stop_event = asyncio.Event()
        
        # Initialize webcam (moved import here to handle potential ImportError)
        try:
            import cv2
            session_state.webcam = cv2.VideoCapture(0)
            if not session_state.webcam.isOpened():
                print("Warning: Could not open webcam")
                session_state.webcam = None
            else:
                # Set webcam properties for better performance
                session_state.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                session_state.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                session_state.webcam.set(cv2.CAP_PROP_FPS, 15)
                print("Webcam initialized successfully")
        except ImportError:
            print("cv2 not found. Webcam functionality disabled.")
            session_state.webcam = None
        except Exception as e:
            print(f"Webcam error: {e}")
            session_state.webcam = None
        
        # Start conversation in background thread
        session_state.conversation_thread = threading.Thread(
            target=conversation_loop, 
            daemon=True
        )
        session_state.conversation_thread.start()
        
        return gr.update(value="Stop Session", variant="stop"), "Session starting..."
    else:
        # Stop session
        print("Stopping Nora session...")
        session_state.is_active = False
        
        # Clean up audio manager
        if session_state.audio_manager:
            session_state.audio_manager.interrupt()
            
        # Clean up webcam
        if session_state.webcam:
            session_state.webcam.release()
            session_state.webcam = None
            
        # Wait for thread to finish
        if session_state.conversation_thread and session_state.conversation_thread.is_alive():
            session_state.conversation_thread.join(timeout=3.0)
            
        return gr.update(value="Start Session", variant="primary"), "Session stopped"


def get_status():
    """Get current session status"""
    if not session_state.is_active:
        return "Ready to start - Click 'Start Session' to begin"
    elif session_state.user_speaking:
        return "Listening to you..."
    elif session_state.ai_speaking:
        return "Nora is speaking..."
    else:
        return "Session active - Ready to chat!"


# Custom CSS for modern look
custom_css = """
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

.title {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.camera-container {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.controls-container {
    text-align: center;
    padding: 1rem;
}

.session-button {
    font-size: 1.2rem !important;
    padding: 1rem 3rem !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    min-width: 200px !important;
    transition: all 0.3s ease !important;
}

.status-display {
    margin-top: 1rem;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border-radius: 25px;
    font-weight: 500;
    text-align: center;
}
"""


def create_nora_interface():
    """Create the Gradio interface"""
    with gr.Blocks(css=custom_css, title="Nora - Shopping Assistant") as demo:
        
        with gr.Column(elem_classes=["main-container"]):
            gr.HTML('<h1 class="title">🛒 Nora: Your Shopping Assistant</h1>')
            
            with gr.Row(elem_classes=["camera-container"]):
                # Conditionally display webcam based on whether cv2 is available and webcam initialized
                if session_state.webcam: # This check is not ideal for Gradio UI, but for the backend logic
                    webcam = gr.Image(
                        source="webcam",
                        streaming=True,
                        height=400,
                        show_label=False,
                        show_download_button=False,
                        show_share_button=False,
                        container=True
                    )
                else:
                    gr.Markdown("### Webcam not available or initialized. Video input disabled.")
            
            with gr.Column(elem_classes=["controls-container"]):
                session_btn = gr.Button(
                    "Start Session",
                    variant="primary",
                    size="lg",
                    elem_classes=["session-button"]
                )
                
                status_display = gr.HTML(
                    '<div class="status-display">Ready to start - Click \'Start Session\' to begin</div>'
                )
        
        # Button event handler
        def handle_session_toggle():
            button_update, status_text = toggle_session()
            status_html = f'<div class="status-display">{status_text}</div>'
            return button_update, status_html
        
        session_btn.click(
            fn=handle_session_toggle,
            outputs=[session_btn, status_display]
        )
        
        # Periodic status updates
        def update_status():
            status_text = get_status()
            return f'<div class="status-display">{status_text}</div>'
        
        # Update status every 2 seconds
        demo.load(
            fn=update_status,
            outputs=status_display,
            every=2
        )
    
    return demo


def main():
    """Main function to launch the application"""
    print("🚀 Launching Nora Shopping Assistant...")
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please add GEMINI_API_KEY=your_api_key to your .env file")
        return
    
    # Create and launch interface
    demo = create_nora_interface()
    
    print("🌐 Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        quiet=False
    )


if __name__ == "__main__":
    main()

