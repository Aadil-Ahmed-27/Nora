import asyncio
import traceback
import os
import threading
import time
from collections import deque
import base64
import io

import pyaudio
import cv2
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
DEFAULT_MODEL = "gemini-2.5-flash-live-preview"

# Session state
class SessionState:
    def __init__(self):
        self.is_active = False
        self.audio_manager = None
        self.session = None
        self.tasks = []
        self.stop_event = asyncio.Event()
        self.user_speaking = False
        self.ai_speaking = False
        self.last_user_audio_time = 0
        self.silence_threshold = 0.5  # seconds of silence before AI can speak
        
session_state = SessionState()

# Gradio interface state
interface_state = {
    "webcam_stream": None,
    "session_active": False
}

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
        self.pya = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_queue = deque()
        self.is_playing = False
        self.playback_task = None
        self.audio_level_callback = None

    async def initialize(self):
        mic_info = self.pya.get_default_input_device_info()
        print(f"Microphone used: {mic_info}")

        self.input_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=self.input_sample_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=self.output_sample_rate,
            output=True,
        )

    def add_audio(self, audio_data):
        """Add audio data to the playback queue"""
        self.audio_queue.append(audio_data)
        
        if self.playback_task is None or self.playback_task.done():
            self.playback_task = asyncio.create_task(self.play_audio())

    async def play_audio(self):
        """Play all queued audio data"""
        print("🗣️ Nora speaking...")
        session_state.ai_speaking = True
        
        while self.audio_queue:
            try:
                # Check if user is speaking (interruption)
                if session_state.user_speaking:
                    print("🛑 User interruption detected, pausing AI speech")
                    self.audio_queue.clear()  # Clear remaining audio
                    break
                    
                audio_data = self.audio_queue.popleft()
                await asyncio.to_thread(self.output_stream.write, audio_data)
            except Exception as e:
                print(f"Error playing audio: {e}")

        session_state.ai_speaking = False
        self.is_playing = False
        print("✅ Nora finished speaking")

    def interrupt(self):
        """Handle interruption by stopping playback and clearing queue"""
        self.audio_queue.clear()
        self.is_playing = False
        session_state.ai_speaking = False

        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()

    def close_streams(self):
        """Close audio streams and terminate PyAudio."""
        if self.input_stream is not None:
            self.input_stream.close()
        if self.output_stream is not None:
            self.output_stream.close()
        if self.pya is not None:
            self.pya.terminate()


def detect_audio_level(audio_data):
    """Detect if user is speaking based on audio level"""
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_np**2))
    return rms > 500  # Threshold for speech detection


async def get_frame_data(cap):
    """Capture and process webcam frame"""
    ret, frame = await asyncio.to_thread(cap.read)
    if not ret:
        print("⚠️ Failed to capture frame from webcam.")
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)
    img.thumbnail([512, 512])
    image_io = io.BytesIO()
    img.save(image_io, format="jpeg", quality=70)
    image_io.seek(0)
    image_bytes = image_io.read()
    return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}


async def initialize_client():
    """Initialize the Gemini client"""
    global CLIENT_INSTANCE
    
    if CLIENT_INSTANCE is None:
        try:
            CLIENT_INSTANCE = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print("✅ Gemini client initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing Gemini client: {e}")
            traceback.print_exc()
            return False
    return True


async def conversation_loop():
    """Main conversation loop"""
    global CLIENT_INSTANCE
    
    if not await initialize_client():
        return
    
    audio_manager = AudioManager(
        input_sample_rate=SEND_SAMPLE_RATE, 
        output_sample_rate=RECEIVE_SAMPLE_RATE
    )
    
    await audio_manager.initialize()
    session_state.audio_manager = audio_manager
    
    audio_send_queue = asyncio.Queue(maxsize=100)
    video_send_queue = asyncio.Queue(maxsize=10)
    
    cap = None
    
    try:
        # Initialize webcam
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        async with (
            CLIENT_INSTANCE.aio.live.connect(
                model=DEFAULT_MODEL, config=LIVE_CONNECT_CONFIG
            ) as session,
            asyncio.TaskGroup() as tg,
        ):
            session_state.session = session
            print("🚀 Nora session started")
            
            async def listen_for_audio():
                """Audio input handler with interruption detection"""
                print("🎤 Audio listener started...")
                while not session_state.stop_event.is_set():
                    try:
                        data = await asyncio.to_thread(
                            audio_manager.input_stream.read,
                            AUDIO_CHUNK_SIZE,
                            exception_on_overflow=False,
                        )
                        
                        # Detect if user is speaking
                        is_speaking = detect_audio_level(data)
                        
                        if is_speaking:
                            session_state.last_user_audio_time = time.time()
                            if not session_state.user_speaking:
                                session_state.user_speaking = True
                                # If AI is currently speaking, interrupt it
                                if session_state.ai_speaking:
                                    print("🛑 User interruption - stopping AI")
                                    audio_manager.interrupt()
                        else:
                            # Check if user stopped speaking
                            if session_state.user_speaking:
                                silence_duration = time.time() - session_state.last_user_audio_time
                                if silence_duration > session_state.silence_threshold:
                                    session_state.user_speaking = False
                                    print("✅ User finished speaking")
                        
                        await audio_send_queue.put(data)
                        
                    except IOError as e:
                        if hasattr(e, "errno") and e.errno == pyaudio.paInputOverflowed:
                            continue
                        print(f"🎤 Audio input error: {e}")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"🎤 Audio listener error: {e}")
                        await asyncio.sleep(0.1)
                        
                print("🎤 Audio listener stopped")

            async def process_and_send_audio():
                """Audio sender"""
                print("🔊 Audio sender started...")
                while not session_state.stop_event.is_set() or not audio_send_queue.empty():
                    try:
                        data = await asyncio.wait_for(audio_send_queue.get(), timeout=0.5)
                        await session.send_realtime_input(
                            media={
                                "data": data,
                                "mime_type": f"audio/pcm;rate={AUDIO_SEND_SAMPLE_RATE}",
                            }
                        )
                        audio_send_queue.task_done()
                    except asyncio.TimeoutError:
                        if session_state.stop_event.is_set() and audio_send_queue.empty():
                            break
                    except Exception as e:
                        print(f"🔊 Audio sender error: {e}")
                        await asyncio.sleep(0.1)
                        
                print("🔊 Audio sender stopped")

            async def stream_video_frames():
                """Video frame streaming"""
                print("📹 Video streamer started...")
                while not session_state.stop_event.is_set():
                    frame_media = await get_frame_data(cap)
                    if frame_media:
                        await video_send_queue.put(frame_media)
                    await asyncio.sleep(VIDEO_FRAME_RATE_DELAY)
                print("📹 Video streamer stopped")

            async def process_and_send_video():
                """Video sender"""
                print("🖼️ Video sender started...")
                while not session_state.stop_event.is_set() or not video_send_queue.empty():
                    try:
                        video_data = await asyncio.wait_for(video_send_queue.get(), timeout=0.5)
                        await session.send_realtime_input(media=video_data)
                        video_send_queue.task_done()
                    except asyncio.TimeoutError:
                        if session_state.stop_event.is_set() and video_send_queue.empty():
                            break
                    except Exception as e:
                        print(f"🖼️ Video sender error: {e}")
                        await asyncio.sleep(0.1)
                print("🖼️ Video sender stopped")

            async def receive_and_play():
                """Response receiver and player"""
                print("👂 Response receiver started...")
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content and server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.inline_data:
                                # Only play audio if user is not currently speaking
                                if not session_state.user_speaking:
                                    audio_manager.add_audio(part.inline_data.data)
                            if part.text:
                                print(f"💬 Nora: {part.text}")

                    if server_content and server_content.turn_complete:
                        print("✅ Nora finished response")
                        
                print("👂 Response receiver stopped")

            # Start all tasks
            tg.create_task(listen_for_audio(), name="AudioListener")
            tg.create_task(process_and_send_audio(), name="AudioSender")
            tg.create_task(stream_video_frames(), name="VideoStreamer")
            tg.create_task(process_and_send_video(), name="VideoSender")
            tg.create_task(receive_and_play(), name="ResponseReceiver")
            
            print("🎯 All systems active - Nora is ready!")
            await session_state.stop_event.wait()
            
    except KeyboardInterrupt:
        print("\n👋 Session interrupted")
    except Exception as e:
        print(f"💥 Session error: {e}")
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        session_state.stop_event.set()
        if audio_manager:
            audio_manager.close_streams()
        if cap and cap.isOpened():
            await asyncio.to_thread(cap.release)
        print("✅ Cleanup complete")


def run_conversation_in_thread():
    """Run the conversation loop in a separate thread"""
    def run_async():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(conversation_loop())
        except Exception as e:
            print(f"Error in conversation thread: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    return thread


def toggle_session():
    """Toggle the conversation session on/off"""
    if not session_state.is_active:
        # Start session
        print("🟢 Starting Nora session...")
        session_state.is_active = True
        session_state.stop_event.clear()
        
        # Start conversation in background thread
        run_conversation_in_thread()
        
        return "🟢 Stop Session", gr.update(variant="stop")
    else:
        # Stop session
        print("🔴 Stopping Nora session...")
        session_state.is_active = False
        session_state.stop_event.set()
        
        # Clean up audio manager
        if session_state.audio_manager:
            session_state.audio_manager.interrupt()
            
        return "▶️ Start Session", gr.update(variant="primary")


def get_webcam_feed():
    """Continuous webcam feed for Gradio"""
    cap = cv2.VideoCapture(0)
    
    def generate():
        while True:
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for Gradio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
            else:
                break
                
    return generate


# Custom CSS for modern, professional look
custom_css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

#title {
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 2rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

#camera-container {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin-bottom: 2rem;
}

#camera-feed {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

#control-container {
    text-align: center;
}

.session-btn {
    font-size: 1.2rem !important;
    padding: 1rem 3rem !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}

.session-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
}

#status-indicator {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    background: rgba(255,255,255,0.2);
    color: white;
    font-weight: 500;
    backdrop-filter: blur(10px);
}
"""


def create_nora_interface():
    """Create the Gradio interface"""
    with gr.Blocks(css=custom_css, title="Nora - Shopping Assistant") as demo:
        gr.HTML("""
            <div id="main-container">
                <h1 id="title">🛒 Nora: Your Shopping Assistant</h1>
            </div>
        """)
        
        with gr.Column(elem_id="camera-container"):
            webcam = gr.Image(
                source="webcam",
                streaming=True,
                elem_id="camera-feed",
                height=400,
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                interactive=False
            )
        
        with gr.Column(elem_id="control-container"):
            session_btn = gr.Button(
                "▶️ Start Session",
                variant="primary",
                size="lg",
                elem_classes=["session-btn"]
            )
            
            status = gr.HTML("""
                <div id="status-indicator">
                    💤 Ready to start your shopping conversation
                </div>
            """)
        
        # Button click handler
        session_btn.click(
            fn=toggle_session,
            outputs=[session_btn, session_btn]
        )
        
        # Update status based on session state
        def update_status():
            if session_state.is_active:
                if session_state.user_speaking:
                    return "🎤 Listening to you..."
                elif session_state.ai_speaking:
                    return "🗣️ Nora is speaking..."
                else:
                    return "✅ Session active - Ready to chat!"
            else:
                return "💤 Ready to start your shopping conversation"
        
        # Periodic status updates
        demo.load(
            fn=lambda: update_status(),
            outputs=status,
            every=1
        )
    
    return demo


def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Launching Nora Shopping Assistant...")
    
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in the .env file")
        return
    
    demo = create_nora_interface()
    
    print("🌐 Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()