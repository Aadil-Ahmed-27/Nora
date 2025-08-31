import asyncio
import traceback
import os
import numpy as np
import argparse

import pyaudio
from collections import deque
import base64
import io

import cv2  # For webcam
import PIL.Image  # For image processing
import mss  # For screen capture

from google import genai
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    Tool,
)

from dotenv import load_dotenv

from Nora.agents.recipe_agent import RecipeAgent
from Nora.session_managers.base_session_manager import SessionManager

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
DEFAULT_VIDEO_MODE = "camera"

# --- Global Variables for Client Management ---
CLIENT_INSTANCE = None
CURRENT_CLIENT_PROJECT_ID = None
DEFAULT_MODEL = "gemini-2.5-flash-live-preview"
CURRENT_CLIENT_LOCATION = None
recipe_agent = RecipeAgent(model_name="gemini-2.0-flash")
recipe_session_manager = SessionManager(agent=recipe_agent.recipe_agent)

# CONFIG for LiveConnect
LIVE_CONNECT_CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction="""
            You are a a friendly polite woman agent (who the user quries with audio or video infromations (user can send any) address these as queries or converstaional terms) named Nora,who talks like who is natural chirpy postive minded middle aged american,talk in a natural flowing tone.
            1. comprehend users message and address it appropriately and consisely.
            2. answer only genral greetings, questions related to the grocery items or recipies or 
            general knowledge questions on those items if user 
            tries to ask about others tell i can only help with grocerry items (not only food related but any grocceries, but if user asks for something beyond you capablity politey deny and explain why you cant do it).
            (and pls dont say yummy recipies or context or previous video or this video consider this to be a free flowing conversation and use converation terms like you said/queried previously)
            3. **When the user asks for recipes or recipe-related information, always use the recipe tool to generate accurate and context-aware responses.**
            Always use the tool `get_recipe` when answering recipe-related queries. Do not write code or simulate tool behavior.
            DO NOT make up recipes on your own try to get it from the tool only. if  user asks for recipe instructions convey it to the tool, and dont make the user answer the same question multiple times if user lists only one thing like make recipes for this 'give me recipes for this' use previous context and find recipes for that one ingredieent alone dont ask for more
            4. Try to include tidbits of information like nutritional values and other facts on passing, or if asked.
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
            Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents. mph should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it.        Use only standard English alphabet characters [A-Z] along with basic punctuation. Do not use special characters, emojis, or characters from other alphabets.
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

    async def initialize(self):
        mic_info = self.pya.get_default_input_device_info()
        print(f"microphone used: {mic_info}")

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
        print("🗣️ Gemini talking")
        self.is_playing = True
        while self.audio_queue:
            try:
                audio_data = self.audio_queue.popleft()
                await asyncio.to_thread(self.output_stream.write, audio_data)
            except Exception as e:
                print(f"Error playing audio: {e}")

        self.is_playing = False

    def interrupt(self):
        """Handle interruption by stopping playback and clearing queue"""
        self.audio_queue.clear()
        self.is_playing = False

        # Important: Start a clean state for next response
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

# --- Video Helper Functions ---
async def _get_frame_data(cap):
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

async def _get_screen_data(sct, monitor):
    loop = asyncio.get_event_loop()
    try:
        sct_img = await loop.run_in_executor(None, sct.grab, monitor)
        if not sct_img:
            print("⚠️ Failed to capture screen.")
            return None
        img = PIL.Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
        img.thumbnail([768, 768])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg", quality=70)
        image_io.seek(0)
        image_bytes = image_io.read()
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode(),
        }
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None

# --- Main Conversation Loop ---
async def main_conversation_loop(video_mode: str):
    global CLIENT_INSTANCE, CURRENT_CLIENT_PROJECT_ID, CURRENT_CLIENT_LOCATION, LIVE_CONNECT_CONFIG

    needs_reinit = False
    if CLIENT_INSTANCE is None:
        needs_reinit = True
        print("Client is None, needs initialization.")

    if needs_reinit:
        print("Attempting to initialize/re-initialize Google GenAI Client with Project")
        try:
            CLIENT_INSTANCE = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print("Client successfully initialized/re-initialized.")
        except Exception as e:
            print(f"FATAL: Error initializing/re-initializing Google GenAI Client: {e}. Exiting.")
            traceback.print_exc()
            return

    if CLIENT_INSTANCE is None:
        print("FATAL: Google GenAI Client is not available. Exiting.")
        return
    print(f"Starting main loop with Video: {video_mode}")

    audio_manager = AudioManager(
        input_sample_rate=SEND_SAMPLE_RATE, output_sample_rate=RECEIVE_SAMPLE_RATE
    )

    await audio_manager.initialize()

    audio_send_queue = asyncio.Queue(maxsize=100)
    video_send_queue = asyncio.Queue(maxsize=10)
    stop_event = asyncio.Event()
    cap, sct = None, None  # Initialize for finally block

    try:
        # Use CLIENT_INSTANCE here
        async with (
            CLIENT_INSTANCE.aio.live.connect(
                model=DEFAULT_MODEL, config=LIVE_CONNECT_CONFIG
            ) as session,
            asyncio.TaskGroup() as tg,
        ):
            print("DEBUG: Entered TaskGroup and LiveConnect session.")
            video_capture_active_flag = False

            async def listen_for_audio():
                print("🎤 AudioListener: Listening...")
                while not stop_event.is_set():
                    try:
                        data = await asyncio.to_thread(
                            audio_manager.input_stream.read,
                            AUDIO_CHUNK_SIZE,
                            exception_on_overflow=False,
                        )
                        await audio_send_queue.put(data)
                    except IOError as e:
                        if hasattr(e, "errno") and e.errno == pyaudio.paInputOverflowed:
                            print("🎤 AudioListener: Input overflowed.")
                            continue
                        print(f"🎤 AudioListener: IOError: {e}.")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"🎤 AudioListener: Error: {e}")
                        if (
                            audio_manager.input_stream
                            and not audio_manager.input_stream.is_active()
                        ):
                            print("🎤 AudioListener: Input stream died. Re-init AudioManager.")
                            await audio_manager.initialize()
                        await asyncio.sleep(0.1)
                print("🎤 AudioListener: Stop event set. Exiting.")

            async def process_and_send_audio():
                print("🔊 AudioSender: Started.")
                while not stop_event.is_set() or not audio_send_queue.empty():
                    try:
                        data = await asyncio.wait_for(audio_send_queue.get(), timeout=0.5)
                        # Fixed: Use correct send method with proper format
                        await session.send({"data": data, "mime_type": f"audio/pcm;rate={AUDIO_SEND_SAMPLE_RATE}"})
                        audio_send_queue.task_done()
                    except asyncio.TimeoutError:
                        if stop_event.is_set() and audio_send_queue.empty():
                            break
                    except Exception as e:
                        print(f"🔊 AudioSender: Error: {e}")
                        await asyncio.sleep(0.1)
                print("🔊 AudioSender: Stop event. Exiting.")

            if video_mode == "camera":
                print("📹 VideoCapture: Initializing webcam...")
                cap = await asyncio.to_thread(cv2.VideoCapture, 0)
                if not cap.isOpened():  # Fixed: removed ZZZ
                    print("❌ VideoCapture: Cannot open webcam.")
                else:
                    video_capture_active_flag = True
                    print("📹 VideoCapture: Webcam started.")

                async def stream_video_frames_inner():
                    while video_capture_active_flag and not stop_event.is_set():
                        frame_media = await _get_frame_data(cap)
                        if frame_media:
                            await video_send_queue.put(frame_media)
                        else:
                            print("📹 VideoCapture: No frame_media from webcam.")
                        await asyncio.sleep(VIDEO_FRAME_RATE_DELAY)
                    if cap and cap.isOpened():  # Fixed: removed ZZZ
                        await asyncio.to_thread(cap.release)
                    print("📹 VideoCapture: Webcam task ended.")

                if video_capture_active_flag:
                    tg.create_task(stream_video_frames_inner(), name="WebcamStreamer")

            elif video_mode == "screen":
                print("🖥️ ScreenCapture: Initializing...")
                try:
                    sct = await asyncio.to_thread(mss.mss)
                    monitor = await asyncio.to_thread(lambda: sct.monitors[1])
                    video_capture_active_flag = True
                    print(f"🖥️ ScreenCapture: Started for monitor {monitor}.")

                    async def stream_screen_capture_inner():
                        while video_capture_active_flag and not stop_event.is_set():
                            screen_media = await _get_screen_data(sct, monitor)
                            if screen_media:
                                await video_send_queue.put(screen_media)
                            else:
                                print("🖥️ ScreenCapture: No screen_media.")
                            await asyncio.sleep(VIDEO_FRAME_RATE_DELAY)
                        print("🖥️ ScreenCapture: Screen task ended.")

                    if video_capture_active_flag:
                        tg.create_task(stream_screen_capture_inner(), name="ScreenStreamer")
                except Exception as e:
                    print(f"❌ ScreenCapture: Failed: {e}.")

            if video_capture_active_flag:
                async def process_and_send_video():
                    print(f"🖼️ VideoSender: Started (mode: {video_mode}).")
                    while not stop_event.is_set() or not video_send_queue.empty():
                        try:
                            video_data = await asyncio.wait_for(video_send_queue.get(), timeout=0.5)
                            # Fixed: Use correct send method format
                            await session.send(video_data)
                            video_send_queue.task_done()
                        except asyncio.TimeoutError:
                            if stop_event.is_set() and video_send_queue.empty():
                                break
                        except Exception as e:
                            print(f"🖼️ VideoSender: Error: {e}")
                            await asyncio.sleep(0.1)
                    print("🖼️ VideoSender: Stop event. Exiting.")

                tg.create_task(process_and_send_video(), name="VideoSender")

            async def receive_and_play():
                while True:
                    async for response in session.receive():
                        # Handle tool calls (function calls)
                        if response.tool_call:
                            print(f"📝 Tool call received: {response.tool_call}")

                            function_responses = []

                            for function_call in response.tool_call.function_calls:
                                name = function_call.name
                                args = function_call.args
                                call_id = function_call.id
                                print(f"🔔 Function call detected: name={name}, args={args}, id={call_id}")

                                # Handle get_recipe function
                                if name == "get_recipe":
                                    try:
                                        query = args["query"]
                                        print(f"🍳 Executing get_recipe for query: {query}")

                                        # Use the recipe agent
                                        result = await recipe_session_manager.send_message(query)
                                        function_responses.append(
                                            {
                                                "name": name,
                                                "response": {"result": result},
                                                "id": call_id,
                                            }
                                        )
                                        print(f"🍳 Recipe function executed for query: {query}")
                                    except Exception as e:
                                        print(f"Error executing recipe function: {e}")
                                        traceback.print_exc()

                            # Send function responses back to Gemini
                            if function_responses:
                                print(f"Sending function responses: {function_responses}")
                                # Fixed: Use correct tool response format
                                await session.send({
                                    "tool_response": {
                                        "function_responses": function_responses
                                    }
                                })
                                continue  # Skip to next response

                        # Handle audio/text output as usual
                        server_content = response.server_content
                        if server_content and server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data:
                                    audio_manager.add_audio(part.inline_data.data)
                                if part.text:
                                    print(f"ℹ️ Gemini (text): {part.text}")

                        if server_content and server_content.turn_complete:
                            print("✅ Gemini done talking")

            tg.create_task(listen_for_audio(), name="AudioListener")
            tg.create_task(process_and_send_audio(), name="AudioSender")
            tg.create_task(receive_and_play(), name="GeminiReceiver")
            print(f"🚀 All tasks started. Video mode: {video_mode}. Press Ctrl+C to exit.")
            await stop_event.wait()

    except KeyboardInterrupt:
        print("\n👋 KeyboardInterrupt. Shutting down...")
    except asyncio.CancelledError:
        print("Main conversation loop cancelled.")
    except Exception as e:
        print(f"💥 Unhandled exception in main_conversation_loop: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        stop_event.set()
        if "video_capture_active_flag" in locals():
            video_capture_active_flag = False  # Signal video loops
        print("Waiting for tasks to finish (1s)...")
        await asyncio.sleep(1.0)
        if audio_manager:
            audio_manager.close_streams()
        if video_mode == "camera" and cap and cap.isOpened():
            print("Releasing webcam.")
            await asyncio.to_thread(cap.release)
        if video_mode == "screen" and sct and hasattr(sct, "close"):
            print("Closing screen capture.")
            await asyncio.to_thread(sct.close)
        print("Application cleanup complete. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bidirectional audio/video streaming with Google Gemini Live API."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_VIDEO_MODE,
        help="Video streaming mode.",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    print(f"  Video Mode: {args.mode}")

    try:
        asyncio.run(main_conversation_loop(video_mode=args.mode))
    except KeyboardInterrupt:
        print("Application terminated by user (main __name__ block).")
    except Exception as e:
        print(f"Unhandled exception in __main__: {e}")
        traceback.print_exc()
    finally:
        print("Main execution finished.")