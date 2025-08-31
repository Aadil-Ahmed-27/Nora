# Nora - AI Grocery Assistant with Gradio UI

A modern, minimal Gradio interface for the multimodal AI chat application featuring Nora, your friendly grocery and recipe assistant. This application provides natural conversation flow with interruption handling, making it feel like talking to a real person over a video call.

## Features

- 🎤 **Voice Conversation**: Natural speech-to-speech interaction
- 📹 **Video Support**: Camera and screen sharing capabilities
- 🛑 **Smart Interruption**: AI pauses when you start speaking (natural conversation flow)
- 🛒 **Grocery Expertise**: Specialized in groceries, recipes, and cooking tips
- 🌐 **Modern UI**: Clean, minimal Gradio interface
- 🔄 **Real-time**: Asynchronous processing for smooth interaction

## Prerequisites

- Python 3.8 or higher
- Microphone access
- Camera access (optional, for video mode)
- Google Gemini API key

## Installation

1. **Clone or download the files**:
   ```bash
   # Make sure you have these files:
   # - main.py
   # - requirements.txt
   # - .env.example
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Gemini API key
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

4. **Get your Gemini API key**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Copy the key to your `.env` file

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Open your browser**:
   - The app will automatically open at `http://localhost:7860`
   - Or manually navigate to the URL shown in the terminal

3. **Choose your mode**:
   - **Audio Only**: Select "none" for voice-only conversation
   - **Camera**: Select "camera" to show Nora what you're looking at
   - **Screen Share**: Select "screen" to share your screen with Nora

4. **Start talking**:
   - Click "🚀 Start Conversation"
   - Begin speaking naturally
   - Nora will respond with voice
   - You can interrupt her at any time by speaking

## Key Features Explained

### Natural Interruption Handling
- When Nora is speaking and you start talking, she will immediately pause
- This creates a natural conversation flow like talking to a real person
- No need to wait for her to finish - just start speaking!

### Grocery & Recipe Focus
- Ask about any grocery items, recipes, or cooking tips
- Get nutritional information and cooking advice
- Nora uses specialized recipe tools for accurate information

### Video Capabilities
- **Camera mode**: Show Nora ingredients, products, or cooking processes
- **Screen mode**: Share recipes, shopping lists, or cooking videos
- **Audio only**: Perfect for hands-free cooking assistance

## Example Conversations

- "Hey Nora, what can I make with chicken and broccoli?"
- "Show me how to pick a good avocado" (with camera)
- "What's the nutritional value of quinoa?"
- "I need a quick dinner recipe for tonight"

## Troubleshooting

### Audio Issues
- Make sure your microphone is working and permissions are granted
- Check that no other applications are using the microphone
- Try adjusting the volume levels

### Camera Issues
- Ensure camera permissions are granted to your browser
- Try refreshing the page if camera doesn't initialize
- Check that no other applications are using the camera

### API Issues
- Verify your Gemini API key is correct in the `.env` file
- Check your internet connection
- Ensure you have API quota available

### Performance Issues
- Close other resource-intensive applications
- Try using "Audio Only" mode for better performance
- Restart the application if it becomes unresponsive

## Technical Details

- **Framework**: Gradio for the web interface
- **AI Model**: Google Gemini 2.5 Flash Live Preview
- **Audio**: PyAudio for real-time audio processing
- **Video**: OpenCV for camera, MSS for screen capture
- **Async**: Full asyncio implementation for smooth real-time interaction

## Customization

You can modify the system prompt in `main.py` to change Nora's personality or focus areas, but the current prompt is optimized for natural conversation and grocery assistance.

## License

This project is for educational and personal use. Please respect Google's Gemini API terms of service.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your API key is valid and has quota
4. Check the console output for error messages

Enjoy talking with Nora! 🛒✨

