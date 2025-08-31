#!/usr/bin/env python3
"""
Basic test script to verify that all imports work correctly
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import asyncio
        print("✅ asyncio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import asyncio: {e}")
        return False
    
    try:
        import gradio as gr
        print("✅ gradio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import gradio: {e}")
        return False
    
    try:
        import pyaudio
        print("✅ pyaudio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pyaudio: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import opencv-python: {e}")
        return False
    
    try:
        import PIL.Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pillow: {e}")
        return False
    
    try:
        import mss
        print("✅ mss imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import mss: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        from google import genai
        print("✅ google-genai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google-genai: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import python-dotenv: {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("\nTesting environment...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        if api_key == "test_key_placeholder":
            print("⚠️  Using test placeholder API key - replace with real key for actual use")
        else:
            print("✅ GEMINI_API_KEY found in environment")
    else:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    return True

def test_audio_system():
    """Test audio system availability"""
    print("\nTesting audio system...")
    
    try:
        import pyaudio
        pya = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        for i in range(pya.get_device_count()):
            device_info = pya.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        
        if input_devices:
            print(f"✅ Found {len(input_devices)} audio input device(s)")
            print(f"   Default input: {pya.get_default_input_device_info()['name']}")
        else:
            print("⚠️  No audio input devices found")
        
        # Check for output devices
        output_devices = []
        for i in range(pya.get_device_count()):
            device_info = pya.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                output_devices.append(device_info['name'])
        
        if output_devices:
            print(f"✅ Found {len(output_devices)} audio output device(s)")
            print(f"   Default output: {pya.get_default_output_device_info()['name']}")
        else:
            print("⚠️  No audio output devices found")
        
        pya.terminate()
        return len(input_devices) > 0 and len(output_devices) > 0
        
    except Exception as e:
        print(f"❌ Audio system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running basic tests for Nora AI Assistant\n")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test environment
    if not test_environment():
        all_passed = False
    
    # Test audio system
    if not test_audio_system():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! The application should work correctly.")
        print("\nNext steps:")
        print("1. Set your real GEMINI_API_KEY in the .env file")
        print("2. Run: python main.py")
        print("3. Open your browser to the displayed URL")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

