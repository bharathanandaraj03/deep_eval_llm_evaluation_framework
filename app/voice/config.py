import os

STT_MODEL = os.getenv("VOICE_STT_MODEL","gpt-4o-transcribe")
TTS_MODEL = os.getenv("VOICE_TTS_MODEL","gpt-4o-mini-tts")
TTS_VOICE = os.getenv("VOICE_TTS_VOICE","alloy")