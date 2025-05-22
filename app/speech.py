from gtts import gTTS
#import pyttsx3  # Uncomment if you want to use pyttsx3 instead of gTTS if windows error occurs
#pyttsx3 is a text-to-speech conversion library in Python. It works offline and is compatible with both Python 2 and 3.
from pydub import AudioSegment
from playsound import playsound
import os
import uuid
import time

def safe_delete(path, retries=5, delay=0.5):
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
        except PermissionError:
            time.sleep(delay)
    print(f"[⚠️] Could not delete {os.path.basename(path)} after {retries} retries.")
    return False

def text_to_speech(text):
    print("\n[🗣️  Generating speech output...]")

    # Generate unique filename in script folder
    base_path = os.path.dirname(os.path.abspath(__file__))
    filename_base = f"speech_{uuid.uuid4().hex}"
    mp3_path = os.path.join(base_path, f"{filename_base}.mp3")
    wav_path = os.path.join(base_path, f"{filename_base}.wav")

    try:
        # Generate TTS and convert to wav
        tts = gTTS(text)
        tts.save(mp3_path)

        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

        # Play the audio
        print(f"[🔊 Playing audio]: {text}")
        playsound(wav_path)

    finally:
        time.sleep(0.5)  # short pause before attempting deletion
        safe_delete(mp3_path)
        safe_delete(wav_path)

    print("[✅ Done playing audio]\n")
