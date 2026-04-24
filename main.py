import sys
import whisper
import pyaudio
import numpy as np
import audioop
import threading
import time
import ollama

from queue import Queue
from threading import Thread, Event

from piper.voice import PiperVoice
import sounddevice as sd
import soundfile as sf
import wave

import io

model = whisper.load_model("base", device="cuda")

TTS_MODEL = "en_US-hfc_female-medium.onnx"
TTS_CONFIG = "en_US-hfc_female-medium.onnx.json"

OLLAMA_MODEL = "qwen2.5:0.5b"

voice = PiperVoice.load(TTS_MODEL, config_path=TTS_CONFIG)

NAME = "BOT" #Still have to think about the name
CALLNAME = "SIR"

RATE = 48000 #mic
TARGET_RATE = 16000
CHUNK_SIZE = 2048

DEVICE_INDEX = 4 #mic / 4 / 24
MIN_VOLUME = 65 
SILENCE_THRESHOLD= 60
SILENCE_LIMIT = 20


p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK_SIZE
)

print("Listening...")

def get_db(block):
    data = np.frombuffer(block, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(data**2))
    return 20 * np.log10(rms) if rms > 0 else -100
    
class TTSEngine:
    def __init__(self, voice):
        self.voice = voice #store voice model
        self.queue = Queue() #message buffer
        self.running = True #running flag
        self.thread = Thread(target=self.worker, daemon=True) #background thread
        self.thread.start() #start thread

    def speak(self, text):
        if not text:
            return
        
        done = Event() #pausing until speech is done
        self.queue.put((text, done))
        done.wait()

    def worker(self):
        while self.running:
            text, done = self.queue.get() #sleeping until someone talks

            try:
                buffer = io.BytesIO() #virtual file in RAM

                with wave.open(buffer, "wb") as f: #generate audio into memory
                    self.voice.synthesize_wav(text, f)

                buffer.seek(0) #reset buffer position

                data, fs = sf.read(buffer) #load from memory
                
                sd.play(data, fs) #play audio
                sd.wait()
            
            except Exception as e:
                print("TTS error:", e)

            done.set()
            self.queue.task_done()

tts = TTSEngine(voice)

def ask_ai(prompt):
    response = ollama.chat(
        model = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}],
        options = {
            "num_predict": 30 #reply size
        }
    )

    reply = response["message"]["content"]
    print(f"{NAME}: ", reply)
    return reply
    

def main():
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        db = get_db(data)
        print(f"Volume: {db:6.2f} dB   ", end="\r", flush=True)

        if db > MIN_VOLUME:
            print("\n🎤 Speaking detected...")

            frames = []
            silence_count = 0

            while True:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(chunk)

                db = get_db(chunk)

                if db < SILENCE_THRESHOLD:
                    silence_count += 1

                else:
                    silence_count = 0

                if silence_count > SILENCE_LIMIT:
                    break

            print("Processing...")

            audio_data = b"".join(frames)

            #Convert 48KHz to 16KHz
            data_16k, _ = audioop.ratecv(audio_data, 2, 1, RATE, TARGET_RATE, None)

            audio_np = np.frombuffer(data_16k, np.int16).astype(np.float32) / 32768.0

            result = model.transcribe(audio_np, language="en")

            user_input = result["text"]

            print("You said:", user_input)
            
            if not user_input:
                continue

            tts.speak(f"One moment. {CALLNAME}")

            reply = ask_ai(user_input)

            tts.speak(reply)

        


try:
    main()
except KeyboardInterrupt:
    sys.exit(0)