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
import json
import re

import datetime
import serial

import os

model = whisper.load_model("base", device="cuda")

TTS_MODEL = "en_US-hfc_female-medium.onnx"
TTS_CONFIG = "en_US-hfc_female-medium.onnx.json"

OLLAMA_MODEL = "qwen2.5:0.5b"

voice = PiperVoice.load(TTS_MODEL, config_path=TTS_CONFIG)

ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200

arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout = 1)
time.sleep(2)

print("Connected to Arduino")

NAME = "PINKIE" #Still have to think about the name
CALLNAME = "SIR"

RATE = 48000 #mic
TARGET_RATE = 16000
CHUNK_SIZE = 1024

DEVICE_INDEX = 4 #mic / 4 / 24
MIN_VOLUME = 65 
SILENCE_THRESHOLD= 60
SILENCE_LIMIT = 20

assistant_awake = False

examples = [
    "User: Hello\nAI: Hello, sir.",
    "User: who are you\nAI: I am your personal assistant, sir.",
    "User: what is your purpose\nAI: To assist you efficiently, sir."
]


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

def warmup_ollama():
    print("Warming up Ollama...")
    
    ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options = {"num_predict": 1}
    )

    print("Ollama ready!")

warmup_ollama()
os.environ["OLLAMA_KEEP_ALIVE"] = "30m"

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

def enforce_callname(text):
    if CALLNAME.lower() not in text.lower():
        text = text.strip()

        if text.endswith((".", "!", "?")):
            text = text[:-1] + f", {CALLNAME}."
        
        else:
            text += f", {CALLNAME}"
    
    return text

def limit_sentences(text, max_sentences = 2):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:max_sentences])

def ask_ai(prompt):
    example_text = "\n".join(examples[-10:])

    full_prompt = (
        f"You must ALWAYS address the user as '{CALLNAME}'. "
        f"Every answer MUST include '{CALLNAME}'. "
        "You must answer in MAXIMUM 3 shorts sentences. "
        "Be concise and direct. \n\n"
        + example_text + f"\nUser: {prompt}\nAI:"
    )

    response = ollama.chat(
        model = OLLAMA_MODEL,
        messages = [{"role": "user", "content": full_prompt}],
        options = {
            "num_predict": 30, #reply size
            "temperature": 0.3
        }
    )

    reply = response["message"]["content"]
    reply = limit_sentences(reply)
    reply = enforce_callname(reply)
    print(f"{NAME}: ", reply)
    return reply

def save_examples():
    with open("personality.json", "w") as f:
        json.dump(examples, f)

def load_examples():
    global examples
    try:
        with open("personality.json", "r") as f:
            examples = json.load()
    
    except:
        pass

def send(command):
    arduino.write((command + "\n").encode())

def set_mode(mode):
    send(mode)

def tell_time(text):
    text = text.lower()
    now = datetime.datetime.now()

    if " time " in text:
        return now.strftime(f"The time is %H:%M, {CALLNAME}")

    if " date " in text:
        return now.strftime(f"Today is %B %d, {CALLNAME}")
    
    if " year " in text:
        return now.strftime(f"It is %Y, {CALLNAME}")
    
    if " day " in text:
        return now.strftime(f"Today is %A, {CALLNAME}")

    return None

def main():
    global examples

    set_mode("IDLE")

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        db = get_db(data)
        print(f"Volume: {db:6.2f} dB   ", end="\r", flush=True)

        if db > MIN_VOLUME:
            set_mode("LISTEN")
            print("\nSpeaking detected...")

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
            set_mode("THINK")

            audio_data = b"".join(frames)

            # Convert 48kHz → 16kHz
            data_16k, _ = audioop.ratecv(audio_data, 2, 1, RATE, TARGET_RATE, None)

            audio_np = np.frombuffer(data_16k, np.int16).astype(np.float32) / 32768.0

            result = model.transcribe(audio_np, language="en")
            user_input = result["text"].strip()

            print("You said:", user_input)

            if user_input.lower() in ["goodbye", "goodbye.", "bye", "bye."]:
                reply = f"Goodbye {CALLNAME}!"
                print(f"{NAME}: {reply}")

                set_mode("TALK")
                tts.speak(reply)
                set_mode("OFF")

                sys.exit(0)

            if not user_input:
                set_mode("IDLE")
                continue

            reply = tell_time(user_input)

            if reply is None:
                reply = ask_ai(user_input)


            if reply and len(reply) < 200 and tell_time(user_input) is None:
                examples.append(f"User: {user_input}\nAI: {reply}")
                examples = examples[-50:]
                save_examples()

            set_mode("TALK")
            tts.speak(reply)

            set_mode("IDLE")

load_examples()

identity = f"User: what is your name\nAI: I am {NAME}, your assistant, {CALLNAME}"

if identity not in examples:
    examples.insert(0, identity)

try:
    main()
except KeyboardInterrupt:
    set_mode("OFF")
    sys.exit(0)