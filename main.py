import sys
import whisper
import pyaudio
import numpy as np
import audioop
import subprocess

model = whisper.load_model("base", device="cuda")

RATE = 48000 #mic
TARGET_RATE = 16000
CHUNK_SIZE = 1024

DEVICE_INDEX = 4 #mic
MIN_VOLUME = 65 
SILENCE_THRESHOLD= 60
SILENCE_LIMIT = 20

TTS_MODEL = "en_US-hfc_female-medium.onnx"

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

def speak(text):
    #generate wav file
    subprocess.run(
        [
            "piper",
            "--model", TTS_MODEL,
            "--output_file", "out.wav"
        ],
        input=text.encode()
    )

    #play wav file
    subprocess.run(["aplay", "out.wav"])


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

            print("You said:", result["text"])

            speak(result["text"])

        


try:
    main()
except KeyboardInterrupt:
    sys.exit(0)