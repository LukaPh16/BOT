import sys
import whisper
import pyaudio
import numpy as np
import audioop
import time

model = whisper.load_model("base", device = "cuda")  # remove cuda unless confirmed working

RATE = 48000
TARGET_RATE = 16000
CHUNK_SIZE = 1024  # small = real-time

DEVICE_INDEX = 4 #mic
MIN_VOLUME = 65  # dB threshold (adjust later)
DELAY = 1.5

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

    if rms > 0:
        db = 20 * np.log10(rms)
    else:
        db = -100  # silence

    return db


def db():
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        db = get_db(data)
        print(f"Volume: {db:.2f} dB    ", end="\r")

        if db > MIN_VOLUME:
            print("Speaking detected!", end="\r")
            time.sleep(DELAY)


try:
    db()

except KeyboardInterrupt:
    sys.exit(0)