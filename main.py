import whisper
import pyaudio
import numpy as np
import audioop

model = whisper.load_model("base", device = "cuda") 

RATE = 48000 #mic 
TARGET_RATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION)

p = pyaudio.PyAudio()

DEVICE_INDEX = 4 #mic

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=1024
)

print("Listening...")

while True:
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

    data_16k, _ = audioop.ratecv(data, 2, 1, RATE, TARGET_RATE, None)

    audio_np = np.frombuffer(data_16k, np.int16).astype(np.float32) / 32768.0

    result = model.transcribe(audio_np, language="en")

    print("You said:", result["text"])