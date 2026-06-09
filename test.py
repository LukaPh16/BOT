# import ollama

# response = ollama.chat(model = "qwen2.5:0.5b", messages=[
#     {
#         'role': 'user',
#         'content': ""
#     }
# ])

# print(response['message']['content'])

# import cv2

# cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

# print("Opened", cap.isOpened())

# ret, frame = cap.read()

# print("frame", ret)

# import pyaudio

# p = pyaudio.PyAudio()

# print("Available audio devices:")

# for i in range(p.get_device_count()):
#     info = p.get_device_info_by_index(i)
#     print(
#         i,
#         "|",
#         info["name"],
#         "| inputs:",
#         info["maxInputChannels"]
#     )

import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)

    if info["maxInputChannels"] > 0:
        print(i, info["name"])