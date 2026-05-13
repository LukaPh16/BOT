# import ollama

# response = ollama.chat(model = "qwen2.5:0.5b", messages=[
#     {
#         'role': 'user',
#         'content': ""
#     }
# ])

# print(response['message']['content'])

import cv2

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

print("Opened", cap.isOpened())

ret, frame = cap.read()

print("frame", ret)