import cv2

def start_face_detection():
    # start face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Started Face Detection. Press Q to Quit")

    while True:
        ret, frame = cap.read()


        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )

        for (x, y, w, h) in faces:
            x *= 2
            y *= 2
            w *= 2
            h *= 2

            cv2.rectangle(frame,
                          (x, y),
                          (x+w, y+h),
                          (0,255,0), 2)

        cv2.imshow("face detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_face_detection()