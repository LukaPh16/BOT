import cv2
import time

face_count = 0

def start_face_detection():
    global face_count

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            continue


        small = cv2.resize(frame, None, fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )

        faces = faces[:3]
        face_count = len(faces)

        for (x, y, w, h) in faces:
            x *= 4
            y *= 4
            w *= 4
            h *= 4

            cv2.rectangle(frame,
                          (x, y),
                          (x+w, y+h),
                          (0,255,0), 2)

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("face detection", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

def get_face_count():
    return face_count

if __name__ == "__main__":
    start_face_detection()