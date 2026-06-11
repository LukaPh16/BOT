import cv2
import mediapipe as mp

face_count = 0

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def start_face_detection():
    global face_count

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_detection.process(rgb)

            face_count = 0

            if results.detections:
                face_count = len(results.detections)

                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("FRIDAY Vision", frame)

            if cv2.waitKey(1) == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


def get_face_count():
    return face_count


if __name__ == "__main__":
    start_face_detection()