import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0,  # 0: 근거리, 1: 원거리
    min_detection_confidence=0.5
) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # 다시 BGR로
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('Face Detection', image)

        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()