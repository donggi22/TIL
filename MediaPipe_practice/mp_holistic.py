import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Pose + Hands + FaceMesh 합친 버전 (올인원 모델)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 처리
        results = holistic.process(image)

        # RGB → BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 🔥 얼굴
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS
            )

        # 🔥 포즈 (몸)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # 🔥 왼손
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # 🔥 오른손
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow('Holistic', image)

        # ESC 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()