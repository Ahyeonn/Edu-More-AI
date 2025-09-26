# uv add mediapipe
import sys
import cv2
import joblib
import mediapipe as mp
import numpy as np

# mediapipe의 Hand Landmark를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# 모델 불러오기
model = joblib.load("C:/Edu-More-AI/training/.hands.pkl")
labels = ["pause", "select", "stop"]
# 웹캠 연결
vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()

    # 좌우 반전
    frame = cv2.flip(frame, 1)

    ###### Hands Landmark 설정하기 ######
    # 손 그리기 준비
    frame.flags.writeable = True

    # 손 감지하기
    results = hands.process(frame)

    # 추출 및 그리기
    if results.multi_hand_landmarks:
        ### 손 하나하나 탐색
        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = frame.shape

            landmarks = []
            for landmark in hand_landmarks.landmark:
                ## 좌표 데이터
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                ## 그리기
                point_x = int(landmark.x * width)
                point_y = int(landmark.y * height)

                cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

            # 예측
            pred = model.predict(np.array([landmarks]))
            cv2.putText(
                frame,
                labels[pred[0]],
                (10, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                2,
            )
    ####################################

    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()
