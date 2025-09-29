import cv2  # 객체 탐지할때 필요 라이브러리
import sys
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import json


def get_hand_landmark():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        # 영상 스트리밍 모드
        static_image_mode=False,
        max_num_hands=1,
        # 손을 처음 감지할 때 최소 신뢰도 (0~1, 값이 클수록 더 확실해야 검출됨)
        min_detection_confidence=0.7,
        # 손 랜드마크를 계속 추적할 때 최소 신뢰도 (0~1)
        min_tracking_confidence=0.7,
    )

    model = joblib.load("./training/models/hands.pkl")
    feature_names = getattr(model, "feature_name_", None)
    labels_mapping = {0: "pause", 1: "move", 2: "stop", 3: "select"}

    start = True

    # 웹캠 연결
    vcap = cv2.VideoCapture(0)

    while True:
        ret, frame = vcap.read()
        if not ret:
            print("카메라가 작동하지 않습니다.")
            sys.exit()

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # 포즈 감지는 RGB 권장
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 손 그리기 설정
        rgb_frame.flags.writeable = False
        # 손 감지하기
        mediapipe_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_landmarks = []

        # 손 감지 시 랜드마크 그리기
        if mediapipe_results.multi_hand_landmarks:  # 한개의 손 탐지
            one_hand = mediapipe_results.multi_hand_landmarks[
                0
            ]  # 한개의 손 landmark는 리스트형태

            height, width, _ = frame.shape  # (h, w, c)

            for landmark in one_hand.landmark:
                # 좌표 뽑아오기
                hand_landmarks.extend([landmark.x, landmark.y, landmark.z])

                # # 좌표 데이터 모으기
                # point_x = int(landmark.x * width)
                # point_y = int(landmark.y * height)

                # # 원 그리기(src - 중심점 - 박지름 - 색상(cv2라 BGR형태로 만들어준다) - 두께
                # cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

            landmarks_feature = np.array(hand_landmarks, dtype=np.float32).reshape(
                1, -1
            )  # (1,63)
            X_infer = pd.DataFrame(landmarks_feature, columns=feature_names)

            pred = model.predict(X_infer)[0]
            state = labels_mapping[pred]
            # if time.sleep(3)
            #     pred = model.predict(X_infer)[0]

            #     state = labels_mapping[pred]
            #     print(state)
            # else:
            #     pred = None

            # 일시정지 하기
            if state == "pause" and start == True:
                start = False
            # 시작하기
            elif state == "pause" and start == False:
                start = True
            # print(hand_landmarks)
            # print(pred, state)

            # ===== 웹소켓 전송용 페이로드 예시 =====
            payload = {
                "x_y_z": None if start == False else hand_landmarks,
                "state": None if pred is None else state,
            }

            # 여기서 나중에 websocket.send(json.dumps(payload)) 하면 됩니다.
            # 지금은 디버그 출력
            print(json.dumps(payload, ensure_ascii=False))

        # 화면 띄우기
        cv2.imshow("webcam - hand detection", frame)

        # 꺼지는 조건
        key = cv2.waitKey(1)
        if key == 27:  # ESC 버튼을 누르면 종료
            break

    vcap.release()
    cv2.destroyAllWindows()


get_hand_landmark()
