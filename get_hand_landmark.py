import cv2  # 객체 탐지할때 필요 라이브러리
import sys
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import json

def make_payload(xyz, state):
    return {"x_y_z": xyz, "state": state}

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

    labels_mapping = {0: "pause", 1: "move", 2: "stop", 3: "select"}
    model = joblib.load("./training/models/hands.pkl")
    feature_names = getattr(model, "feature_name_", None)

    start = True
    last_predict = None
    count = 0

    # 웹캠 연결
    vcap = cv2.VideoCapture(0)
    if not vcap.isOpened():
        print("카메라를 열 수 없습니다.")
        sys.exit()

    while True:
        ret, frame = vcap.read()
        if not ret:
            print("카메라가 작동하지 않습니다.")
            break

        # # 좌우 반전
        # frame = cv2.flip(frame, 1)

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

            # height, width, _ = frame.shape  # (h, w, c)

            for landmark in one_hand.landmark:
                # 좌표 뽑아오기
                hand_landmarks.extend([landmark.x, landmark.y, landmark.z])

                # # 좌표 데이터 모으기
                # point_x = int(landmark.x * width)
                # point_y = int(landmark.y * height)

                # # 원 그리기(src - 중심점 - 박지름 - 색상(cv2라 BGR형태로 만들어준다) - 두께
                # cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

            landmarks_feature = np.array(hand_landmarks, dtype=np.float32).reshape(1, -1)  # (1,63)
            X_infer = pd.DataFrame(landmarks_feature, columns=feature_names)

            pred = model.predict(X_infer)[0]
            state = labels_mapping[pred]
            # print(hand_landmarks)
            # print(pred, state)


            if last_predict != state:
                last_predict = state
                count = 1
            else:
                count += 1

            # ===== 웹소켓 전송용 페이로드 =====
            payload = None

            # 일시정지 상태가 아니고(start == True) 손이 움직일때 좌표만 보내기 | count 조건문 영향 x 
            if last_predict == "move" and start == True:
                payload = make_payload(hand_landmarks, None)
                # print("move", json.dumps(payload), "last_predict:", last_predict, "start:", start, "count:", count)
            
            # 일시정지 상태가 아닌데(start == True) 일시정지 상태로 변환시 좌표 없이 state 값만 보내기
            elif last_predict == "pause" and start == True:
                if count >= 50:
                    start = False

                    payload = make_payload(None, "pause")
                    count = 0
                    # print("pause 50", json.dumps(payload), "last_predict:", last_predict, "count:", count)
                
                # 일시정지 상태로 변환할 조건인 count 가 50 미만일시 좌표만 보내기
                else:
                    payload = make_payload(hand_landmarks, None)

                    # print(f"pause {count}", json.dumps(payload))

                
            # 일시정지 상태인데 (start == False) 시작상태로 변환시 state 값과 좌표 같이 보내기
            elif last_predict == "pause" and start == False:
                if count >= 50:
                    start = True

                    payload = make_payload(hand_landmarks, "resume")
                    count = 0
                    
                    # print("start 50:", json.dumps(payload), "last_predict:", last_predict, "count:", count)
            

            # 일시정지 상태가 아니고(start == True) 현재 창을 닫을시 state 값과 좌표 같이 보내기
            elif last_predict == "stop" and start == True:
                if count >= 50:
                    payload = make_payload(hand_landmarks, "stop")
                    count = 0
                    # print("stop 50:", json.dumps(payload), "last_predict:", last_predict, "count:", count)
                                    
                # 일시정지 상태가 아니고 현재 창을 닫을 조건인 count 가 50 미만일시 좌표만 보내기
                else:
                    payload = make_payload(hand_landmarks, None)

                    # print(f"stop {count}:", json.dumps(payload))


            # 일시정지 상태가 아니고(start == True) 선택 할시 state 값과 좌표 같이 보내기
            elif last_predict == "select" and start == True:
                if count >= 50:
                    payload = make_payload(hand_landmarks, "select")
                    count = 0
                    # print("select 30", json.dumps(payload), "last_predict:", last_predict, "count:", count)
                
                # 일시정지 상태가 아니고 선택 조건인 count 가 50 미만일시 좌표만 보내기
                else:
                    payload = make_payload(hand_landmarks, None)
                    # print(f"select {count}", json.dumps(payload))

            # print(f"{state}{count}\n")

            # 제너레이터 함수
            if payload is not None: 
                yield payload # 외부 웹소켓으로 넘김

        # 화면 띄우기
        cv2.imshow("웹캠 시작", frame)

        # 꺼지는 조건
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 버튼을 누르면 종료
            break

    vcap.release()
    cv2.destroyAllWindows()


