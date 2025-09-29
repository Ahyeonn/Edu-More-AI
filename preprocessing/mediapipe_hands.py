import sys
import cv2
import mediapipe as mp
import csv
from pathlib import Path

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

# 저장할 데이터 폴더 만들기
out_dir = Path("./data/raws")
out_dir.mkdir(parents=True, exist_ok=True)

# 저장할 파일 만들기
file_names = {
    "pause": "hand_pause_data.csv",
    "move": "hand_move_data.csv",
    "stop": "hand_stop_data.csv",
    "select": "hand_select_data.csv",
}

file_paths = {}

for key, name in file_names.items():
    path = out_dir / name
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f)  # 그냥 빈 파일 생성
    file_paths[key] = path

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

    # 손 감지 시 랜드마크 그리기
    if mediapipe_results.multi_hand_landmarks:  # 한개의 손 탐지
        one_hand = mediapipe_results.multi_hand_landmarks[
            0
        ]  # 한개의 손 landmark는 리스트형태

        landmarks = []
        height, width, _ = frame.shape  # (h, w, c)

        for landmark in one_hand.landmark:
            # 좌표 뽑아오기
            landmarks.extend([landmark.x, landmark.y, landmark.z])

            # 좌표 데이터 모으기
            point_x = int(landmark.x * width)
            point_y = int(landmark.y * height)

            # 원 그리기(src - 중심점 - 박지름 - 색상(cv2라 BGR형태로 만들어준다) - 두께
            cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

        # (주먹 = 일시 정지/다시 시작) 1번 누르면 data/hand_pause_data.csv로 저장
        if key == ord("1"):
            # 정답 라벨 추가
            landmarks.append("pause")
            # 데이터 추가
            with open(file_paths["pause"], "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(
                    frame,
                    "Save Pause",
                    (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

        # (이동 = 선택) 2번 누르면 data/hand_select_data.csv로 저장
        elif key == ord("2"):
            # 정답 라벨 추가
            landmarks.append("move")
            # 데이터 추가
            with open(file_paths["move"], "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(
                    frame,
                    "Save move",
                    (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

        # (손펴기 = 종료) 3번 누르면 data/hand_stop_data.csv로 저장
        elif key == ord("3"):
            # 정답 라벨 추가
            landmarks.append("stop")
            # 데이터 추가
            with open(file_paths["stop"], "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(
                    frame,
                    "Save Stop",
                    (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
        
        elif key == ord("4"):
            # 정답 라벨 추가
            landmarks.append("select")
            # 데이터 추가
            with open(file_paths["select"], "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(
                    frame,
                    "Save Select",
                    (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

    # 화면 띄우기
    cv2.imshow("webcam - hand detection", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:  # ESC 버튼을 누르면 종료
        break

vcap.release()
cv2.destroyAllWindows()
