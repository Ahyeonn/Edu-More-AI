# uv run uvicorn fastmain:app --host 0.0.0.0
# http://127.0.0.1:8000/
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, File, Request
from get_hand_landmark import get_hand_landmark  # 제너레이터 함수 import
import shutil
import os
import joblib
import json, asyncio, threading

from edumore_llm_fastapi import transcribe_audio, ask_gpt

app = FastAPI()

model = joblib.load("./training/models/hands.pkl")

upload_dir = "./audios"
os.makedirs(upload_dir, exist_ok=True)


@app.get("/")
def main():
    return "안녕하세요"

# GET wav audio file
# 클라이언트에서 요청 보낸걸 서버에서 wav파일을 받아 저장
@app.post("/get_audio")
async def get_audio_file(request: Request):
    # raw body 그대로 읽기
    body = await request.body()

    # 업로드된 파일을 저장
    file_path = os.path.join(upload_dir, "record.wav")
    with open(file_path, "wb") as f:
        f.write(body)

    # 음성 파일 STT 변환
    stt_text = transcribe_audio(file_path)

    # JSON 형태 응답 반환 -> 클라이언트에서 응답 확인
    payload = {
        "llm_result": ask_gpt(stt_text),  # LLM 답변
        "user_question": stt_text,  # 음성을 텍스트로 변환한 질문
    }

    return payload


# 언리얼(클라이언트)이 여기에 연결한다고 가정: ws://localhost:8000/ws/hand
# 제너레이터 스레드로 분리해서 WebScoket 은 비동기 큐로 받기
@app.websocket("/ws/hand")
async def websocket_hand(websocket: WebSocket):
    await websocket.accept()

    # 현재 실행 중인 asyncio 이벤트 루프를 '여기서' 잡아둔다.
    loop = asyncio.get_running_loop()
    # 비동기 큐 (스레드 -> 이벤트루프 전달용)
    q = asyncio.Queue()

    stop_flag = False  # 간단한 종료 플래그

    # 동기 제너레이터를 돌릴 작업 스레드
    def worker():
        # nonlocal stop_flag  # 바깥 변수 참조
        try:
            for payload in get_hand_landmark():
                if stop_flag:
                    break
                # 스레드에서 직접 q.put() 하지 말고, 메인 루프에 "이 코루틴 실행해줘"라고 부탁한다.
                asyncio.run_coroutine_threadsafe(q.put(payload), loop)
        finally:
            # 종료 신호(None)도 같은 방식으로 보낸다.
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    try:
        while True:
            payload = await q.get()
            if payload is None:
                break
            await websocket.send_text(json.dumps(payload))

        await websocket.send_text("[END]")
        # # 클라이언트의 시작 메시지 받기
        # data = await websocket.receive_json()  # {"cmd": "start"}
        # print("client says:", data)

    except WebSocketDisconnect:
        print("클라이언트가 연결 종료")
    except Exception as e:
        print("WebSocket 에러:", e)
        try:
            await websocket.send_text("[END]")
        except:
            pass
    finally:
        stop_flag = True
        try:
            await websocket.close()
        except:
            pass
        print("WebSocket 연결 종료")
