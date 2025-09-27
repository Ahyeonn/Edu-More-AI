# uv run uvicorn fastmain:app --host 0.0.0.0
# http://127.0.0.1:8000/
from fastapi import FastAPI, UploadFile, WebSocket, File
from get_hand_landmark import get_hand_landmark
import shutil
import os
import joblib

from LLM_STT import transcribe_audio, ask_gpt

app = FastAPI()

model = joblib.load("./training/models/hands.pkl")

upload_dir = "./audios"
os.makedirs(upload_dir, exist_ok=True)

@app.get('/')
def main():
    return '안녕하세요'

# GET wav audio file
# 클라이언트에서 요청 보낸걸 서버에서 wav파일을 받아 저장
@app.post("/get_audio")
async def get_audio_File(file: UploadFile = File(...)):
    # 확장자가 wav인지 체크 (선택사항)
    if not file.filename.lower().endswith(".wav"):
        return {"error": "Only .wav files are allowed"}
    file_path = os.path.join(upload_dir, file.filename)
    
    # 업로드된 파일을 저장
    with open(file_path, "wb") as buffer: # write + binary
        shutil.copyfileobj(file.file, buffer)

    # 음성 파일을 STT로 변환한 텍스트
    stt_text = transcribe_audio(file_path)

    # JSON 형태로 묶어서 전송
    payload = {
        "llm_result": ask_gpt(stt_text), # LLM 답변
        "user_question": stt_text # 음성을 텍스트로 변환한 질문
    }

    # json으로 응답 반환 -> 클라이언트에서 응답 확인
    return payload
