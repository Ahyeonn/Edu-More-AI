# uv run uvicorn fastmain:app --reload
# uv run uvicorn fastmain:app --host 0.0.0.0
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import os

app = FastAPI()

upload_dir = "./audios"
os.makedirs(upload_dir, exist_ok=True)

@app.get('/')
def main():
    return {'result': '안녕하세요'}


# wav파일 가져와 저장
@app.post("/get_audio")
async def get_audio_File(file: UploadFile = File(...)):
    # ① 디렉토리 제거 → 순수 파일명만
    filename = Path(file.filename).name.strip()

    # ② 확장자 체크 (대소문자 무시)
    if Path(filename).suffix.lower() != ".wav":
        return {"error": "Only .wav files are allowed"}
    
    # 확장자가 wav인지 체크 (선택사항)
    if not file.filename.lower().endswith(".wav"):
        return {"error": "Only .wav files are allowed"}
    file_path = os.path.join(upload_dir, file.filename)
    
    # 업로드된 파일을 저장
    with open(file_path, "wb") as buffer: # write + binary
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"File '{file.filename}' saved successfully", "path": file_path}