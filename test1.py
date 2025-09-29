# client.py
import requests
import os

SERVER_URL = "http://172.16.20.183:8000"  # FastAPI 서버 주소
FILE_PATH = "C:/Edu-More_AI/uploads/sample.wav"  # 업로드할 wav 파일 경로

# 1. wav 파일 업로드
with open(FILE_PATH, "rb") as f:
    files = {"file": (os.path.basename(FILE_PATH), f, "audio/wav")}
    response = requests.post(f"{SERVER_URL}/get_audio", files=files)

print("업로드 응답:", response.json())


# # my_server.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn

# app = FastAPI()

# # LLM에서 반환한 텍스트 구조
# class Result(BaseModel):
#     filename: str
#     parsed_text: str

# @app.post("/receive_result/")
# async def receive_result(result: Result):
#     print(f"파일명: {result.filename}")
#     print(f"LLM 결과: {result.parsed_text}")
#     return {"status": "OK", "message": "결과 잘 받았습니다"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)  # 9000 포트로 실행
