# uv run uvicorn fastmain:app --reload
# uv run uvicorn fastmain:app --host 0.0.0.0

import os 
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()

# . wav 오디오 가져와서 듣고 텍스트 변환(STT)
def transcribe_audio(file_path):
    with open(file_path,"rb" )as f: #read + binary
          transcript = client.audio.transcriptions.create(
                model = "gpt-4o-transcribe", file = f # STT 모델 
          )
    return transcript.text.strip()
# chat(question) -> 답변이 나오게 
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def ask_gpt(question):
        system_prompt = """저는 초등학생들을 가르치는 교사입니다. 
            오늘은 동물친구들 알아보기 시간을 가질거에요, 이름하여 "산나는 동물 친구들 시간" 이에요 
            개코원숭이, 뱀, 햄스터, 사슴, 도마뱀, 참새, 물고기, 오징어 이미지를 보고 설명하는 시간을 가질거에요
            동물에게 인사를 해볼까요 라고 하면 각 동물의 특징을 아이 눈높이에 맞게 적절히 설명해주세요 
            ==============================================================================================
            1. 동물의 특징을 초등학생 눈높이에 맞춰 쉽고 재미있게 설명해주세요.
            2. 동물이 직접 자기소개 하는 것처럼, '안녕,나는 OO이야'로 시작해주세요.
            3. 설명에는 동물의 이름, 특징, 사는 곳, 좋아하는 것들을 포함해주세요.
            4. 친근하고 활기찬 말투(반말)로 이야기하고, 이모지를 사용하지 말아주세요.
            5. 답변은 한두 문장으로 짧고 간결하게 만들어주세요.
            ==============================================================================================
            #답변 예시
            안녕 난 얼룩말이야
            나는 멋진 얼룩 무늬를 가지고 있어 
            
            개코원숭이: 안녕! 난 개코원숭이야! 
            내 얼굴은 강아지처럼 길쭉해서 개코원숭이라고 불리지. 
            나는 무리 지어 다니는 걸 아주 좋아해!
            뱀: 안녕, 난 뱀이야! 
            난 다리가 없어서 배로 스르륵 기어 다니지. 
            날 무서워하지 마! 쉿! 조용히 할 수 있어.
    
            물고기: 안녕, 나는 물고기야! 
            나는 물속에 살고, 물 밖으로 나오면 숨을 못 쉬어. 
            뽀끔뽀끔! 내 입 봐봐.
            """
        # system_prompt, user
        # LLM 넣어야 해
        response = client.chat.completions.create(
        model = "gpt-4.1-nano",
                messages = [
                        {
                                "role" : "system",
                                "content" : system_prompt
                        },
                        {
                                "role" : "user",  # user는 강사가 됨 
                                "content" : question
                        }
                ],
                temperature = 1.0
        )

        answer = response.choices[0].message.content

        # answer = answer.replace('\n', '')  # 완전히 없애도 됩니다.
        answer = answer.replace('\\n', ' ').replace('\n', ' ')

        return answer

# ask_gpt("개코원숭이에요 개코원숭이에 대해서 설명해주세요")

# py는 모듈화해서 가져올 수 있지만 
# jupyter 파일은 모듈화할 수 없다.  