import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
import noisereduce as nr
import numpy as np
from pydub import AudioSegment

# .env 파일에서 환경 변수 불러오기 (API 키)
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. WAV 파일의 노이즈를 제거하는 함수
def denoise_audio(input_file):
    """
    WAV 파일의 노이즈를 제거하고 임시 파일 경로를 반환합니다.
    """
    try:
        # pydub를 사용하여 WAV 파일 불러오기
        audio = AudioSegment.from_wav(input_file)
        
        # 오디오 데이터를 넘파이(numpy) 배열로 변환
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # noisereduce를 사용하여 노이즈 제거
        reduced_noise_data = nr.reduce_noise(y=samples, sr=sample_rate, stationary=True)
        reduced_audio = audio._spawn(reduced_noise_data.astype(np.int16).tobytes())
        
        # 노이즈가 제거된 오디오를 임시 파일로 저장
        temp_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        reduced_audio.export(temp_wav_path, format="wav")
        
        print(f"'{input_file}' 파일의 노이즈가 제거되어 임시 파일에 저장되었습니다.")
        return temp_wav_path
    
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"노이즈 제거 중 오류가 발생했습니다: {e}")
        return None

# 2. 오디오 파일을 텍스트로 변환하는 함수 (STT)
def transcribe_audio(file_path):
    """
    오디오 파일을 텍스트로 변환합니다. 변환 전 노이즈 제거가 자동으로 실행됩니다.
    """
    temp_file_path = None
    try:
        # 노이즈 제거 함수 호출
        temp_file_path = denoise_audio(file_path)
        if not temp_file_path:
            return ""

        # 임시 파일을 사용해 STT 변환 진행
        with open(temp_file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        return transcript.text.strip()
        
    finally:
        # 작업이 끝나면 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"임시 파일 '{temp_file_path}'이 삭제되었습니다.")

# 3. 텍스트를 AI에게 질문하고 답변을 받는 함수 (ChatGPT)
def ask_gpt(question):
    """
    텍스트 질문을 GPT 모델에 보내고, 초등학생 눈높이에 맞는 답변을 받습니다.
    """
    system_prompt = """
    저는 초등학생들을 가르치는 교사입니다. 
    오늘은 동물친구들 알아보기 시간을 가질거에요, 이름하여 "산나는 동물 친구들 시간" 이에요 
    개코원숭이, 뱀, 햄스터, 사슴, 도마뱀, 참새, 물고기, 오징어 이미지를 보고 설명하는 시간을 가질거에요
    동물에게 인사를 해볼까요 라고 하면 각 동물의 특징을 아이 눈높이에 맞게 적절히 설명해주세요 
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=1.0
    )
    answer = response.choices[0].message.content
    answer = answer.replace('\\n', ' ').replace('\n', ' ')
    return answer

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # ⭐️ 이 부분을 수정하면 됩니다. 
    # audio_path 변수에 처리할 오디오 파일의 경로를 적어주세요.
    audio_path = "개코-원숭이.wav"
    
    if os.path.exists(audio_path):
        print(f"'{audio_path}' 파일의 STT 변환을 시작합니다. 노이즈 제거를 먼저 진행합니다...")
        
        # 1. 노이즈 제거 후 STT 변환
        transcribed_text = transcribe_audio(audio_path)
        
        if transcribed_text:
            print(f"\n✅ STT 변환 완료된 텍스트: {transcribed_text}")
            
            # 2. 변환된 텍스트를 AI에게 전달하여 답변 받기
            print("\n🤖 GPT 답변을 생성하는 중...")
            gpt_response = ask_gpt(transcribed_text)
            
            print(f"\n🌟 GPT의 답변: {gpt_response}")
        else:
            print("STT 변환에 실패했습니다.")
    else:
        print(f"오류: '{audio_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")