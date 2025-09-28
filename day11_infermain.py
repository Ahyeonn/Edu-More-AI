from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel # 데이터 예외처리시 사용
from typing import List
from PIL import Image # 모델 사용시 사용
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
'''
서버는 저장된 모델한테 이미지를 받음 -> 이미지를 문자열로 바꿔서 보냄 -> 이 문자열을 다시 이미지로 변환 
-> transform(전처리기)에서 tensor or numpy의 array로 바꿔야함 ->모델한테 넣기 -> 결과를 사용자한테 전달

데이터 전처리 중요
실전에서는 데이터 수집 중요
ㄴ why? => 재학습시킬 수 있고, 돈이 되기 때문
pydantic => 데이터 검증 라이브러리
'''
# <전체 흐름>

# 1. Model 로드

# 2. 클라이언트에게 받은 이미지 transforms(tensor/numpy)으로 변환 - model에 맞게 전처리

# *** mlops/aiops의 경우 사용자 요청 이미지를 따로 저장(S3/DISK)하여 학습용으로 활용 
# *** 1주일/1달 단위 활용하는 것으로 포트폴리오에서 설계 어필!

# 3. 전처리 결과물 Model로 추론

# 4. 추론 결과를 클라이언트에게 전달

# 데이터 수집이 가장 중요. 나중에 재학습시킬 용도.

app = FastAPI(title="ResNet34 Inference")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 불러와 사용
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=3, bias=True) # 학습 할 때 파라미터로 통일
# 모델에 저장된 가중치 불러오기
model.load_state_dict(torch.load('./models/mymodel.pth'))
model.eval()
model.to(device)

# 학습할 때와 같은 값 변환: 학습용이 아님으로 Test값을 가져온다 (똑같이 불러온 모델의 전처리 해줘야한다)
transforms_infer = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)

# Pydantic은 유효성 검증을 자동으로 해준다: 상대방에게 전달할시 데이터 타입 정의
class predict_response(BaseModel):
    name : str
    score : float
    type : int

# 엔드포인트 정의 
@app.post("/predict", response_model=predict_response)
async def predict(file: UploadFile=File(...)): # UploadFile=File(...)은 fastapi가 정해놓은 형식이다
    image = Image.open(file.file) # 이미지 불러옴
    # uuid, index, timestamp
    image.save("imgdata/test.jpg") # 실제로는 test를 uuid 카운트, timestamp로 저장하여 여러 이미지 이름들을 unique화
    img_tensor = transforms_infer(image).unsqueeze(0).to(device) # [3, 224, 224] => [1, 3, 224, 224] 이미지 전처리

    with torch.no_grad(): # 기울기 값 계산 안하기
        pred = model(img_tensor)
        print('예측값 :', pred)

    pred_result = torch.max(pred,dim=1)[1].item() # 0, 1, 2 최댓값 인덱스 추출
    # 세개의 정답값이 나왔을때 값을 1로 기준을 잡아서 정답값을 좁혀준다 (활성화 함수인 ReLU 처럼)
    score = nn.Softmax()(pred)[0] # 전체를 0 ~ 1로 봤을 때 결과값을 그 사이 값으로 변환해줌  (0 idx: 예측 값, 1 idx: 모델이 정답이라 예측한 클래스 인덱스 값)
    print("Softmax :", score)
    classname = ['박재범', '아이유', '카리나']
    name = classname[pred_result]
    print('name :', name)

    return predict_response(name=name, score=float(score[pred_result]), type=pred_result)
