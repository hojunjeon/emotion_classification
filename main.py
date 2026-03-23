from fastapi import FastAPI
import torch
from transformers import pipeline
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1. gpu 세팅
device = 0 if torch.cuda.is_available() else -1

if device :
    print("GPU 세팅 성공!")

# 2. model load
pipe = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

print("모델 로드 완료")

@app.get("/")
def home():
    return {"메세지" : "서버 정상적으로 실행 중!"}

@app.post("/predict")
def predict(data : dict):
    # 1. 사용자로부터 문장을 전달받음
    text = data.get("text","")

    # 2. 문장의 감정을 분석
    raw_result = pipe(text)
    print(f"raw_Result = {raw_result}")
    # 3. 분석 결과를 return

    return {
        "메세지" : text,
        "점수" : float(raw_result[0]["score"]),
        "감정" : raw_result[0]["label"]
    }
