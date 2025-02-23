from fastapi import FastAPI
from modelp import LogicModel, XORModel
import numpy as np

app = FastAPI()

# ✅ 입력 데이터 & 정답 레이블 정의
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs = np.array([0, 0, 0, 1])
or_outputs = np.array([0, 1, 1, 1])
not_inputs = np.array([[0], [1]])
not_outputs = np.array([1, 0])

# ✅ 모든 모델 초기화 및 로드
and_model = LogicModel("and_model", inputs, and_outputs)
or_model = LogicModel("or_model", inputs, or_outputs)
not_model = LogicModel("not_model", not_inputs, not_outputs)
xor_model = XORModel()

and_model.load_model()
or_model.load_model()
not_model.load_model()
xor_model.load_model()

# ✅ 모델 학습 엔드포인트 (POST 요청)
@app.post("/train/{model_type}")
def train_model(model_type: str):
    if model_type == "and":
        and_model.train()
    elif model_type == "or":
        or_model.train()
    elif model_type == "not":
        not_model.train()
    elif model_type == "xor":
        xor_model.train_model()
    else:
        return {"error": "Invalid model type"}
    return {"operation": model_type.upper(), "result": "Training Completed"}

# ✅ 모델 예측 엔드포인트 (GET 요청)
@app.get("/predict/{model_type}/{left}/{right}")
def predict(model_type: str, left: int, right: int):
    if model_type == "and":
        result = and_model.predict([left, right])
    elif model_type == "or":
        result = or_model.predict([left, right])
    elif model_type == "xor":
        result = xor_model.predict([left, right])
    else:
        return {"error": "Invalid model type"}
    return {"operation": model_type.upper(), "left": left, "right": right, "result": result}

@app.get("/predict/not/{value}")
def predict_not(value: int):
    result = not_model.predict([value])
    return {"operation": "NOT", "value": value, "result": result}