from typing import Union
from fastapi import FastAPI

import model

and_model = model.AndModel()
or_model = model.OrModel()
not_model = model.NotModel()
xor_model = model.XORModel()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# AND 연산 예측
@app.get("/predict/and/{left}/{right}") # endpoint 엔드포인트
def predict_and(left: int, right: int):
    result = and_model.predict([left, right])
    return {"result": result}

# # OR 연산 예측
@app.get("/predict/or/{left}/{right}")
def predict_or(left: int, right: int):
    result = or_model.predict([left, right])
    return {"result": result}

# NOT 연산 예측
@app.get("/predict/not/{value}")
def predict_not(value: int):
    result = not_model.predict([value])
    return {"result": result}

# XOR 연산 예측
@app.get("/predict/xor/{left}/{right}")
def predict_xor(left: int, right: int):
    result = xor_model.predict([left, right])
    return {"result": result}


# 모델의 학습을 요청한다. 생성 기능은 POST로 한다.
# @app.get("/train")
# def train():
#     model.train()
#     return {"result": "OK"}

# # 모델의 학습을 요청한다. 생성 기능은 POST로 한다.
# @app.post("/train")
# def train():
#     model.train()
#     return {"result": "OK"}

# AND 모델 학습
@app.post("/train/and")
def train_and():
    and_model.train()
    return {"operation": "AND", "result": "Training Completed"}

# XOR 모델 학습
@app.post("/train/xor")
def train_xor():
    xor_model.train()
    return {"operation": "XOR", "result": "Training Completed"}


