from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model

app = FastAPI()

model = app_model.AppModel()

@app.get("/say")
def say_app(text: str = Query()):
    response = model.get_response(text)
    return {"content": response.content}

# @app.get("/translate")
# def translate(text: str = Query(), language: str = Query()):
#     response = model.get_prompt_response(language, text)
#     return {"content": response.content}