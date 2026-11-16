from fastapi import FastAPI
from language_control import language_

app = FastAPI()

@app.get("/recog")
def run_speech_reg():
    language_()

