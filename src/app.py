import os
import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

app = FastAPI()


class PredictionInput(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.on_event("startup")
def load_model():
    app.state.model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(data: PredictionInput):
    features = [[
        data.age,
        data.sex,
        data.bmi,
        data.bp,
        data.s1,
        data.s2,
        data.s3,
        data.s4,
        data.s5,
        data.s6
    ]]

    prediction = app.state.model.predict(features)[0]

    return {"prediction": float(prediction)}