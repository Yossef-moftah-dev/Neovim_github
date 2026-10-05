from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import SentimentModel

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = SentimentModel()  # load once at startup
    yield
    state.clear()


app = FastAPI(title="Arabic Sentiment API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["المنتج رائع جدا وأنصح به"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(422, "text must not be blank")
    return state["model"].predict([req.text])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    return state["model"].predict(req.texts)
