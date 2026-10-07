from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time

logger = logging.getLogger("uvicorn.error")

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    age: float = Field(gt=0)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(gt=0)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0)
    slope: Literal[1, 2, 3]

    # These fields must be supplied, but may contain JSON null.
    ca: Literal[0, 1, 2, 3] | None
    thal: Literal[3, 6, 7] | None


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: str
    confidence: float = Field(ge=0, le=1)
    disease_probability: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    yield
    del app.state.model


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps assignment API for the UCI Heart Disease model.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "method=%s path=%s status=500 duration_ms=%.2f",
            request.method,
            request.url.path,
            (time.perf_counter() - started) * 1000,
        )
        raise

    logger.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )

    return response

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = request.app.state.model

    # Preserve training feature order and convert null values to NaN.
    features = pd.DataFrame(
        [payload.model_dump()],
        columns=FEATURES,
    ).astype("float64")

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    positive_index = classes.index(1)
    predicted_index = classes.index(prediction)

    return PredictionResponse(
        prediction=prediction,
        label="Present" if prediction == 1 else "Absent",
        confidence=float(probabilities[predicted_index]),
        disease_probability=float(probabilities[positive_index]),
    )