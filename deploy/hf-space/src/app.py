"""FastAPI service: POST /predict -> prediction + confidence, with logging + metrics."""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from predict import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart_api")

REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds", "Request latency in seconds", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("api_predictions_total", "Predictions made", ["label"])
CONFIDENCE = Histogram(
    "api_prediction_confidence", "Confidence of predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
QUIET_PATHS = {"/health", "/metrics"}      # not logged: probes would flood the logs

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()      # load once at startup
    logger.info(json.dumps({"event": "startup", "model_loaded": True}))
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts presence of heart disease from patient health data.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:8])
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception(json.dumps({"event": "unhandled_error", "request_id": request_id}))
        raise
    finally:
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"      # template, keeps labels low-cardinality
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(duration)
        if path not in QUIET_PATHS:
            logger.info(json.dumps({
                "event": "request", "request_id": request_id,
                "method": request.method, "path": request.url.path,
                "status": status, "duration_ms": round(duration * 1000, 2),
            }))


class Patient(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels coloured (0-4)")
    thal: Literal[3, 6, 7] = Field(..., description="3 normal, 6 fixed defect, 7 reversible")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    result = predict([patient.model_dump()], model=state["model"])[0]
    PREDICTIONS.labels(label=result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    logger.info(json.dumps({
        "event": "prediction", "prediction": result["prediction"],
        "confidence": result["confidence"],
    }))
    return result
