"""Template: Real-time inference API (FastAPI).

Copy this for use cases requiring synchronous prediction responses.
Replace all TODO items with your implementation.

Before using this template, confirm real-time is the right choice.
See docs/decision-frameworks.md §2 (Batch vs Real-time).
Consider the pre-compute + cache pattern first — it often satisfies
"real-time" requirements with batch infrastructure simplicity.

Run locally:
  uvicorn templates.realtime_api.app:app --reload

Health check:
  curl http://localhost:8000/health

Predict:
  curl -X POST http://localhost:8000/v1/predict \
       -H "Content-Type: application/json" \
       -d '{"feature_1": 12, "feature_2": "A"}'
"""
from datetime import datetime, timezone

import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# TODO: Define request and response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    # TODO: replace with your actual input features
    feature_1: float
    feature_2: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    scored_at: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="ML Inference API", version="v1")

# TODO: set your model name and load strategy
MODEL_NAME = "your-model-name"
MODEL_VERSION = "1.0"
_model = None  # loaded lazily on first request


def get_model():
    global _model
    if _model is None:
        # TODO: replace with your model URI or registry lookup
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    return _model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()

    # TODO: build feature vector from request fields
    # Must match the feature engineering used during training exactly
    import pandas as pd
    features = pd.DataFrame([{
        "feature_1": request.feature_1,
        "feature_2_A": int(request.feature_2 == "A"),  # example encoding
    }])

    try:
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=f"{MODEL_NAME}:{MODEL_VERSION}",
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
