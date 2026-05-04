"""Sample real-time inference API — FastAPI, no MLflow required.

Models are loaded from a .joblib file at startup. Any sklearn-compatible
model saved by LocalFileTracker (or joblib.dump directly) works out of the box.

Customise for your use case:
  1. Set MODEL_PATH via environment variable or edit the default below.
  2. Replace GenericRequest with a typed schema for your feature set.
  3. Add feature engineering before the model.predict() call if needed.

Run locally:
  uvicorn templates.realtime_api.app:app --reload --port 8000

Health check:
  curl http://localhost:8000/health

Predict (generic dict of features):
  curl -X POST http://localhost:8000/v1/predict \
       -H "Content-Type: application/json" \
       -d '{"features": {"tenure_months": 12, "monthly_charges": 45.0, "num_products": 2, "support_calls_90d": 1}}'

Requires: pip install "fastapi[standard]"
"""
import os
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config — set via environment variable or edit defaults
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/runs/latest/model.joblib")
MODEL_NAME = os.environ.get("MODEL_NAME", "ml-model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0")

# ---------------------------------------------------------------------------
# Request / response schemas
#
# GenericRequest accepts any feature dict — no schema changes needed as features
# change. For production use cases, replace with a typed schema for validation:
#
#   class ChurnRequest(BaseModel):
#       tenure_months: float
#       monthly_charges: float
#       num_products: int
#       support_calls_90d: int
#
# Then update the predict() function to build features_df from the typed fields.
# ---------------------------------------------------------------------------

class GenericRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    model_version: str
    scored_at: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Inference API",
    description=(
        "Generic model serving endpoint. Works with any sklearn-compatible model "
        "saved as a .joblib file. Replace GenericRequest with a typed schema for production."
    ),
    version="v1",
)

_model = None


@app.on_event("startup")
def load_model():
    global _model
    import pathlib
    path = pathlib.Path(MODEL_PATH)
    if not path.exists():
        print(f"WARNING: Model not found at {MODEL_PATH}. /predict will return 503.")
        return
    _model = joblib.load(path)
    print(f"Model loaded: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: GenericRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Set MODEL_PATH env var (current: {MODEL_PATH})",
        )

    try:
        # Build a single-row DataFrame from the feature dict.
        # Feature names must match exactly what the model was trained on.
        features_df = pd.DataFrame([request.features])
        prediction = int(_model.predict(features_df)[0])
        probability = float(_model.predict_proba(features_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
