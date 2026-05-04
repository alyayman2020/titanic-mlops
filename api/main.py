"""
api/main.py — FastAPI inference server for Titanic survival prediction.

Endpoints:
    GET  /health          → health check + model info
    GET  /model/info      → detailed model metadata
    POST /predict         → single OR batch prediction (same endpoint)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.model_loader import registry
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PassengerPrediction,
    PredictRequest,
    PredictResponse,
)

# ── Lifespan: load model on startup ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Titanic MLOps API — loading model…")
    try:
        registry.load()
        logger.info(f"Model ready: {registry.model_source} / {registry.model_version}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        # Don't crash the server — /health will report model_loaded=False
    yield
    logger.info("Shutting down API")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "MLOps Lab 4 — Serving the best registered model from DagsHub.\n\n"
        "Accepts single or batch passenger records and returns survival predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "PassengerId", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]

def _confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Medium"
    return "Low"

def _passengers_to_df(request: PredictRequest) -> pd.DataFrame:
    """Convert request passengers to a DataFrame matching training format."""
    rows = []
    for i, p in enumerate(request.passengers):
        rows.append({
            "PassengerId": i + 1,
            "Pclass":      p.Pclass,
            "Name":        p.Name,
            "Sex":         p.Sex,
            "Age":         p.Age,
            "SibSp":       p.SibSp,
            "Parch":       p.Parch,
            "Ticket":      p.Ticket,
            "Fare":        p.Fare,
            "Cabin":       p.Cabin,
            "Embarked":    p.Embarked,
        })
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Titanic Survival Prediction API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Health check — returns model status and source."""
    return HealthResponse(
        status        = "ok" if registry.model is not None else "degraded",
        model_loaded  = registry.model is not None,
        model_name    = registry.model_name,
        model_source  = registry.model_source,
        model_version = registry.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Detailed model metadata."""
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name    = registry.model_name,
        model_source  = registry.model_source,
        model_version = registry.model_version,
        model_type    = registry.model_type,
        features      = EXPECTED_COLUMNS,
        description   = (
            "Titanic survival classifier trained on 712 passengers. "
            "Supports single and batch prediction."
        ),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict survival for one or more passengers.

    - **Single request**: pass 1 passenger in the list
    - **Batch request**: pass N passengers in the list

    Returns survival prediction (0/1) and probability for each passenger.
    """
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    t0 = time.perf_counter()

    try:
        df = _passengers_to_df(request)
        preds, probas = registry.predict(df)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    passenger_results = [
        PassengerPrediction(
            passenger_index = i,
            survived        = int(preds[i]),
            probability     = round(float(probas[i]), 4),
            confidence      = _confidence(float(probas[i])),
        )
        for i in range(len(preds))
    ]

    survived_count = sum(p.survived for p in passenger_results)

    logger.info(
        f"Predicted {len(preds)} passengers | "
        f"{survived_count} survived | {elapsed_ms}ms"
    )

    return PredictResponse(
        predictions       = passenger_results,
        total_passengers  = len(preds),
        survived_count    = survived_count,
        model_name        = registry.model_name,
        model_source      = registry.model_source,
        processing_time_ms = elapsed_ms,
    )
