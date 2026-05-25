"""
FastAPI application for the Customer Intelligence Platform.

Endpoints:
  GET  /health         -- liveness + model readiness probe
  POST /predict        -- single-row prediction
  POST /batch-score    -- multi-row prediction (bad rows flagged, not crashed)
  GET  /metrics        -- lightweight model metadata for monitoring dashboards

Start locally:
    uvicorn src.serving.serve:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up ml-service
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.model_loader import get_loader, load_model
from src.serving.schemas import (
    BatchRecord,
    BatchScoreRequest,
    BatchScoreResponse,
    CustomerFeatures,
    PredictionResponse,
)


# -----------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # No teardown needed -- GC handles the loaded pipeline


app = FastAPI(
    title="Customer Intelligence API",
    description="Bank campaign conversion scoring (UCI Bank Marketing dataset)",
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Threshold band helper
# -----------------------------------------------------------------------------
def _band(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness + readiness probe.  Returns model_version so callers can verify
    which model artefact is loaded without querying MLflow directly."""
    loader = get_loader()
    return {
        "status":        "ok",
        "model_version": loader.model_version,
        "run_id":        loader.run_id,
        "is_ready":      loader.is_ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Score a single customer.  Returns probability and threshold band."""
    loader = get_loader()
    t0 = time.perf_counter()

    df   = pd.DataFrame([features.to_raw_dict()])
    prob = float(loader.predict(df)[0])

    return PredictionResponse(
        probability        = round(prob, 6),
        threshold_decision = _band(prob),
        model_version      = loader.model_version,
        run_id             = loader.run_id,
        latency_ms         = round((time.perf_counter() - t0) * 1_000, 2),
        request_id         = str(uuid.uuid4()),
    )


@app.post("/batch-score", response_model=BatchScoreResponse)
def batch_score(request: BatchScoreRequest) -> BatchScoreResponse:
    """
    Score a list of customers.  A single bad row never aborts the batch --
    it is returned with error set and probability=None.
    """
    loader  = get_loader()
    results: list[BatchRecord] = []
    errors  = 0

    for features in request.records:
        req_id = str(uuid.uuid4())
        try:
            t0   = time.perf_counter()
            df   = pd.DataFrame([features.to_raw_dict()])
            prob = float(loader.predict(df)[0])
            results.append(BatchRecord(
                probability        = round(prob, 6),
                threshold_decision = _band(prob),
                model_version      = loader.model_version,
                run_id             = loader.run_id,
                latency_ms         = round((time.perf_counter() - t0) * 1_000, 2),
                request_id         = req_id,
            ))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            results.append(BatchRecord(
                model_version = loader.model_version,
                run_id        = loader.run_id,
                request_id    = req_id,
                error         = str(exc),
            ))

    return BatchScoreResponse(
        results = results,
        total   = len(results),
        errors  = errors,
    )


@app.get("/metrics")
def metrics() -> dict:
    """Lightweight model metadata for monitoring dashboards / alerting."""
    loader = get_loader()
    return {
        "model_version": loader.model_version,
        "run_id":        loader.run_id,
        "threshold":     loader.threshold,
    }
