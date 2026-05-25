"""
FastAPI application for the Customer Intelligence Platform.

Endpoints:
  GET  /health            -- liveness + model readiness probe
  POST /predict           -- single-row bank campaign conversion score
  POST /batch-score       -- multi-row scoring (bad rows flagged, not crashed)
  GET  /metrics           -- model metadata for monitoring dashboards
  POST /ask-complaints    -- CFPB complaint intelligence RAG query

Start locally:
    uvicorn src.serving.serve:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up ml-service

Note on /ask-complaints:
  Requires a built FAISS index (run: python -m src.rag.build_index).
  If the index is absent, the endpoint returns HTTP 503 rather than crashing
  the server -- the prediction endpoints remain fully operational.
"""
from __future__ import annotations

import logging
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
    ComplaintAnswer,
    ComplaintQuery,
    CustomerFeatures,
    PredictionResponse,
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ML prediction model -- required; crash fast if missing
    load_model()

    # RAG retriever -- optional; degrade gracefully if index not built yet
    try:
        from src.rag.retrieve import load_retriever
        load_retriever()
        log.info("RAG retriever loaded -- /ask-complaints is available")
    except FileNotFoundError:
        log.warning(
            "FAISS index not found -- /ask-complaints will return 503. "
            "Build it with:  python -m src.rag.build_index"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("RAG retriever failed to load (%s) -- /ask-complaints disabled", exc)

    yield


app = FastAPI(
    title       = "Customer Intelligence API",
    description = "Bank campaign scoring + CFPB complaint intelligence",
    version     = "2.0.0",
    lifespan    = lifespan,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _band(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def _rag_retriever():
    """Return the RAG retriever or raise HTTP 503 if unavailable."""
    try:
        from src.rag.retrieve import get_retriever
        return get_retriever()
    except RuntimeError:
        raise HTTPException(
            status_code = 503,
            detail      = "RAG index not available. "
                          "Run:  python -m src.rag.build_index",
        )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness + readiness probe.  Returns model_version for canary checks."""
    loader = get_loader()
    rag_ready = False
    try:
        from src.rag.retrieve import get_retriever
        rag_ready = get_retriever().is_ready
    except RuntimeError:
        pass
    return {
        "status":        "ok",
        "model_version": loader.model_version,
        "run_id":        loader.run_id,
        "is_ready":      loader.is_ready,
        "rag_ready":     rag_ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Score a single customer for bank campaign conversion."""
    loader = get_loader()
    t0     = time.perf_counter()

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
    """Score multiple customers.  A bad row is flagged, not crashed."""
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

    return BatchScoreResponse(results=results, total=len(results), errors=errors)


@app.get("/metrics")
def metrics() -> dict:
    """Model metadata for dashboards and alerting."""
    loader = get_loader()
    return {
        "model_version": loader.model_version,
        "run_id":        loader.run_id,
        "threshold":     loader.threshold,
    }


@app.post("/ask-complaints", response_model=ComplaintAnswer)
def ask_complaints(request: ComplaintQuery) -> ComplaintAnswer:
    """
    Answer a natural-language question using CFPB complaint evidence.

    Pre-filters the complaint corpus by product/company/issue/date before
    running cosine search.  Returns a structured refusal (refusal=True,
    retrieved_ids=[]) when no sufficiently relevant evidence is found --
    the LLM is NEVER called in that case.
    """
    _rag_retriever()   # raises 503 if index absent

    # Build filter dict from non-null request fields
    filters: dict | None = None
    raw_filters = {
        k: v for k, v in {
            "product":   request.product,
            "company":   request.company,
            "date_from": request.date_from,
            "issue":     request.issue,
        }.items() if v is not None
    }
    if raw_filters:
        filters = raw_filters

    from src.rag.answer import ask
    result = ask(request.question, filters=filters)

    return ComplaintAnswer(
        answer               = result.answer,
        retrieved_ids        = result.retrieved_ids,
        evidence_sufficiency = result.evidence_sufficiency,
        prompt_version       = result.prompt_version,
        refusal              = result.refusal,
    )
