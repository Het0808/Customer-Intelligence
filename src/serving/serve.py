"""
FastAPI application for the Customer Intelligence Platform.

Endpoints:
  GET  /health            -- liveness + model readiness probe
  POST /predict           -- single-row bank campaign conversion score
  POST /batch-score       -- multi-row scoring (bad rows flagged, not crashed)
  GET  /metrics           -- model metadata for monitoring dashboards
  POST /ask-complaints    -- CFPB complaint intelligence RAG query
  POST /customer-intel    -- combined ML conversion score + RAG complaint themes

Start locally:
    uvicorn src.serving.serve:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up ml-service

Note on /ask-complaints and /customer-intel:
  Require a built FAISS index (run: python -m src.rag.build_index).
  If the index is absent, /ask-complaints returns HTTP 503; /customer-intel
  returns a valid response with complaint_themes=[] (graceful degradation).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.model_loader import get_loader, load_model
from src.serving.schemas import (
    BatchRecord,
    BatchScoreRequest,
    BatchScoreResponse,
    ComplaintAnswer,
    ComplaintQuery,
    ComplaintTheme,
    CustomerFeatures,
    CustomerIntelRequest,
    CustomerIntelResponse,
    PredictionResponse,
)

log = logging.getLogger(__name__)

# Paths used by /customer-intel helpers
_LOG_DIR   = Path(__file__).resolve().parents[2] / "logs"
_INDEX_BIN = Path(__file__).resolve().parents[2] / "faiss_index" / "index.bin"


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


def _get_index_version() -> str:
    """Return the FAISS index build timestamp as YYYYMMDDTHHMMSSZ, or 'unknown'."""
    try:
        if _INDEX_BIN.exists():
            mtime = _INDEX_BIN.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        pass
    return "unknown"


def _build_rag_question(features: CustomerFeatures, product: str | None) -> str:
    """
    Construct a natural-language CFPB complaint search query from customer features.

    Surfaces the most discriminative demographic/financial signals so the
    embedding model retrieves complaints from similar customer profiles.
    """
    parts = [f"customer aged {features.age} working as {features.job}"]
    if features.housing == "yes":
        parts.append("with a housing loan")
    if features.loan == "yes":
        parts.append("with a personal loan")
    subject = " ".join(parts)
    if product:
        return f"Complaints about {product} from {subject}"
    return f"Financial service complaints from {subject}"


def _build_complaint_themes(chunks: list) -> list[ComplaintTheme]:
    """
    Cluster retrieved chunks into themes using the CFPB issue taxonomy.

    Groups chunks by chunk.metadata['issue'] (the CFPB-assigned issue label).
    Within each group the highest-scoring chunk provides the representative text.
    Themes come directly from the retrieved corpus -- the LLM is never involved.

    Fallback: when all chunks share the same issue (or issue is empty), groups
    are formed by the most frequent non-trivial keyword in each chunk's text.
    """
    groups: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        issue = (chunk.metadata.get("issue") or "").strip() or "Other"
        groups[issue].append(chunk)

    # If every chunk landed in a single bucket, fall back to keyword clustering
    if len(groups) == 1:
        keyword_groups: dict[str, list] = defaultdict(list)
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
                "that", "for", "on", "with", "it", "as", "at", "be", "by"}
        for chunk in chunks:
            words = [
                w.lower().strip(".,!?;:\"'")
                for w in chunk.chunk_text.split()
                if len(w) > 3 and w.lower() not in stop
            ]
            top_word = max(set(words), key=words.count) if words else "Other"
            keyword_groups[top_word].append(chunk)
        groups = keyword_groups

    themes: list[ComplaintTheme] = []
    for label, group_chunks in groups.items():
        top           = max(group_chunks, key=lambda c: c.score)
        evidence_ids  = list({c.complaint_id for c in group_chunks})
        themes.append(ComplaintTheme(
            theme                = label,
            evidence_ids         = evidence_ids,
            representative_chunk = top.chunk_text[:300],
        ))
    return themes


def _log_customer_intel(
    request_id: str,
    request:    dict,
    response:   dict,
    latency_ms: float,
) -> None:
    """Append one JSONL audit record to logs/customer_intel.jsonl."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "request":    request,
            "response":   response,
            "latency_ms": latency_ms,
        }
        with (_LOG_DIR / "customer_intel.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)


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


@app.post("/customer-intel", response_model=CustomerIntelResponse)
def customer_intel(request: CustomerIntelRequest) -> CustomerIntelResponse:
    """
    Combined ML + RAG customer intelligence endpoint.

    (a) Scores the customer for bank campaign conversion using the trained
        XGBoost pipeline (always present in the response).

    (b) Retrieves the top CFPB complaint themes for this customer profile
        via FAISS cosine search, pre-filtered by product / issue / date_from.
        Themes are clustered directly from retrieved chunks using the CFPB
        issue taxonomy -- the LLM is NOT involved in theme generation.

    RAG is best-effort: if the index is absent or retrieval fails, the response
    still contains a valid conversion score with complaint_themes=[].

    Every call is appended to logs/customer_intel.jsonl for audit purposes.
    """
    t0     = time.perf_counter()
    loader = get_loader()
    req_id = str(uuid.uuid4())

    # ── ML: bank campaign conversion score ────────────────────────────────────
    df   = pd.DataFrame([request.customer_features.to_raw_dict()])
    prob = float(loader.predict(df)[0])
    band = _band(prob)

    # ── RAG: complaint themes (graceful degradation) ──────────────────────────
    complaint_themes: list[ComplaintTheme] = []
    index_version = _get_index_version()

    try:
        retriever = _rag_retriever()   # raises HTTPException(503) if index absent
        rag_filters = {k: v for k, v in {
            "product":   request.product,
            "issue":     request.issue,
            "date_from": request.date_from,
        }.items() if v is not None} or None

        question   = _build_rag_question(request.customer_features, request.product)
        rag_result = retriever.retrieve(question, filters=rag_filters, top_k=10)

        if not rag_result.insufficient_evidence:
            complaint_themes = _build_complaint_themes(rag_result.chunks)

    except HTTPException:
        pass   # RAG index absent -- complaint_themes stays []
    except Exception as exc:
        log.warning("RAG retrieval error (%s) -- complaint_themes will be empty", exc)

    latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

    response = CustomerIntelResponse(
        conversion_band        = band,
        conversion_probability = round(prob, 6),
        model_version          = loader.model_version,
        complaint_themes       = complaint_themes,
        index_version          = index_version,
        latency_ms             = latency_ms,
    )

    # ── Audit log ─────────────────────────────────────────────────────────────
    _log_customer_intel(
        request_id = req_id,
        request    = request.model_dump(),
        response   = response.model_dump(),
        latency_ms = latency_ms,
    )

    return response
