"""
API tests for src/serving/serve.py.

Uses FastAPI's TestClient with mocked singletons:
  - ModelLoader  : patched so no MLflow run is needed
  - RAG Retriever: patched so no FAISS index is needed

The lifespan's load_model() and load_retriever() calls are both intercepted
by fixtures, installing mock objects that satisfy get_loader() / get_retriever()
without touching disk.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.model_loader as ml_module
import src.rag.retrieve as rag_module
from src.serving.serve import app

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 180,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.299,
    "nr.employed": 5099.1,
}


@pytest.fixture
def mock_loader() -> MagicMock:
    loader = MagicMock(spec=ml_module.ModelLoader)
    loader.is_ready = True
    loader.model_version = "XGBoost_improved"
    loader.run_id = "deadbeef0123456789abcdef01234567"
    loader.threshold = 0.5
    loader.predict.return_value = np.array([0.72])
    return loader


@pytest.fixture
def mock_retriever() -> MagicMock:
    from src.rag.retrieve import Retriever
    r = MagicMock(spec=Retriever)
    r.is_ready = True
    r.corpus_size = 500
    return r


@pytest.fixture
def mock_ask() -> MagicMock:
    """
    Mock the full ask() pipeline so no LLM call or vector search is needed
    in API-layer tests.  Unit tests for the pipeline logic live in test_rag.py.
    """
    from src.rag.answer import AnswerResult
    m = MagicMock()
    m.return_value = AnswerResult(
        answer="Based on [E1], customers commonly face payment processing delays.",
        retrieved_ids=["1000001"],
        evidence_sufficiency="sufficient",
        prompt_version="v1.0",
        refusal=False,
        model_name="gpt-4o-mini",
        latency_ms=234.5,
        token_count=87,
        generation_succeeded=True,
    )
    return m


@pytest.fixture
def client(mock_loader: MagicMock,
           mock_retriever: MagicMock,
           mock_ask: MagicMock) -> TestClient:
    """
    TestClient with the three singletons mocked:
      - load_model  patched on src.serving.serve  (top-level import there)
      - load_retriever patched on src.rag.retrieve (lifespan does lazy import)
      - ask()  patched on src.rag.answer         (route does lazy import)
    """
    def _install_ml():
        ml_module._loader = mock_loader

    def _install_rag(index_dir=None):
        rag_module._retriever = mock_retriever
        return mock_retriever

    with patch("src.serving.serve.load_model",   side_effect=_install_ml), \
         patch("src.rag.retrieve.load_retriever", side_effect=_install_rag), \
         patch("src.rag.answer.ask",              mock_ask):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_contains_model_version(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "model_version" in data
        assert data["model_version"] == "XGBoost_improved"

    def test_is_ready_true(self, client: TestClient) -> None:
        assert client.get("/health").json()["is_ready"] is True

    def test_run_id_present(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "run_id" in data and len(data["run_id"]) > 0


# ---------------------------------------------------------------------------
# /predict -- happy path
# ---------------------------------------------------------------------------
class TestPredict:
    def test_valid_request_returns_200(self, client: TestClient) -> None:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_fields_present(self, client: TestClient) -> None:
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        for field in ("probability", "threshold_decision", "model_version",
                      "run_id", "latency_ms", "request_id"):
            assert field in data, f"Missing field: {field}"

    def test_model_version_in_response(self, client: TestClient) -> None:
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["model_version"] == "XGBoost_improved"

    def test_high_probability_maps_to_high_band(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.75])
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["threshold_decision"] == "high"

    def test_medium_probability_maps_to_medium_band(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.45])
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["threshold_decision"] == "medium"

    def test_low_probability_maps_to_low_band(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.15])
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["threshold_decision"] == "low"

    def test_probability_boundary_0_6_is_high(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.6])
        assert client.post("/predict", json=VALID_PAYLOAD).json()["threshold_decision"] == "high"

    def test_probability_boundary_0_3_is_medium(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.3])
        assert client.post("/predict", json=VALID_PAYLOAD).json()["threshold_decision"] == "medium"

    def test_probability_just_below_0_3_is_low(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.return_value = np.array([0.2999])
        assert client.post("/predict", json=VALID_PAYLOAD).json()["threshold_decision"] == "low"


# ---------------------------------------------------------------------------
# /predict -- validation errors
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_age_below_minimum_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "age": 5}
        assert client.post("/predict", json=payload).status_code == 422

    def test_age_above_maximum_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "age": 150}
        assert client.post("/predict", json=payload).status_code == 422

    def test_campaign_zero_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "campaign": 0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_duration_negative_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "duration": -1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_default_value_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "default": "maybe"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_emp_var_rate_out_of_range_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "emp.var.rate": 99.0}
        assert client.post("/predict", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# /batch-score
# ---------------------------------------------------------------------------
class TestBatchScore:
    def test_valid_batch_returns_200(self, client: TestClient) -> None:
        resp = client.post("/batch-score", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
        assert resp.status_code == 200

    def test_response_total_matches_input(self, client: TestClient) -> None:
        resp = client.post("/batch-score", json={"records": [VALID_PAYLOAD] * 3})
        data = resp.json()
        assert data["total"] == 3

    def test_model_version_in_every_result(self, client: TestClient) -> None:
        resp = client.post("/batch-score", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
        for row in resp.json()["results"]:
            assert row["model_version"] == "XGBoost_improved"

    def test_prediction_error_row_has_error_field(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        """A RuntimeError on one row must not abort the batch; it gets error set."""
        call_count = 0

        def _side_effect(df):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("simulated pipeline failure")
            return np.array([0.5])

        mock_loader.predict.side_effect = _side_effect
        resp = client.post("/batch-score", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
        data = resp.json()
        assert data["total"] == 2
        assert data["errors"] == 1
        error_rows = [r for r in data["results"] if r.get("error") is not None]
        assert len(error_rows) == 1
        assert "simulated" in error_rows[0]["error"]

    def test_error_row_still_carries_model_version(
        self, client: TestClient, mock_loader: MagicMock
    ) -> None:
        mock_loader.predict.side_effect = RuntimeError("boom")
        resp = client.post("/batch-score", json={"records": [VALID_PAYLOAD]})
        row = resp.json()["results"][0]
        assert row["model_version"] == "XGBoost_improved"
        assert row["error"] is not None


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200

    def test_contains_model_version(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert data["model_version"] == "XGBoost_improved"

    def test_contains_threshold(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)


# ---------------------------------------------------------------------------
# /ask-complaints
# ---------------------------------------------------------------------------
COMPLAINT_PAYLOAD = {"question": "What problems do customers have with mortgages?"}

COMPLAINT_PAYLOAD_FILTERED = {
    "question": "What payment issues do mortgage customers face?",
    "product":  "Mortgage",
    "issue":    "Trouble during payment process",
}


class TestAskComplaints:
    def test_valid_question_returns_200(self, client: TestClient) -> None:
        assert client.post("/ask-complaints", json=COMPLAINT_PAYLOAD).status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        data = client.post("/ask-complaints", json=COMPLAINT_PAYLOAD).json()
        for f in ("answer", "retrieved_ids", "evidence_sufficiency",
                  "prompt_version", "refusal"):
            assert f in data, f"Missing field: {f}"

    def test_sufficient_evidence_response_structure(
        self, client: TestClient
    ) -> None:
        data = client.post("/ask-complaints", json=COMPLAINT_PAYLOAD).json()
        assert data["evidence_sufficiency"] == "sufficient"
        assert data["refusal"] is False
        assert len(data["retrieved_ids"]) > 0
        assert data["answer"] is not None

    def test_filtered_query_passes_filters_to_ask(
        self, client: TestClient, mock_ask: MagicMock
    ) -> None:
        """Filters from the request must be forwarded to ask() as a dict."""
        client.post("/ask-complaints", json=COMPLAINT_PAYLOAD_FILTERED)
        _, kwargs = mock_ask.call_args
        filters = kwargs.get("filters") or {}
        assert filters.get("product") == "Mortgage"
        assert filters.get("issue") == "Trouble during payment process"

    def test_null_filter_fields_omitted_from_ask(
        self, client: TestClient, mock_ask: MagicMock
    ) -> None:
        """None-valued filter fields must not appear in the filters dict."""
        client.post("/ask-complaints", json={"question": "test", "product": None})
        _, kwargs = mock_ask.call_args
        filters = kwargs.get("filters")
        # None fields should produce filters=None or empty dict, not {"product": None}
        if filters:
            assert "product" not in filters or filters["product"] is not None

    def test_insufficient_evidence_returns_refusal(
        self, client: TestClient, mock_ask: MagicMock
    ) -> None:
        from src.rag.answer import AnswerResult
        mock_ask.return_value = AnswerResult(
            answer=None, retrieved_ids=[], evidence_sufficiency="insufficient",
            prompt_version="v1.0", refusal=True, model_name="gpt-4o-mini",
            latency_ms=5.2, token_count=0, generation_succeeded=False,
        )
        data = client.post("/ask-complaints", json=COMPLAINT_PAYLOAD).json()
        assert data["refusal"] is True
        assert data["retrieved_ids"] == []
        assert data["evidence_sufficiency"] == "insufficient"
        assert data["answer"] is None

    def test_refusal_answer_is_null_and_prompt_version_present(
        self, client: TestClient, mock_ask: MagicMock
    ) -> None:
        from src.rag.answer import AnswerResult
        mock_ask.return_value = AnswerResult(
            answer=None, retrieved_ids=[], evidence_sufficiency="insufficient",
            prompt_version="v1.0", refusal=True, model_name="gpt-4o-mini",
            latency_ms=5.2, token_count=0, generation_succeeded=False,
        )
        data = client.post("/ask-complaints", json=COMPLAINT_PAYLOAD).json()
        assert data["answer"] is None
        assert data["prompt_version"] == "v1.0"

    def test_missing_question_returns_422(self, client: TestClient) -> None:
        assert client.post("/ask-complaints", json={}).status_code == 422

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        assert client.post("/ask-complaints",
                           json={"question": ""}).status_code == 422

    def test_rag_unavailable_returns_503(self, mock_loader: MagicMock) -> None:
        """When FAISS index absent the lifespan logs a warning and the endpoint
        returns 503 -- the prediction endpoints must still work."""
        def _install_ml():
            ml_module._loader = mock_loader

        rag_module._retriever = None

        # Simulate absent index regardless of whether one is on disk.
        with patch("src.serving.serve.load_model", side_effect=_install_ml), \
             patch("src.rag.retrieve.load_retriever",
                   side_effect=FileNotFoundError("no index (simulated)")):
            with TestClient(app) as c:
                resp = c.post("/ask-complaints", json=COMPLAINT_PAYLOAD)
        assert resp.status_code == 503

    def test_health_reports_rag_ready(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "rag_ready" in data
        assert data["rag_ready"] is True
