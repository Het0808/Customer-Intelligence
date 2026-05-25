"""
API tests for src/serving/serve.py.

Uses FastAPI's TestClient with a mocked ModelLoader so no real MLflow runs
or trained models are required.  The mock is injected by patching the
module-level _loader singleton in model_loader and replacing load_model()
with a no-op that installs the mock instead of loading from disk.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.model_loader as ml_module
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
def client(mock_loader: MagicMock) -> TestClient:
    """TestClient with load_model() patched to install the mock loader."""

    def _install_mock():
        ml_module._loader = mock_loader
        return mock_loader

    with patch("src.serving.serve.load_model", side_effect=_install_mock):
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
