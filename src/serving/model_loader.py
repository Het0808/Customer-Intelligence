"""
MLflow model loader for the Customer Intelligence API.

Finds the most recent PROMOTED run that has a 'full_pipeline' artifact
(preprocessing + model in one sklearn Pipeline), loads it once at startup,
and exposes it via a module-level singleton so all request handlers share
the same in-memory object without re-loading from disk per request.

Requires training to have been run with:
    python -m src.training.train           # saves full_pipeline to MLflow
The full_pipeline artifact was added in the Phase 3 update to train.py.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# MLflow client helpers
# -----------------------------------------------------------------------------
def _find_best_promoted_run():
    """
    Return the most recent MLflow run tagged gate_decision='PROMOTED' that
    also has a 'full_pipeline' artifact.  Raises RuntimeError if none exists.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found at "
            f"{MLFLOW_TRACKING_URI}. Run 'python -m src.training.train' first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.gate_decision = 'PROMOTED'",
        order_by=["start_time DESC"],
    )

    for run in runs:
        top_level = [a.path for a in client.list_artifacts(run.info.run_id)]
        if "full_pipeline" in top_level:
            log.info(
                "Found PROMOTED run with full_pipeline: run_id=%s  name=%s",
                run.info.run_id,
                run.data.tags.get("mlflow.runName", ""),
            )
            return run

    raise RuntimeError(
        "No PROMOTED run with a 'full_pipeline' artifact found.\n"
        "Re-run training:  python -m src.training.train\n"
        "(The full_pipeline artifact was added in the Phase 3 update.)"
    )


# -----------------------------------------------------------------------------
# ModelLoader
# -----------------------------------------------------------------------------
class ModelLoader:
    """Wraps the loaded full sklearn Pipeline and exposes serving metadata."""

    def __init__(self) -> None:
        self._pipeline = None
        self._run_id: str | None = None
        self._model_version: str | None = None
        self._threshold: float = 0.5
        self._is_ready: bool = False

    def load(self) -> None:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        run = _find_best_promoted_run()
        self._run_id = run.info.run_id

        artifact_uri = f"{run.info.artifact_uri}/full_pipeline"
        log.info("Loading full_pipeline from %s", artifact_uri)
        self._pipeline = mlflow.sklearn.load_model(artifact_uri)

        self._threshold = float(run.data.metrics.get("val_threshold", 0.5))
        self._model_version = run.data.tags.get(
            "mlflow.runName", self._run_id[:8]
        )
        self._is_ready = True
        log.info(
            "Model ready -- version=%s  run_id=%s  threshold=%.4f",
            self._model_version, self._run_id, self._threshold,
        )

    # -- Properties ------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self._run_id or ""

    @property
    def model_version(self) -> str:
        return self._model_version or "unknown"

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # -- Inference -------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(subscribe=1) for each row in *df* (raw feature DataFrame)."""
        if not self._is_ready:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._pipeline.predict_proba(df)[:, 1]


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------
_loader: ModelLoader | None = None


def load_model() -> ModelLoader:
    """Create and load the singleton ModelLoader.  Called once at API startup."""
    global _loader
    _loader = ModelLoader()
    _loader.load()
    return _loader


def get_loader() -> ModelLoader:
    """Return the loaded singleton.  Raises if load_model() was never called."""
    if _loader is None:
        raise RuntimeError(
            "ModelLoader not initialised. load_model() must be called at startup."
        )
    return _loader
