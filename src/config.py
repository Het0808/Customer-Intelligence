"""
Central config — all values read from .env / environment variables.
Import from here instead of calling os.getenv() directly in each module.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
# _ROOT is the project root regardless of where the script is invoked from.
_ROOT = Path(__file__).resolve().parents[1]
_BASE = Path(os.getenv("DATA_DIR", _ROOT / "data"))

DATA_DIR      = _BASE
RAW_DIR       = Path(os.getenv("RAW_DATA_DIR",       _BASE / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", _BASE / "processed"))
SAMPLES_DIR   = Path(os.getenv("SAMPLES_DATA_DIR",   _BASE / "samples"))

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",    f"file:{_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-intelligence")

# ── Serving ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
