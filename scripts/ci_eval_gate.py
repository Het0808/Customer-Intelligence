#!/usr/bin/env python3
"""
CI eval-gate: assert the promotion gate correctly BLOCKS a degraded model.

Purpose
-------
This script is run exclusively by the eval-gate CI job.  It trains two models
on the committed 500-row sample, runs the promotion gate from evaluate.py, and
exits non-zero if the degraded candidate is ever PROMOTED -- proving that the
gate logic is enforced in code, not merely documented.

Models
------
  Baseline  : LogisticRegression(class_weight='balanced')   -- the production bar
  Candidate : DummyClassifier(strategy='most_frequent')     -- always predicts 0

Why DummyClassifier instead of max_depth=1 XGBoost:
  On a 500-row sample (300-row training set), a shallow XGBoost ensemble of
  100 trees can outperform LR due to scale_pos_weight handling class imbalance
  better than LR on tiny data.  A DummyClassifier is *analytically* guaranteed
  to be blocked regardless of sample size:
    - predict_proba[:,1] is all-zeros → PR-AUC = prevalence ≈ 0.11
    - F1 = 0 (never predicts positive class)
    Both gate conditions fail:
      PR-AUC delta:  0.11 - 0.63  = -0.52  (need +0.03)   → FAIL
      F1 drop:       0    - 0.63  = -0.63  (max -0.02)    → FAIL

Exit codes
----------
  0  degraded model was BLOCKED  (gate logic is correct -- CI passes)
  1  degraded model was PROMOTED (gate logic is broken  -- CI fails)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import SAMPLES_DIR
from src.data_pipeline.features import build_preprocessing_pipeline, encode_target
from src.training.evaluate import compute_metrics, print_gate_result, promotion_gate

BANK_SAMPLE = SAMPLES_DIR / "bank_marketing_sample.csv"


def main() -> None:
    # ── Load sample ───────────────────────────────────────────────────────────
    if not BANK_SAMPLE.exists():
        print(f"ERROR: sample not found at {BANK_SAMPLE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(BANK_SAMPLE, sep=",")
    X  = df.drop(columns=["y"])
    y  = encode_target(df["y"])

    # ── 80/20 stratified split ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe       = build_preprocessing_pipeline()
    X_train_t  = pipe.fit_transform(X_train)
    X_val_t    = pipe.transform(X_val)
    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42
    )
    lr.fit(X_train_t, y_train_np)
    baseline_m = compute_metrics(y_val_np, lr.predict_proba(X_val_t)[:, 1])

    # ── Degraded candidate: DummyClassifier ───────────────────────────────────
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train_t, y_train_np)
    candidate_m = compute_metrics(y_val_np, dummy.predict_proba(X_val_t)[:, 1])

    # ── Run the gate (reuses production gate logic from evaluate.py) ──────────
    gate = promotion_gate(baseline_m, candidate_m)
    print_gate_result(
        "LogisticRegression_baseline",
        "DummyClassifier_degraded",
        baseline_m,
        candidate_m,
        gate,
    )

    # ── Assert BLOCKED ────────────────────────────────────────────────────────
    if gate.decision == "PROMOTED":
        print(
            "\nFAIL: DummyClassifier_degraded was PROMOTED -- "
            "promotion gate logic is broken!",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nPASS: DummyClassifier_degraded correctly BLOCKED by promotion gate.")
    sys.exit(0)


if __name__ == "__main__":
    main()
