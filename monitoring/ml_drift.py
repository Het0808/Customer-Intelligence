#!/usr/bin/env python3
"""
ML drift monitoring report for the Customer Intelligence Platform.

Generates an Evidently HTML drift report comparing a reference snapshot
(first 250 rows of the committed sample) against a synthetic current
dataset with three deliberate distribution shifts:

  Shift 1 -- Demographic:         age += 10, clipped to UCI upper bound 98
  Shift 2 -- Data quality:        15 % of campaign values set to NaN
  Shift 3 -- Categorical collapse: "blue-collar" -> "services" (category drift)

Retrain trigger logic:
  Evidently reports a KS-test p-value as drift_score for each numeric feature
  (lower p-value = stronger drift).  We convert to drift_intensity = 1 - p_value
  so the scale is intuitive (higher = more drift).  If drift_intensity > 0.3 for
  ANY feature, RETRAIN TRIGGERED is printed and the feature list is reported.

Output:
  monitoring/reports/ml_drift_report.html  -- interactive Evidently dashboard

Usage:
    python monitoring/ml_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import SAMPLES_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
BANK_SAMPLE             = SAMPLES_DIR / "bank_marketing_sample.csv"
REPORT_DIR              = Path(__file__).resolve().parent / "reports"
REPORT_PATH             = REPORT_DIR / "ml_drift_report.html"
DRIFT_TRIGGER_THRESHOLD = 0.3   # drift_intensity = 1 - KS_pvalue > this → retrain

FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign",
    "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]


# ── Data helpers ──────────────────────────────────────────────────────────────
def _load_reference(n: int = 250) -> pd.DataFrame:
    """Load first *n* rows of the bank sample as the reference distribution."""
    df = pd.read_csv(BANK_SAMPLE, sep=",")
    return df[FEATURE_COLS].head(n).reset_index(drop=True)


def _make_synthetic_current(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Apply three deliberate distribution shifts to the reference data.

    Shift 1 -- Demographic:         age += 10 (older-cohort campaign)
    Shift 2 -- Data quality:        15 % of 'campaign' values set to NaN
    Shift 3 -- Categorical collapse: blue-collar workers re-labelled 'services'
    """
    current = reference.copy()

    # Shift 1: age distribution shift (older cohort)
    current["age"] = (current["age"] + 10).clip(upper=98)

    # Shift 2: introduce 15 % missing values in the campaign column
    null_idx = current.sample(frac=0.15, random_state=42).index
    current.loc[null_idx, "campaign"] = float("nan")

    # Shift 3: job category collapse (blue-collar absorbed into services)
    current["job"] = current["job"].replace("blue-collar", "services")

    return current


# ── Evidently report ──────────────────────────────────────────────────────────
def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Build an Evidently DataDrift report.
    Saves HTML to REPORT_PATH and returns the report as a dict.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(REPORT_PATH))
    print(f"Drift report saved -> {REPORT_PATH}")

    return report.as_dict()


def extract_drift_summary(result_dict: dict) -> dict[str, float]:
    """
    Extract per-feature drift intensity from the Evidently result dict.

    Evidently 0.4.x layout:
      result_dict["metrics"][1]["result"]["drift_by_columns"][col]["drift_score"]

    drift_score is the KS test p-value for numeric columns; lower = more drift.
    We return drift_intensity = 1 - p_value so higher = more drift.
    """
    drift_by_col: dict = {}
    try:
        drift_by_col = result_dict["metrics"][1]["result"]["drift_by_columns"]
    except (KeyError, IndexError, TypeError):
        # Search all metric entries for the drift_by_columns key
        for metric in result_dict.get("metrics", []):
            candidate = metric.get("result", {}).get("drift_by_columns")
            if candidate:
                drift_by_col = candidate
                break

    summary: dict[str, float] = {}
    for col, info in drift_by_col.items():
        p_value = info.get("drift_score", 1.0)
        summary[col] = round(1.0 - float(p_value), 4)
    return summary


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_drift_summary(summary: dict[str, float], threshold: float) -> list[str]:
    """
    Print a formatted drift intensity table.
    Returns the list of feature names that exceeded the trigger threshold.
    """
    w = 64
    print(f"\n{'=' * w}")
    print("  ML Drift Monitor -- Customer Intelligence Platform")
    print(f"  Retrain threshold: drift_intensity > {threshold}")
    print(f"{'=' * w}")
    print(f"  {'Feature':<34}  {'Drift Intensity':>15}  Status")
    print(f"  {'-' * 34}  {'-' * 15}  ------")

    triggered: list[str] = []
    for feature, intensity in sorted(summary.items(), key=lambda x: -x[1]):
        status = "DRIFT" if intensity > threshold else "ok"
        if intensity > threshold:
            triggered.append(feature)
        print(f"  {feature:<34}  {intensity:>15.4f}  {status}")

    print(f"{'=' * w}")
    if triggered:
        print("\n  *** RETRAIN TRIGGERED ***")
        print(f"  Features with drift_intensity > {threshold}:")
        for feat in triggered:
            print(f"    - {feat}")
    else:
        print("\n  NO RETRAIN -- all features within tolerance.")
    print(f"{'=' * w}\n")
    return triggered


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not BANK_SAMPLE.exists():
        print(f"ERROR: sample not found at {BANK_SAMPLE}", file=sys.stderr)
        sys.exit(1)

    reference   = _load_reference(n=250)
    current     = _make_synthetic_current(reference)
    result_dict = run_drift_report(reference, current)
    summary     = extract_drift_summary(result_dict)
    print_drift_summary(summary, DRIFT_TRIGGER_THRESHOLD)


if __name__ == "__main__":
    main()
