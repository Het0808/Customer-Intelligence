"""
Promotion gate and business-reading report for bank campaign models.

Gate rule (both conditions must hold to PROMOTE):
  (1) PR-AUC(candidate) - PR-AUC(baseline) >= PR_AUC_DELTA_MIN   (+3 pp)
  (2) F1(baseline)      - F1(candidate)     <= F1_DROP_MAX         (<= 2 pp worse)

Why these margins:
  +3 pp PR-AUC: At 11 % prevalence the random-baseline PR-AUC is ~0.11.
  A well-tuned LR typically reaches 0.50-0.60 on this dataset. A 3 pp lift
  is above the noise from a 60/20 train/val split (~40 k rows) and represents
  ~5-6 % relative improvement -- business-meaningful.  Choosing 1 pp would
  promote on noise; choosing 10 pp would block real improvements.

  2 pp F1 guard: F1 prevents trading precision for recall to chase AUC.
  In a bank campaign each contact costs money; a model that flags 90 % of
  customers as converters increases cost without proportional revenue. The
  2 pp tolerance absorbs random split variance without masking real degradation.

Usage (standalone):
    python -m src.training.evaluate --baseline-run <RUN_ID> --candidate-run <RUN_ID>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# -----------------------------------------------------------------------------
# Gate constants
# -----------------------------------------------------------------------------
PR_AUC_DELTA_MIN: float = 0.03   # candidate must beat baseline by at least 3 pp
F1_DROP_MAX: float      = 0.02   # candidate F1 may not be more than 2 pp worse


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """All evaluation metrics for a single model on a single split."""
    roc_auc:   float
    pr_auc:    float
    f1:        float
    precision: float
    recall:    float
    threshold: float
    n_pos:     int
    n_total:   int


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on this split."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one extra element at the end
    p, r = precision_arr[:-1], recall_arr[:-1]
    denom = np.maximum(p + r, 1e-10)         # avoid true division by zero
    f1_arr = np.where((p + r) == 0, 0.0, 2 * p * r / denom)
    if len(f1_arr) == 0:
        return 0.5
    return float(thresholds[np.argmax(f1_arr)])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float | None = None,
) -> ModelMetrics:
    """
    Compute all gate-relevant metrics.

    If threshold is None, the threshold that maximises F1 on this split is used
    (appropriate for val/test evaluation).  For production scoring a fixed
    threshold should be passed.
    """
    if threshold is None:
        threshold = _optimal_f1_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    return ModelMetrics(
        roc_auc   = float(roc_auc_score(y_true, y_prob)),
        pr_auc    = float(average_precision_score(y_true, y_prob)),
        f1        = float(f1_score(y_true, y_pred, zero_division=0)),
        precision = float(np.mean(y_pred[y_pred == 1] == y_true[y_pred == 1])
                          if y_pred.sum() > 0 else 0.0),
        recall    = float(y_pred[y_true == 1].sum() / max(y_true.sum(), 1)),
        threshold = threshold,
        n_pos     = int(y_true.sum()),
        n_total   = int(len(y_true)),
    )


# -----------------------------------------------------------------------------
# Promotion gate
# -----------------------------------------------------------------------------
class GateResult(NamedTuple):
    decision: str    # "PROMOTED" or "BLOCKED"
    reason:   str    # human-readable explanation
    pr_delta: float  # candidate PR-AUC - baseline PR-AUC
    f1_delta: float  # candidate F1    - baseline F1


def promotion_gate(
    baseline:  ModelMetrics,
    candidate: ModelMetrics,
    *,
    pr_auc_delta_min: float = PR_AUC_DELTA_MIN,
    f1_drop_max:      float = F1_DROP_MAX,
) -> GateResult:
    """
    Decide whether candidate should replace baseline in production.

    Both conditions must hold:
      (1) PR-AUC improvement  >= pr_auc_delta_min
      (2) F1 regression       <= f1_drop_max

    Returns a GateResult with decision, reason, and the raw deltas for logging.
    """
    pr_delta = candidate.pr_auc - baseline.pr_auc
    f1_delta = candidate.f1    - baseline.f1      # positive = improvement

    gate_pr  = pr_delta >= pr_auc_delta_min
    gate_f1  = f1_delta >= -f1_drop_max           # allow up to f1_drop_max regression

    if gate_pr and gate_f1:
        reason = (
            f"PR-AUC +{pr_delta:.4f} (>= +{pr_auc_delta_min:.2f} threshold)  |  "
            f"F1 delta {f1_delta:+.4f} (within -{f1_drop_max:.2f} tolerance)"
        )
        return GateResult("PROMOTED", reason, pr_delta, f1_delta)

    # Collect specific failure reasons
    failures = []
    if not gate_pr:
        failures.append(
            f"PR-AUC delta {pr_delta:+.4f} < required +{pr_auc_delta_min:.2f}"
        )
    if not gate_f1:
        failures.append(
            f"F1 regressed {f1_delta:+.4f} (limit -{f1_drop_max:.2f})"
        )
    return GateResult("BLOCKED", "  |  ".join(failures), pr_delta, f1_delta)


# -----------------------------------------------------------------------------
# Business reading
# -----------------------------------------------------------------------------
def business_reading(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
    """
    Translate model output into campaign-operation language.

    Returns a dict with:
      threshold          -- decision cutoff used
      pct_contacted      -- % of the population the model would flag to call
      pct_converters_captured -- % of actual subscribers captured (recall)
      precision          -- fraction of flagged customers who actually subscribe
      contacts_per_subscriber -- expected calls per successful subscription
      lift               -- recall / (% contacted)  [>1 means better than random]
    """
    if threshold is None:
        threshold = _optimal_f1_threshold(y_true, y_prob)

    y_pred   = (y_prob >= threshold).astype(int)
    n        = len(y_true)
    n_pos    = y_true.sum()
    flagged  = y_pred.sum()
    tp       = int((y_pred & y_true).sum())

    pct_contacted   = flagged / n
    recall          = tp / max(n_pos, 1)
    prec            = tp / max(flagged, 1)
    return {
        "threshold":                   round(float(threshold), 4),
        "pct_contacted":               round(float(pct_contacted) * 100, 1),
        "pct_converters_captured":     round(float(recall) * 100, 1),
        "precision":                   round(float(prec) * 100, 1),
        "contacts_per_subscriber":     round(float(1 / max(prec, 1e-9)), 1),
        "lift_vs_random":              round(float(recall / max(pct_contacted, 1e-9)), 2),
        "n_flagged":                   int(flagged),
        "n_total":                     int(n),
        "true_positive_rate":          round(float(recall) * 100, 1),
        "false_positive_rate":         round(
            float((y_pred & ~y_true.astype(bool)).sum() / max((~y_true.astype(bool)).sum(), 1)) * 100, 1
        ),
    }


def print_business_report(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float | None = None,
) -> None:
    """Print a formatted campaign-operations report for one model."""
    br = business_reading(y_true, y_prob, threshold=threshold)
    print(f"\n{'-' * 62}")
    print(f"  Business Reading  --  {model_name}")
    print(f"{'-' * 62}")
    print(f"  Decision threshold           : {br['threshold']:.4f}")
    print(f"  Customers flagged to call    : {br['pct_contacted']:.1f} %  "
          f"({br['n_flagged']:,} of {br['n_total']:,})")
    print(
        f"  Converters captured          : {br['pct_converters_captured']:.1f} %  "
        f"(recall)"
    )
    print(f"  Precision (of calls placed)  : {br['precision']:.1f} %")
    print(f"  Calls needed per subscriber  : {br['contacts_per_subscriber']:.1f}")
    print(f"  Lift vs. random contact      : {br['lift_vs_random']:.2f}x")
    print(
        f"\n  -> At threshold {br['threshold']:.2f}, we contact "
        f"{br['pct_contacted']:.1f} % of customers "
        f"and capture {br['pct_converters_captured']:.1f} % of converters.\n"
    )


def print_gate_result(
    baseline_name:  str,
    candidate_name: str,
    baseline_m:     ModelMetrics,
    candidate_m:    ModelMetrics,
    gate:           GateResult,
) -> None:
    """Print a formatted gate decision table."""
    w = 62
    print(f"\n{'=' * w}")
    print(f"  PROMOTION GATE: {candidate_name} vs {baseline_name}")
    print(f"{'=' * w}")
    print(f"  {'Metric':<22}  {'Baseline':>10}  {'Candidate':>10}  {'Delta':>10}")
    print(f"  {'-' * 54}")
    print(f"  {'PR-AUC':<22}  {baseline_m.pr_auc:>10.4f}  {candidate_m.pr_auc:>10.4f}  "
          f"  {gate.pr_delta:>+8.4f}")
    print(f"  {'ROC-AUC':<22}  {baseline_m.roc_auc:>10.4f}  {candidate_m.roc_auc:>10.4f}  "
          f"  {candidate_m.roc_auc - baseline_m.roc_auc:>+8.4f}")
    print(f"  {'F1 (opt threshold)':<22}  {baseline_m.f1:>10.4f}  {candidate_m.f1:>10.4f}  "
          f"  {gate.f1_delta:>+8.4f}")
    print(f"  {'Threshold':<22}  {baseline_m.threshold:>10.4f}  {candidate_m.threshold:>10.4f}")
    print(f"{'-' * w}")
    symbol = "[OK] PROMOTED" if gate.decision == "PROMOTED" else "[X] BLOCKED "
    print(f"  {symbol}  --  {gate.reason}")
    print(f"{'=' * w}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline-run",  help="MLflow run ID for baseline model")
    p.add_argument("--candidate-run", help="MLflow run ID for candidate model")
    return p.parse_args()


def main() -> None:
    """Re-run gate from saved MLflow metrics (standalone mode)."""
    args = _parse_args()
    if not args.baseline_run or not args.candidate_run:
        print("Pass --baseline-run and --candidate-run to evaluate saved runs.")
        print("Otherwise run  python -m src.training.train  which calls the gate automatically.")
        raise SystemExit(0)

    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        def _load(run_id: str) -> ModelMetrics:
            m = client.get_run(run_id).data.metrics
            return ModelMetrics(
                roc_auc   = m["val_roc_auc"],
                pr_auc    = m["val_pr_auc"],
                f1        = m["val_f1"],
                precision = m.get("val_precision", 0.0),
                recall    = m.get("val_recall",    0.0),
                threshold = m.get("val_threshold", 0.5),
                n_pos     = int(m.get("n_pos", 0)),
                n_total   = int(m.get("n_total", 0)),
            )
        baseline_m  = _load(args.baseline_run)
        candidate_m = _load(args.candidate_run)
        gate = promotion_gate(baseline_m, candidate_m)
        print_gate_result(
            args.baseline_run, args.candidate_run,
            baseline_m, candidate_m, gate,
        )
        sys.exit(0 if gate.decision == "PROMOTED" else 1)
    except Exception as exc:
        print(f"Error loading MLflow runs: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
