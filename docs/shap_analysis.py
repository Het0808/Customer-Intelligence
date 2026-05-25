#!/usr/bin/env python3
"""
SHAP governance analysis — Customer Intelligence Platform.

Produces per-prediction force plots, a global summary bar chart, and a
segment error analysis table broken down by job category and marital status.

Outputs (all written to docs/shap_samples/):
  force_plot_00.png ... force_plot_09.png   per-prediction waterfall plots
  summary_bar.png                           global mean |SHAP| bar chart
  segment_error_report.md                   F1/precision/recall by segment

Usage:
    python docs/shap_analysis.py

Requirements:
    pip install shap matplotlib
    (pyarrow >= 18.0 required for NumPy 2.x compatibility)

Design notes:
  - Uses the same 60/20/20 stratified split (seed=42) as training so the
    10 held-out predictions come from the same unseen test rows.
  - SHAP TreeExplainer operates on the raw XGBClassifier, not the sklearn
    Pipeline, to avoid the wrapper overhead on shap value calculation.
  - The preprocessing pipeline is fit on X_train only (no leakage).
  - Segment analysis uses the full test set (not just the 10 explained rows)
    so F1 per segment is computed on a realistic sample size.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")     # headless — no display needed
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, SAMPLES_DIR
from src.data_pipeline.features import (
    build_preprocessing_pipeline,
    encode_target,
    get_feature_names,
)

# ── Config ────────────────────────────────────────────────────────────────────
BANK_SAMPLE = SAMPLES_DIR / "bank_marketing_sample.csv"
OUT_DIR     = Path(__file__).resolve().parent / "shap_samples"
N_EXPLAIN   = 10
SEED        = 42

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_promoted_model():
    """
    Load the most recently PROMOTED full_pipeline from MLflow.
    Returns (full_sklearn_pipeline, xgb_model, threshold, model_name).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found. "
            "Run: python -m src.training.train --sample"
        )

    runs = client.search_runs(
        experiment_ids = [exp.experiment_id],
        filter_string  = "tags.gate_decision = 'PROMOTED'",
        order_by       = ["start_time DESC"],
    )
    for run in runs:
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
        if "full_pipeline" in artifacts:
            pipe = mlflow.sklearn.load_model(
                f"{run.info.artifact_uri}/full_pipeline"
            )
            threshold  = float(run.data.metrics.get("val_threshold", 0.5))
            model_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
            xgb_model  = pipe.named_steps["model"]
            return pipe, xgb_model, threshold, model_name

    raise RuntimeError(
        "No PROMOTED run with full_pipeline found. "
        "Run: python -m src.training.train --sample"
    )


def _band(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def _save_waterfall(
    explainer:     shap.TreeExplainer,
    shap_vals:     np.ndarray,
    features_row:  np.ndarray,
    feature_names: list[str],
    prob:          float,
    truth:         int,
    pred:          int,
    idx:           int,
    out_path:      Path,
) -> None:
    """Save a SHAP waterfall plot for one prediction as a PNG."""
    explanation = shap.Explanation(
        values       = shap_vals,
        base_values  = float(explainer.expected_value),
        data         = features_row,
        feature_names= feature_names,
    )
    # waterfall_plot creates its own figure via plt
    shap.plots.waterfall(explanation, max_display=15, show=False)
    fig = plt.gcf()
    correct = "CORRECT" if pred == truth else "WRONG"
    fig.suptitle(
        f"Sample {idx:02d}  |  P(subscribe) = {prob:.3f}  "
        f"|  band = {_band(prob)}  "
        f"|  truth = {'yes' if truth else 'no'}  "
        f"|  pred = {'yes' if pred else 'no'}  [{correct}]",
        fontsize = 9,
        y = 1.01,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_summary_bar(
    shap_vals:     np.ndarray,
    feature_names: list[str],
    model_name:    str,
    out_path:      Path,
    top_n:         int = 20,
) -> None:
    """Save a global mean |SHAP| bar chart (horizontal, top-N features)."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.38)))
    y_pos   = np.arange(top_n)
    vals    = mean_abs[top_idx][::-1]   # ascending for barh
    labels  = [feature_names[j] for j in top_idx[::-1]]

    bars = ax.barh(y_pos, vals, color="steelblue", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (log-odds units)")
    ax.set_title(
        f"Global Feature Importance via SHAP\n{model_name}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(
            val + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=7, color="black",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary bar chart saved → %s", out_path)


def _segment_error_analysis(
    X_test_orig: pd.DataFrame,
    y_test_arr:  np.ndarray,
    prob_arr:    np.ndarray,
    threshold:   float,
    model_name:  str,
    out_path:    Path,
) -> pd.DataFrame:
    """
    Compute precision / recall / F1 for each job and marital sub-group on the
    test set.  Groups with fewer than 2 samples are skipped.
    Returns the results DataFrame (also written to out_path as Markdown).
    """
    pred_arr = (prob_arr >= threshold).astype(int)
    rows     = []

    for seg_type, col in [("job", "job"), ("marital", "marital")]:
        for val in sorted(X_test_orig[col].unique()):
            mask = (X_test_orig[col] == val).to_numpy()
            if mask.sum() < 2:
                continue
            yt = y_test_arr[mask]
            yp = pred_arr[mask]
            rows.append({
                "segment_type":  seg_type,
                "segment_value": val,
                "n":             int(mask.sum()),
                "n_pos":         int(yt.sum()),
                "prevalence":    round(float(yt.mean()) * 100, 1),
                "f1":            round(float(f1_score(yt, yp, zero_division=0)), 4),
                "precision":     round(float(precision_score(yt, yp, zero_division=0)), 4),
                "recall":        round(float(recall_score(yt, yp, zero_division=0)), 4),
            })

    seg_df = pd.DataFrame(rows).sort_values(["segment_type", "f1"])

    # ── Markdown report ────────────────────────────────────────────────────────
    n_test  = len(y_test_arr)
    n_pos   = int(y_test_arr.sum())
    overall_f1 = float(f1_score(y_test_arr, pred_arr, zero_division=0))

    lines = [
        "# SHAP Segment Error Analysis",
        "",
        f"**Model:** `{model_name}`  |  **Threshold:** `{threshold:.4f}`",
        f"**Test set:** {n_test} rows, {n_pos} positives ({n_pos/n_test*100:.1f}%)",
        f"**Overall F1:** {overall_f1:.4f}",
        "",
        "---",
        "",
        "## By Job Category",
        "",
        "| Job | N | N+ | Prev% | F1 | Precision | Recall |",
        "|-----|---|----|-------|-----|-----------|--------|",
    ]
    for _, r in seg_df[seg_df["segment_type"] == "job"].iterrows():
        lines.append(
            f"| {r['segment_value']} | {r['n']} | {r['n_pos']} "
            f"| {r['prevalence']:.1f}% "
            f"| **{r['f1']:.4f}** | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    lines += [
        "",
        "## By Marital Status",
        "",
        "| Marital | N | N+ | Prev% | F1 | Precision | Recall |",
        "|---------|---|----|-------|-----|-----------|--------|",
    ]
    for _, r in seg_df[seg_df["segment_type"] == "marital"].iterrows():
        lines.append(
            f"| {r['segment_value']} | {r['n']} | {r['n_pos']} "
            f"| {r['prevalence']:.1f}% "
            f"| **{r['f1']:.4f}** | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    # Key findings: worst and best segments
    job_seg     = seg_df[seg_df["segment_type"] == "job"]
    marital_seg = seg_df[seg_df["segment_type"] == "marital"]

    findings = [
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]
    if not job_seg.empty:
        worst_job = job_seg.iloc[0]
        best_job  = job_seg.iloc[-1]
        findings += [
            f"- **Worst job segment**: `{worst_job['segment_value']}` — "
            f"F1={worst_job['f1']:.4f}, N={worst_job['n']}, "
            f"N+={worst_job['n_pos']} ({worst_job['prevalence']:.1f}% positive rate).",
            f"  This segment underperforms overall F1 ({overall_f1:.4f}) by "
            f"{overall_f1 - worst_job['f1']:.4f} points.",
            f"- **Best job segment**: `{best_job['segment_value']}` — "
            f"F1={best_job['f1']:.4f}.",
        ]
    if not marital_seg.empty:
        worst_marital = marital_seg.iloc[0]
        findings += [
            f"- **Worst marital segment**: `{worst_marital['segment_value']}` — "
            f"F1={worst_marital['f1']:.4f}, N={worst_marital['n']}, "
            f"N+={worst_marital['n_pos']}.",
        ]

    findings += [
        "",
        "### Interpretation",
        "",
        "Segments with F1 below the overall mean are candidates for:",
        "1. **Threshold adjustment**: lower the decision threshold for these segments",
        "   to improve recall at the cost of precision.",
        "2. **Feature engineering**: add segment-specific features (e.g. interaction",
        "   terms for job × duration or marital × campaign).",
        "3. **Data augmentation**: if N+ is 0–1 in a segment, the model has no",
        "   positive examples to learn from — these segments are effectively",
        "   excluded from training signal.",
        "",
        "### Caveat",
        "",
        "All results are on the 100-row test split of the 500-row committed sample.",
        "Cell sizes are too small for statistically significant conclusions.",
        "Repeat this analysis after training on the full 41,163-row UCI dataset.",
    ]

    lines.extend(findings)
    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    log.info("Segment error report saved → %s", out_path)
    return seg_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BANK_SAMPLE.exists():
        log.error("Sample not found at %s — run:  python -m src.training.train --sample", BANK_SAMPLE)
        sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading sample from %s", BANK_SAMPLE)
    df = pd.read_csv(BANK_SAMPLE, sep=",")
    X  = df.drop(columns=["y"])
    y  = encode_target(df["y"])

    # Exact same 60/20/20 split used during training
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    log.info("Split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    # ── Preprocessing (fit on train only — no leakage) ────────────────────────
    log.info("Fitting preprocessing pipeline on train split …")
    preproc = build_preprocessing_pipeline()
    preproc.fit(X_train)
    X_test_t  = preproc.transform(X_test)
    feature_names = get_feature_names(preproc)
    log.info("Feature matrix: %d features", len(feature_names))

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Loading promoted model from MLflow …")
    full_pipe, xgb_model, threshold, model_name = _load_promoted_model()
    log.info("Model: %s | threshold: %.4f", model_name, threshold)

    # ── Probabilities for test set ─────────────────────────────────────────────
    # Use full_pipe to get probabilities (it carries its own fitted preprocessor
    # from training — different from our refitted preproc above, but that's fine
    # for segment analysis which only needs predictions)
    probs_test = full_pipe.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= threshold).astype(int)
    y_test_arr = y_test.reset_index(drop=True).to_numpy()

    overall_f1 = f1_score(y_test_arr, preds_test, zero_division=0)
    log.info(
        "Test set: n=%d  n_pos=%d  overall_F1=%.4f",
        len(y_test_arr), y_test_arr.sum(), overall_f1,
    )

    # ── SHAP explainer on the raw XGBoost model ────────────────────────────────
    log.info("Building SHAP TreeExplainer …")
    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_test_t)  # (n_test, n_features)
    expected_v = float(explainer.expected_value)
    log.info(
        "SHAP expected value (log-odds): %.4f  |  shap_vals shape: %s",
        expected_v, shap_vals.shape,
    )

    # ── Per-prediction waterfall plots ─────────────────────────────────────────
    log.info("Saving %d waterfall plots …", N_EXPLAIN)
    X_test_arr = X_test_t if isinstance(X_test_t, np.ndarray) else X_test_t.toarray()

    for i in range(min(N_EXPLAIN, len(X_test_arr))):
        out_path = OUT_DIR / f"force_plot_{i:02d}.png"
        _save_waterfall(
            explainer     = explainer,
            shap_vals     = shap_vals[i],
            features_row  = X_test_arr[i],
            feature_names = feature_names,
            prob          = float(probs_test[i]),
            truth         = int(y_test_arr[i]),
            pred          = int(preds_test[i]),
            idx           = i,
            out_path      = out_path,
        )
        log.info("  [%02d] P=%.3f  truth=%s  pred=%s  → %s",
                 i, probs_test[i],
                 "yes" if y_test_arr[i] else "no",
                 "yes" if preds_test[i] else "no",
                 out_path.name)

    # ── Global summary bar chart ───────────────────────────────────────────────
    _save_summary_bar(
        shap_vals     = shap_vals,
        feature_names = feature_names,
        model_name    = model_name,
        out_path      = OUT_DIR / "summary_bar.png",
    )

    # ── Segment error analysis ─────────────────────────────────────────────────
    log.info("Computing segment error analysis …")
    seg_df = _segment_error_analysis(
        X_test_orig = X_test.reset_index(drop=True),
        y_test_arr  = y_test_arr,
        prob_arr    = probs_test,
        threshold   = threshold,
        model_name  = model_name,
        out_path    = OUT_DIR / "segment_error_report.md",
    )

    # ── Print SHAP top-10 features to terminal ─────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top10    = np.argsort(mean_abs)[::-1][:10]

    w = 60
    print(f"\n{'=' * w}")
    print(f"  SHAP Analysis Complete — {model_name}")
    print(f"{'=' * w}")
    print(f"  {'Rank':<5} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'-' * 54}")
    for rank, j in enumerate(top10, 1):
        print(f"  {rank:<5} {feature_names[j]:<35} {mean_abs[j]:>12.5f}")

    print(f"\n  Overall test F1 @ threshold {threshold:.4f}: {overall_f1:.4f}")
    print("\n  Segment summary (sorted by F1 ascending):")
    print(f"  {'Type':<8} {'Value':<20} {'N':>4} {'N+':>4} {'F1':>7}")
    print(f"  {'-' * 50}")
    for _, r in seg_df.iterrows():
        print(f"  {r['segment_type']:<8} {r['segment_value']:<20} "
              f"{r['n']:>4} {r['n_pos']:>4} {r['f1']:>7.4f}")

    print("\n  Outputs:")
    print(f"    Force plots : {OUT_DIR}/force_plot_00.png … force_plot_09.png")
    print(f"    Summary bar : {OUT_DIR}/summary_bar.png")
    print(f"    Seg report  : {OUT_DIR}/segment_error_report.md")
    print(f"{'=' * w}\n")


if __name__ == "__main__":
    main()
