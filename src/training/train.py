#!/usr/bin/env python3
"""
Training script -- Customer Intelligence Platform, Phase 2.

Trains two models on the UCI Bank Marketing dataset and runs a relative
promotion gate to decide if the improved model should replace the baseline.

Models:
  baseline  -- LogisticRegression (balanced class weight)
  improved  -- XGBoostClassifier  (scale_pos_weight to handle imbalance)
  stump     -- XGBoostClassifier(max_depth=1)  intentionally weak, shown BLOCKED

Splits:
  train 60 % | val 20 % | test 20 %  (stratified on target)

MLflow logging per run:
  params  -- model hyperparameters + dataset hash + split sizes
  metrics -- roc_auc, pr_auc, f1, precision, recall, threshold (val + test)
  tags    -- model_type, gate_decision, gate_reason
  artifacts -- confusion matrix, PR curve, calibration curve, feature importance

Usage:
    python -m src.training.train
    python -m src.training.train --sample          # 500-row sample (fast CI)
    python -m src.training.train --include-stump   # also train the blocked demo
    python -m src.training.train --sample --include-stump
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import sys
from pathlib import Path

import tempfile

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from xgboost import XGBClassifier

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# -- project imports -----------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLES_DIR,
)
from src.data_pipeline.features import (
    build_preprocessing_pipeline,
    encode_target,
    get_feature_names,
)
from src.training.evaluate import (
    ModelMetrics,
    business_reading,
    compute_metrics,
    print_business_report,
    print_gate_result,
    promotion_gate,
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
BANK_RAW    = RAW_DIR    / "bank_marketing" / "bank-additional-full.csv"
BANK_SAMPLE = SAMPLES_DIR / "bank_marketing_sample.csv"


def load_data(use_sample: bool = False) -> pd.DataFrame:
    if use_sample:
        path, sep = BANK_SAMPLE, ","
    else:
        path, sep = BANK_RAW, ";"
    if not path.exists():
        log.error("Data not found at %s -- run ingest.py first", path)
        raise SystemExit(1)
    df = pd.read_csv(path, sep=sep)
    before = len(df)
    df = df.drop_duplicates()          # remove 12 duplicate pairs found in Phase 1
    if before != len(df):
        log.info("Dropped %d duplicate rows (%d -> %d)", before - len(df), before, len(df))
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """MD5 of the raw bytes -- reproducibility fingerprint for MLflow."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:10]


# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------
def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Stratified 60/20/20 split.
    We split train from (val+test) first, then split val from test.
    Stratify on y at every step to preserve the ~11 % positive rate.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    log.info(
        "Split sizes -- train %d | val %d | test %d  "
        "(positive rate: train %.3f | val %.3f | test %.3f)",
        len(y_train), len(y_val), len(y_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# MLflow plot helpers
# -----------------------------------------------------------------------------
def _plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    pr_auc: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC={pr_auc:.3f})")
    ax.axhline(y=y_true.mean(), ls="--", color="grey", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve -- {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    fig.tight_layout()
    return fig


def _plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve -- {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    top_n: int = 20,
) -> plt.Figure | None:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, max(5, top_n * 0.35)))
    ax.barh([feature_names[i] for i in idx], imp[idx])
    ax.set_title(f"Feature Importances -- {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def _log_fig(fig: plt.Figure | None, name: str) -> None:
    """Save figure to a temp PNG and log it as an MLflow artifact."""
    if fig is None:
        return
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, format="png", dpi=120, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Single-model training + MLflow logging
# -----------------------------------------------------------------------------
def train_and_log(
    model_name:       str,
    model,
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    X_val:            np.ndarray,
    y_val:            np.ndarray,
    X_test:           np.ndarray,
    y_test:           np.ndarray,
    feature_names:    list[str],
    extra_params:     dict,
    run_tags:         dict,
    preproc_pipeline=None,   # sklearn Pipeline; if given, saves full preprocess+model pipeline
) -> tuple[ModelMetrics, ModelMetrics, str]:
    """
    Fit model, evaluate on val + test, log everything to MLflow.
    Returns (val_metrics, test_metrics, run_id).
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # -- Hyperparams -------------------------------------------------------
        mlflow.log_params({**extra_params, **run_tags})
        mlflow.set_tags({**run_tags, "model_name": model_name})

        # -- Train -------------------------------------------------------------
        log.info("Training %s …", model_name)
        model.fit(X_train, y_train)

        # -- Evaluate ----------------------------------------------------------
        val_prob  = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        val_m  = compute_metrics(y_val,  val_prob)
        test_m = compute_metrics(y_test, test_prob)

        # Use val-optimal threshold on test for consistent comparison
        test_m_at_val_thr = compute_metrics(
            y_test, test_prob, threshold=val_m.threshold
        )

        # -- Log metrics -------------------------------------------------------
        mlflow.log_metrics({
            "val_roc_auc":   val_m.roc_auc,
            "val_pr_auc":    val_m.pr_auc,
            "val_f1":        val_m.f1,
            "val_precision": val_m.precision,
            "val_recall":    val_m.recall,
            "val_threshold": val_m.threshold,
            "test_roc_auc":  test_m.roc_auc,
            "test_pr_auc":   test_m.pr_auc,
            "test_f1":       test_m.f1,
            "test_f1_at_val_threshold": test_m_at_val_thr.f1,
            "n_pos":   float(val_m.n_pos),
            "n_total": float(val_m.n_total),
        })

        # -- Artifacts ---------------------------------------------------------
        y_pred_val = (val_prob  >= val_m.threshold).astype(int)
        y_pred_tst = (test_prob >= val_m.threshold).astype(int)

        _log_fig(_plot_pr_curve(y_val,  val_prob,  model_name, val_m.pr_auc),
                 f"pr_curve_val_{model_name}.png")
        _log_fig(_plot_pr_curve(y_test, test_prob, model_name, test_m.pr_auc),
                 f"pr_curve_test_{model_name}.png")
        _log_fig(_plot_confusion_matrix(y_val,  y_pred_val, model_name),
                 f"cm_val_{model_name}.png")
        _log_fig(_plot_confusion_matrix(y_test, y_pred_tst, model_name),
                 f"cm_test_{model_name}.png")
        _log_fig(_plot_calibration(y_val, val_prob, model_name),
                 f"calibration_{model_name}.png")
        _log_fig(
            _plot_feature_importance(model, feature_names, model_name),
            f"feature_importance_{model_name}.png",
        )

        # -- Log model ---------------------------------------------------------
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save full preprocess+model pipeline so serving can call predict_proba
        # on raw DataFrames without needing to apply the scaler/OHE separately.
        if preproc_pipeline is not None:
            full_pipe = SKPipeline([
                ("preprocess", preproc_pipeline),
                ("model", model),
            ])
            mlflow.sklearn.log_model(full_pipe, artifact_path="full_pipeline")
            log.info("Logged full_pipeline (preprocess+model) to MLflow artifact path 'full_pipeline'")

        log.info(
            "%s | val PR-AUC %.4f | val F1 %.4f | test PR-AUC %.4f",
            model_name, val_m.pr_auc, val_m.f1, test_m.pr_auc,
        )

    return val_m, test_m, run_id


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sample", action="store_true",
        help="Use 500-row sample instead of full dataset (fast CI mode)",
    )
    p.add_argument(
        "--include-stump", action="store_true",
        help="Also train XGBoost(max_depth=1) to demonstrate a BLOCKED gate",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # -- MLflow setup ----------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    log.info("MLflow tracking: %s | experiment: %s",
             MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    # -- Load data -------------------------------------------------------------
    df = load_data(use_sample=args.sample)
    dhash = dataset_hash(df)
    log.info("Dataset loaded: %d rows | hash=%s | sample=%s",
             len(df), dhash, args.sample)

    # -- Feature engineering & target -----------------------------------------
    X = df.drop(columns=["y"])
    y = encode_target(df["y"])

    # -- Stratified splits -----------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        X, y, seed=args.seed
    )

    # -- Preprocessing pipeline (fit on train only) ----------------------------
    log.info("Fitting preprocessing pipeline on training data …")
    pipe = build_preprocessing_pipeline()
    X_train_t = pipe.fit_transform(X_train)
    X_val_t   = pipe.transform(X_val)
    X_test_t  = pipe.transform(X_test)
    feature_names = get_feature_names(pipe)

    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()
    y_test_np  = y_test.to_numpy()

    pos_rate   = y_train_np.mean()
    neg_pos    = (1.0 - pos_rate) / max(pos_rate, 1e-9)  # for XGB scale_pos_weight

    shared_tags = {
        "dataset_hash":  dhash,
        "n_train":       len(y_train),
        "n_val":         len(y_val),
        "n_test":        len(y_test),
        "pos_rate_train": round(float(pos_rate), 4),
        "use_sample":    str(args.sample),
    }

    # -- Baseline: Logistic Regression -----------------------------------------
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",   # reweights samples by pos_rate internally
        max_iter=1000,
        solver="lbfgs",
        random_state=args.seed,
    )
    baseline_val_m, baseline_test_m, baseline_run_id = train_and_log(
        model_name        = "LogisticRegression_baseline",
        model             = lr,
        X_train           = X_train_t,
        y_train           = y_train_np,
        X_val             = X_val_t,
        y_val             = y_val_np,
        X_test            = X_test_t,
        y_test            = y_test_np,
        feature_names     = feature_names,
        extra_params      = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        run_tags          = {**shared_tags, "model_type": "LogisticRegression"},
        preproc_pipeline  = pipe,
    )

    # -- Improved: XGBoost -----------------------------------------------------
    xgb = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = neg_pos,    # mirror of class_weight='balanced' for XGB
        eval_metric      = "aucpr",
        verbosity        = 0,
        random_state     = args.seed,
    )
    improved_val_m, improved_test_m, improved_run_id = train_and_log(
        model_name        = "XGBoost_improved",
        model             = xgb,
        X_train           = X_train_t,
        y_train           = y_train_np,
        X_val             = X_val_t,
        y_val             = y_val_np,
        X_test            = X_test_t,
        y_test            = y_test_np,
        feature_names     = feature_names,
        extra_params      = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": round(neg_pos, 2),
        },
        run_tags          = {**shared_tags, "model_type": "XGBoostClassifier"},
        preproc_pipeline  = pipe,
    )

    # -- Gate: baseline vs improved --------------------------------------------
    gate_main = promotion_gate(baseline_val_m, improved_val_m)
    print_gate_result(
        "LogisticRegression", "XGBoost_improved",
        baseline_val_m, improved_val_m, gate_main,
    )

    # Write gate decision back to the XGBoost MLflow run
    with mlflow.start_run(run_id=improved_run_id):
        mlflow.set_tags({
            "gate_decision": gate_main.decision,
            "gate_reason":   gate_main.reason,
        })

    # -- Optional: stump model (demonstrates BLOCKED gate) ---------------------
    if args.include_stump:
        log.info("Training stump model to demonstrate BLOCKED gate …")
        stump = XGBClassifier(
            n_estimators     = 100,
            max_depth        = 1,            # deliberately weak -- single-level splits
            learning_rate    = 0.1,
            scale_pos_weight = neg_pos,
            eval_metric      = "aucpr",
            verbosity        = 0,
            random_state     = args.seed,
        )
        stump_val_m, stump_test_m, stump_run_id = train_and_log(
            model_name        = "XGBoost_stump_depth1",
            model             = stump,
            X_train           = X_train_t,
            y_train           = y_train_np,
            X_val             = X_val_t,
            y_val             = y_val_np,
            X_test            = X_test_t,
            y_test            = y_test_np,
            feature_names     = feature_names,
            extra_params      = {
                "n_estimators": 100, "max_depth": 1, "learning_rate": 0.1,
                "scale_pos_weight": round(neg_pos, 2),
            },
            run_tags          = {**shared_tags, "model_type": "XGBoostClassifier_stump"},
            preproc_pipeline  = pipe,
        )
        gate_stump = promotion_gate(baseline_val_m, stump_val_m)
        print_gate_result(
            "LogisticRegression", "XGBoost_stump_depth1",
            baseline_val_m, stump_val_m, gate_stump,
        )
        with mlflow.start_run(run_id=stump_run_id):
            mlflow.set_tags({
                "gate_decision": gate_stump.decision,
                "gate_reason":   gate_stump.reason,
            })

    # -- Business reading on TEST set (final unbiased evaluation) -------------
    print("\n" + "-" * 62)
    print("  FINAL TEST-SET EVALUATION (held out, never seen during training)")
    print("-" * 62)

    lr_test_prob  = lr.predict_proba(X_test_t)[:, 1]
    xgb_test_prob = xgb.predict_proba(X_test_t)[:, 1]

    print_business_report(
        "LogisticRegression (baseline)", y_test_np, lr_test_prob,
        threshold=baseline_val_m.threshold,
    )
    print_business_report(
        "XGBoost (improved)", y_test_np, xgb_test_prob,
        threshold=improved_val_m.threshold,
    )

    if args.include_stump:
        stump_test_prob = stump.predict_proba(X_test_t)[:, 1]
        print_business_report(
            "XGBoost stump depth=1 (BLOCKED)", y_test_np, stump_test_prob,
            threshold=stump_val_m.threshold,
        )

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'-' * 62}")
    print(f"  SUMMARY")
    print(f"{'-' * 62}")
    print(f"  Baseline  run_id: {baseline_run_id}")
    print(f"  Improved  run_id: {improved_run_id}")
    print(f"  Gate decision  : {gate_main.decision}")
    print(f"  MLflow URI     : {MLFLOW_TRACKING_URI}")
    print(f"  View UI        : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print()

    sys.exit(0 if gate_main.decision == "PROMOTED" else 1)


if __name__ == "__main__":
    main()
