"""
Feature engineering for the UCI Bank Marketing dataset.

Design contract:
  • Every pure function is side-effect-free and returns a new DataFrame.
  • Stateful steps (StandardScaler, OneHotEncoder) live inside the sklearn
    Pipeline returned by build_preprocessing_pipeline().  That pipeline is
    fit ONLY on the training split and is the single transform artefact
    saved to MLflow -- guaranteeing zero train-serve skew.
  • The same pipeline that transforms X_train is loaded at serving time.

Column handling after pure transforms:
  SCALE_COLS  (20) -- all numeric + encoded ordinals + business features
  OHE_COLS    (3)  -- job, marital, contact   (nominal, unknown-safe)
  Dropped           -- poutcome  (signal captured by `prev_success`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Constants -- sourced from UCI bank-additional-names.txt
# -----------------------------------------------------------------------------
PDAYS_NOT_CONTACTED = 999   # sentinel for "never contacted in prior campaign"

EDUCATION_MAP: dict[str, float] = {
    "illiterate":         0.0,
    "basic.4y":           1.0,
    "basic.6y":           2.0,
    "basic.9y":           3.0,
    "high.school":        4.0,
    "professional.course":5.0,
    "university.degree":  6.0,
    "unknown":           -1.0,   # median-imputed by downstream scaler
}
MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2,  "mar": 3,  "apr": 4,
    "may": 5, "jun": 6,  "jul": 7,  "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DAY_MAP:  dict[str, int] = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
YESNO_MAP: dict[str, int] = {"yes": 1, "no": 0, "unknown": -1}
TARGET_MAP: dict[str, int] = {"yes": 1, "no": 0}

# Columns passed to StandardScaler (after pure transforms)
SCALE_COLS: list[str] = [
    # original numerics
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    # ordinals encoded to int
    "education", "month", "day_of_week",
    # binary flags encoded to int
    "default", "housing", "loan",
    # business features (see add_business_features)
    "was_previously_contacted", "prev_success", "log_contact_recency", "age_segment",
]

# Columns passed to OneHotEncoder
OHE_COLS: list[str] = ["job", "marital", "contact"]

# poutcome is dropped after prev_success is extracted (see encode_categoricals)


# -----------------------------------------------------------------------------
# Pure feature functions  (no state -- safe to call without fitting)
# -----------------------------------------------------------------------------
def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four business-motivated derived columns.

    was_previously_contacted  -- binary: was the client ever called in a prior
        campaign?  Clients with prior contact are 4x more likely to subscribe.

    prev_success  -- binary: did the *prior* campaign result in subscription?
        The strongest single predictor (poutcome='success' -> 65% conversion).

    log_contact_recency  -- log(1 + pdays) when pdays != 999, else 0.
        Captures diminishing returns of recency: difference between 3 and 6
        days matters more than between 90 and 93.  Zero for never-contacted
        clients keeps the two sub-populations clearly separated.

    age_segment  -- 0 (young, <30), 1 (middle, 30–60), 2 (senior, >60).
        Banks' campaign ROI differs by life stage; bucketing avoids the
        model treating age as linear when the relationship is U-shaped.
    """
    out = df.copy()
    out["was_previously_contacted"] = (out["pdays"] != PDAYS_NOT_CONTACTED).astype(np.float32)
    out["prev_success"] = (out["poutcome"] == "success").astype(np.float32)
    out["log_contact_recency"] = np.where(
        out["pdays"] != PDAYS_NOT_CONTACTED,
        np.log1p(out["pdays"].clip(upper=PDAYS_NOT_CONTACTED - 1).astype(float)),
        0.0,
    ).astype(np.float32)
    out["age_segment"] = (
        pd.cut(out["age"], bins=[0, 30, 60, 120], labels=[0, 1, 2], right=False)
        .astype(float)
        .fillna(1.0)                # middle-age default for any out-of-range ages
        .astype(np.float32)
    )
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinal and binary categorical columns in-place.
    Nominal columns (job, marital, contact) are left as strings for the
    downstream OneHotEncoder -- this keeps the encoding self-describing.
    Drops poutcome after prev_success has captured its signal.
    """
    out = df.copy()

    # Ordinal: education has a natural attainment ladder
    out["education"] = out["education"].map(EDUCATION_MAP).fillna(-1.0)

    # Cyclic-but-ordinal: month and day retain their linear order for tree models
    # (sin/cos would suit linear models; ordinal integer is better for XGBoost)
    out["month"]       = out["month"].map(MONTH_MAP)
    out["day_of_week"] = out["day_of_week"].map(DAY_MAP)

    # Binary yes/no/unknown  -> 1/0/−1
    for col in ("default", "housing", "loan"):
        out[col] = out[col].map(YESNO_MAP)

    # Drop poutcome -- its information is now in prev_success
    out = out.drop(columns=["poutcome"])
    return out


def encode_target(y: pd.Series) -> pd.Series:
    """Map 'yes'/'no' target to 1/0."""
    return y.map(TARGET_MAP).astype(int)


def pure_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: applies add_business_features then encode_categoricals.
    Called by the FunctionTransformer step inside the sklearn Pipeline so both
    training and serving go through exactly the same code path.
    """
    return encode_categoricals(add_business_features(df))


# -----------------------------------------------------------------------------
# Sklearn transformer wrapper (enables Pipeline serialisation)
# -----------------------------------------------------------------------------
class PureTransformer(BaseEstimator, TransformerMixin):
    """
    Stateless sklearn-compatible wrapper around pure_transform().
    No parameters are learned; fit() is a no-op.
    Included in the Pipeline so the entire transform chain is one saveable
    object and the pure steps are not applied twice.
    """
    def fit(self, X: pd.DataFrame, y=None) -> "PureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pure_transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(SCALE_COLS + OHE_COLS, dtype=object)


# -----------------------------------------------------------------------------
# Stateful pipeline  (fit once on training data, reuse everywhere)
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline() -> Pipeline:
    """
    Return an **unfitted** sklearn Pipeline:

      1. PureTransformer  -- pure_transform() (no state)
      2. ColumnTransformer:
           scale  -> StandardScaler on SCALE_COLS
           ohe    -> OneHotEncoder  on OHE_COLS  (handle_unknown='ignore')

    Call pipeline.fit(X_train) to fit scaler + encoder on training data only.
    Then pipeline.transform(X) for any split or live request.

    Why StandardScaler on binary/ordinal columns:
      Logistic Regression is sensitive to feature scale; scaling everything
      gives a consistent gradient landscape.  XGBoost is invariant to scale,
      so the overhead is negligible for the tree model.
    """
    ct = ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                SCALE_COLS,
            ),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",           # avoids perfect multicollinearity
                    sparse_output=False,
                    dtype=np.float32,
                ),
                OHE_COLS,
            ),
        ],
        remainder="drop",   # safety net -- no unnamed columns sneak through
    )
    return Pipeline(
        steps=[
            ("pure", PureTransformer()),
            ("transform", ct),
        ]
    )


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Return human-readable feature names after the fitted pipeline transforms."""
    ct = fitted_pipeline.named_steps["transform"]
    return list(ct.get_feature_names_out())
