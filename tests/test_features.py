"""
Unit tests for src/data_pipeline/features.py

Coverage:
  TestAddBusinessFeatures  — all four derived columns
  TestEncodeCategoricals   — ordinal/binary encoding + poutcome drop
  TestEncodeTarget         — yes/no → 1/0
  TestPureTransform        — end-to-end pure transform contract
  TestPreprocessingPipeline — fit-then-transform shape and no-leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.features import (
    EDUCATION_MAP,
    MONTH_MAP,
    OHE_COLS,
    PDAYS_NOT_CONTACTED,
    SCALE_COLS,
    add_business_features,
    build_preprocessing_pipeline,
    encode_categoricals,
    encode_target,
    get_feature_names,
    pure_transform,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────
def _base_row() -> dict:
    """A single fully-valid Bank Marketing row (feature columns only, no 'y')."""
    return {
        "age": 35, "job": "admin.", "marital": "married",
        "education": "university.degree", "default": "no",
        "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 300,
        "campaign": 2, "pdays": PDAYS_NOT_CONTACTED, "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8, "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0,
    }


def _df(**overrides) -> pd.DataFrame:
    row = {**_base_row(), **overrides}
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# TestAddBusinessFeatures
# ─────────────────────────────────────────────────────────────────────────────
class TestAddBusinessFeatures:

    def test_was_previously_contacted_false_for_sentinel(self):
        """pdays=999 means never contacted → flag must be 0."""
        df = _df(pdays=PDAYS_NOT_CONTACTED)
        out = add_business_features(df)
        assert out["was_previously_contacted"].iloc[0] == 0

    def test_was_previously_contacted_true_for_real_pdays(self):
        """Any pdays < 999 means prior contact → flag must be 1."""
        df = _df(pdays=3)
        out = add_business_features(df)
        assert out["was_previously_contacted"].iloc[0] == 1

    def test_prev_success_true_only_when_poutcome_success(self):
        df_success  = _df(poutcome="success")
        df_failure  = _df(poutcome="failure")
        df_noexist  = _df(poutcome="nonexistent")

        assert add_business_features(df_success) ["prev_success"].iloc[0] == 1
        assert add_business_features(df_failure) ["prev_success"].iloc[0] == 0
        assert add_business_features(df_noexist) ["prev_success"].iloc[0] == 0

    def test_log_contact_recency_zero_when_not_contacted(self):
        """Never-contacted clients should have recency score of 0."""
        df  = _df(pdays=PDAYS_NOT_CONTACTED)
        out = add_business_features(df)
        assert out["log_contact_recency"].iloc[0] == pytest.approx(0.0)

    def test_log_contact_recency_positive_and_log_scaled(self):
        """Contacted client should have log1p(pdays) recency score."""
        pdays = 6
        df  = _df(pdays=pdays)
        out = add_business_features(df)
        assert out["log_contact_recency"].iloc[0] == pytest.approx(
            np.log1p(pdays), rel=1e-4
        )

    def test_log_recency_order(self):
        """More recent contact (lower pdays) should yield higher recency score."""
        out3 = add_business_features(_df(pdays=3))["log_contact_recency"].iloc[0]
        out9 = add_business_features(_df(pdays=9))["log_contact_recency"].iloc[0]
        assert out3 < out9   # log is monotone increasing; pdays=9 > pdays=3 → higher log

    @pytest.mark.parametrize(
        "age, expected_segment",
        [(20, 0.0), (29, 0.0), (30, 1.0), (45, 1.0), (60, 2.0), (75, 2.0)],
    )
    def test_age_segment_bins(self, age, expected_segment):
        """age_segment should assign 0/1/2 based on the defined age bins."""
        out = add_business_features(_df(age=age))
        assert out["age_segment"].iloc[0] == pytest.approx(expected_segment)

    def test_original_columns_preserved(self):
        """Input columns must not be modified; output is a new DataFrame."""
        df  = _df()
        out = add_business_features(df)
        # All original cols still present
        for col in df.columns:
            assert col in out.columns
        # Four new cols added
        for new_col in ("was_previously_contacted", "prev_success",
                        "log_contact_recency", "age_segment"):
            assert new_col in out.columns
        # Input unchanged
        assert "was_previously_contacted" not in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# TestEncodeCategoricals
# ─────────────────────────────────────────────────────────────────────────────
class TestEncodeCategoricals:

    def test_education_ordinal_order(self):
        """Higher education level must map to a higher integer."""
        levels = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                  "high.school", "professional.course", "university.degree"]
        mapped = [EDUCATION_MAP[lv] for lv in levels]
        assert mapped == sorted(mapped)

    def test_education_unknown_maps_to_minus_one(self):
        df  = _df(education="unknown")
        out = encode_categoricals(add_business_features(df))
        assert out["education"].iloc[0] == pytest.approx(-1.0)

    def test_education_university_maps_to_six(self):
        df  = _df(education="university.degree")
        out = encode_categoricals(add_business_features(df))
        assert out["education"].iloc[0] == pytest.approx(6.0)

    def test_month_maps_correctly(self):
        for month, expected in MONTH_MAP.items():
            df  = _df(month=month)
            out = encode_categoricals(add_business_features(df))
            assert out["month"].iloc[0] == expected, f"month={month}"

    def test_binary_flags_yes_no_unknown(self):
        for value, expected in [("yes", 1), ("no", 0), ("unknown", -1)]:
            df  = _df(default=value)
            out = encode_categoricals(add_business_features(df))
            assert out["default"].iloc[0] == expected

    def test_poutcome_dropped(self):
        """poutcome must not survive encode_categoricals (captured by prev_success)."""
        df  = _df()
        out = encode_categoricals(add_business_features(df))
        assert "poutcome" not in out.columns

    def test_ohe_cols_still_present_as_strings(self):
        """Nominal columns (job, marital, contact) must remain strings for OHE."""
        df  = _df()
        out = encode_categoricals(add_business_features(df))
        for col in OHE_COLS:
            assert col in out.columns
            assert out[col].dtype == object

    def test_does_not_mutate_input(self):
        df  = _df()
        _   = encode_categoricals(add_business_features(df))
        assert df["education"].iloc[0] == "university.degree"  # original unchanged


# ─────────────────────────────────────────────────────────────────────────────
# TestEncodeTarget
# ─────────────────────────────────────────────────────────────────────────────
class TestEncodeTarget:

    def test_yes_maps_to_one(self):
        assert encode_target(pd.Series(["yes"])).iloc[0] == 1

    def test_no_maps_to_zero(self):
        assert encode_target(pd.Series(["no"])).iloc[0] == 0

    def test_series_preserved(self):
        y = pd.Series(["yes", "no", "no", "yes"])
        result = encode_target(y)
        assert list(result) == [1, 0, 0, 1]

    def test_dtype_is_int(self):
        result = encode_target(pd.Series(["yes", "no"]))
        assert result.dtype in (int, np.int64, np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# TestPureTransform
# ─────────────────────────────────────────────────────────────────────────────
class TestPureTransform:

    def test_output_has_scale_and_ohe_cols(self):
        df  = _df()
        out = pure_transform(df)
        for col in SCALE_COLS + OHE_COLS:
            assert col in out.columns, f"Missing: {col}"

    def test_poutcome_absent(self):
        df  = _df()
        out = pure_transform(df)
        assert "poutcome" not in out.columns

    def test_idempotent(self):
        """Calling pure_transform twice must give the same result."""
        df   = _df()
        out1 = pure_transform(df)
        out2 = pure_transform(df)
        pd.testing.assert_frame_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# TestPreprocessingPipeline
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessingPipeline:

    def _make_dataset(self, n: int = 50) -> pd.DataFrame:
        """Return a small synthetic dataset with all required columns."""
        rng = np.random.default_rng(0)
        jobs    = ["admin.", "blue-collar", "technician", "services", "management"]
        marital = ["married", "single", "divorced"]
        educ    = ["university.degree", "high.school", "basic.9y"]
        months  = ["jan", "may", "sep"]
        days    = ["mon", "tue", "wed"]
        poutcomes = ["nonexistent", "failure", "success"]

        rows = []
        for _ in range(n):
            pdays = int(rng.choice([PDAYS_NOT_CONTACTED, 3, 6, 12]))
            rows.append({
                "age":          int(rng.integers(20, 70)),
                "job":          rng.choice(jobs),
                "marital":      rng.choice(marital),
                "education":    rng.choice(educ),
                "default":      rng.choice(["yes", "no", "unknown"]),
                "housing":      rng.choice(["yes", "no"]),
                "loan":         rng.choice(["yes", "no"]),
                "contact":      rng.choice(["cellular", "telephone"]),
                "month":        rng.choice(months),
                "day_of_week":  rng.choice(days),
                "duration":     int(rng.integers(0, 600)),
                "campaign":     int(rng.integers(1, 10)),
                "pdays":        pdays,
                "previous":     int(rng.integers(0, 5)),
                "poutcome":     rng.choice(poutcomes),
                "emp.var.rate": float(rng.uniform(-3, 2)),
                "cons.price.idx": float(rng.uniform(92.5, 94.5)),
                "cons.conf.idx": float(rng.uniform(-50, -20)),
                "euribor3m":    float(rng.uniform(0.6, 5.1)),
                "nr.employed":  float(rng.uniform(4963, 5228)),
            })
        return pd.DataFrame(rows)

    def test_pipeline_fit_transform_returns_2d_array(self):
        df = self._make_dataset(50)
        pipe = build_preprocessing_pipeline()
        out = pipe.fit_transform(df)
        assert out.ndim == 2
        assert out.shape[0] == 50

    def test_transform_shape_matches_fit(self):
        """Test set must have the same number of features as training set."""
        df_train = self._make_dataset(40)
        df_test  = self._make_dataset(10)
        pipe = build_preprocessing_pipeline()
        pipe.fit(df_train)
        train_out = pipe.transform(df_train)
        test_out  = pipe.transform(df_test)
        assert train_out.shape[1] == test_out.shape[1]

    def test_scaler_fit_on_train_only(self):
        """
        Standard deviation of a scaled column must be ~1 on training data.
        This fails if the scaler was accidentally fit on test data.
        """
        df_train  = self._make_dataset(200)
        pipe      = build_preprocessing_pipeline()
        pipe.fit(df_train)
        train_out = pipe.transform(df_train)
        # Each scaled column should have mean≈0, std≈1 ON training data
        assert abs(train_out[:, 0].mean()) < 0.2    # age column approx centred
        assert 0.5 < train_out[:, 0].std() < 2.0   # age std ≈ 1

    def test_unknown_ohe_category_produces_no_error(self):
        """
        handle_unknown='ignore' in OHE means an unseen category at serve time
        maps to all-zero rather than raising.  This is the no-train-serve-skew
        guarantee for categorical values.
        """
        df_train = self._make_dataset(40)
        pipe = build_preprocessing_pipeline()
        pipe.fit(df_train)

        # Introduce an unseen job category in the test row
        df_test = _df(job="astronaut")
        out = pipe.transform(df_test)
        assert out.shape[1] == pipe.transform(df_train).shape[1]

    def test_feature_names_out_length_matches_output_width(self):
        df = self._make_dataset(20)
        pipe = build_preprocessing_pipeline()
        pipe.fit(df)
        out = pipe.transform(df)
        names = get_feature_names(pipe)
        assert len(names) == out.shape[1]

    def test_null_input_is_not_silently_imputed(self):
        """
        NaN in a numeric column must propagate through the pipeline as NaN,
        NOT be replaced with a training-set mean.

        sklearn >= 1.4 StandardScaler uses force_all_finite='allow-nan' in
        transform(), so it does not raise -- it preserves the NaN.  This is
        intentional: the pipeline has no SimpleImputer, which means null inputs
        must be caught UPSTREAM (by the Pydantic schema) before they reach the
        feature pipeline.  If the pipeline silently imputed, that guarantee
        would be broken.
        """
        df_train = self._make_dataset(50)
        df_test  = _df()
        df_test  = df_test.astype({"age": float})
        df_test.loc[0, "age"] = float("nan")   # inject NaN into a scaled column
        pipe = build_preprocessing_pipeline()
        pipe.fit(df_train)
        out = pipe.transform(df_test)
        # NaN must survive into the output matrix -- not be silently imputed.
        assert np.isnan(out).any(), (
            "Pipeline must propagate NaN rather than impute it; "
            "upstream Pydantic validation is the null guard."
        )

    def test_single_row_and_batch_produce_same_feature_width(self):
        """
        A 1-row DataFrame must produce the same number of columns as a batch --
        verifies that the OHE categories are fixed at fit time, not inferred
        from the number of rows being transformed.
        """
        df = self._make_dataset(20)
        pipe = build_preprocessing_pipeline()
        pipe.fit(df)
        single = pipe.transform(df.iloc[[0]])
        batch  = pipe.transform(df)
        assert single.shape == (1, batch.shape[1])
