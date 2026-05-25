"""
Unit tests for src/data_pipeline/validate.py

Strategy:
  - Build minimal valid DataFrames from a known-good fixture row.
  - Override one field at a time to trigger each business rule.
  - Verify that validate() returns False (not raises) on bad data.
  - The inject_bad_records() helper is tested separately to confirm it
    actually produces a failing DataFrame.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_pipeline.validate import (
    BANK_SCHEMA,
    VALID_JOBS,
    inject_bad_records,
    validate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _valid_row() -> dict:
    """One row that satisfies every rule in BANK_SCHEMA."""
    return {
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
        "duration": 300,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
        "y": "no",
    }


def _df(**overrides) -> pd.DataFrame:
    """Return a 1-row DataFrame using _valid_row() with *overrides* applied."""
    row = {**_valid_row(), **overrides}
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────
class TestValidSchema:
    def test_single_valid_row_passes(self):
        assert validate(_df()) is True

    def test_all_valid_jobs_accepted(self):
        rows = [{**_valid_row(), "job": job} for job in sorted(VALID_JOBS)]
        assert validate(pd.DataFrame(rows)) is True

    def test_target_yes_accepted(self):
        assert validate(_df(y="yes")) is True

    def test_pdays_zero_accepted(self):
        assert validate(_df(pdays=0)) is True


# ─────────────────────────────────────────────────────────────────────────────
# Business rules  (one parametrize per logical rule)
# ─────────────────────────────────────────────────────────────────────────────
class TestBusinessRules:

    @pytest.mark.parametrize("age", [0, 16, 99, 150, -1])
    def test_rule1_age_out_of_bounds(self, age):
        """Rule 1: age must be 17–98."""
        assert validate(_df(age=age)) is False

    @pytest.mark.parametrize("age", [17, 50, 98])
    def test_rule1_age_boundary_valid(self, age):
        assert validate(_df(age=age)) is True

    def test_rule2_invalid_job(self):
        """Rule 2: job must be in UCI controlled vocabulary."""
        assert validate(_df(job="hacker")) is False

    def test_rule3_invalid_marital(self):
        """Rule 3: marital must be divorced/married/single/unknown."""
        assert validate(_df(marital="it's complicated")) is False

    def test_rule4_invalid_education(self):
        """Rule 4: education must be one of the UCI categories."""
        assert validate(_df(education="phd")) is False

    def test_rule5_negative_duration(self):
        """Rule 5: call duration cannot be negative."""
        assert validate(_df(duration=-1)) is False

    def test_rule5_zero_duration_valid(self):
        """Duration of 0 is valid (call connected but 0 seconds recorded)."""
        assert validate(_df(duration=0)) is True

    def test_rule6_zero_campaign_contacts(self):
        """Rule 6: campaign must be ≥ 1 (record exists because a contact was made)."""
        assert validate(_df(campaign=0)) is False

    def test_rule7_pdays_below_minus_one(self):
        """Rule 7: pdays sentinel is -1; values < -1 are invalid."""
        assert validate(_df(pdays=-2)) is False

    def test_rule8_negative_previous_contacts(self):
        """Rule 8: previous contacts cannot be negative."""
        assert validate(_df(previous=-1)) is False

    @pytest.mark.parametrize("cpi", [91.9, 95.1, 200.0])
    def test_rule9_cpi_out_of_range(self, cpi):
        """Rule 9: cons.price.idx must be 92.0–95.0 (UCI data band)."""
        assert validate(_df(**{"cons.price.idx": cpi})) is False

    @pytest.mark.parametrize("ne", [3999.9, 6000.1, 100.0])
    def test_rule10_nr_employed_out_of_range(self, ne):
        """Rule 10: nr.employed must be 4000–6000."""
        assert validate(_df(**{"nr.employed": ne})) is False

    @pytest.mark.parametrize("target", ["maybe", "yes/no", "", "YES"])
    def test_rule11_invalid_target(self, target):
        """Rule 11: y must be exactly 'yes' or 'no'."""
        assert validate(_df(y=target)) is False


# ─────────────────────────────────────────────────────────────────────────────
# Null / missing checks
# ─────────────────────────────────────────────────────────────────────────────
class TestNullChecks:
    def test_null_age_fails(self):
        df = _df()
        df.loc[0, "age"] = None
        assert validate(df) is False

    def test_null_target_fails(self):
        df = _df()
        df.loc[0, "y"] = None
        assert validate(df) is False


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
class TestDuplicateDetection:
    def test_exact_duplicate_rows_caught(self):
        row = _valid_row()
        df = pd.DataFrame([row, row])
        assert validate(df) is False

    def test_near_duplicate_different_age_passes(self):
        row_a = _valid_row()
        row_b = {**row_a, "age": row_a["age"] + 1}
        assert validate(pd.DataFrame([row_a, row_b])) is True


# ─────────────────────────────────────────────────────────────────────────────
# inject_bad_records helper
# ─────────────────────────────────────────────────────────────────────────────
class TestInjectBadRecords:
    def test_inject_adds_exactly_one_row(self):
        df = _df()
        corrupted = inject_bad_records(df)
        assert len(corrupted) == len(df) + 1

    def test_injected_dataframe_fails_validation(self):
        """Core smoke test: bad records must be caught — not silently accepted."""
        corrupted = inject_bad_records(_df())
        assert validate(corrupted) is False

    def test_clean_dataframe_still_passes_before_injection(self):
        df = _df()
        assert validate(df) is True
        corrupted = inject_bad_records(df)
        assert validate(corrupted) is False
