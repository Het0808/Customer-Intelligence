"""
Unit tests for src/serving/schemas.py Pydantic v2 models.

Covers:
  TestCustomerFeaturesValidation  -- happy path, missing fields, range violations
  TestComplaintQueryValidation    -- empty/missing question, optional filters
  TestComplaintAnswerModel        -- output model construction
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    ComplaintAnswer,
    ComplaintQuery,
    CustomerFeatures,
)

# ---------------------------------------------------------------------------
# Shared fixture -- a fully valid CustomerFeatures payload
# ---------------------------------------------------------------------------
VALID_PAYLOAD: dict = {
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
    "duration": 180,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.299,
    "nr.employed": 5099.1,
}


# ---------------------------------------------------------------------------
# CustomerFeatures
# ---------------------------------------------------------------------------
class TestCustomerFeaturesValidation:

    def test_valid_payload_passes(self):
        """A fully populated valid payload must deserialise without error."""
        obj = CustomerFeatures(**VALID_PAYLOAD)
        assert obj.age == 35
        assert obj.emp_var_rate == pytest.approx(-1.8)

    def test_missing_required_field_raises(self):
        """Omitting a required field (age) must raise ValidationError."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_age_below_minimum_raises(self):
        """age < 17 (UCI lower bound) must be rejected."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "age": 5})

    def test_age_above_maximum_raises(self):
        """age > 98 (UCI upper bound) must be rejected."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "age": 150})

    def test_invalid_default_value_raises(self):
        """default must match ^(yes|no|unknown)$ pattern."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "default": "maybe"})

    def test_negative_duration_raises(self):
        """Call duration cannot be negative."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "duration": -1})

    def test_campaign_zero_raises(self):
        """campaign >= 1 because the record exists because a contact was made."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "campaign": 0})

    def test_emp_var_rate_out_of_range_raises(self):
        """emp.var.rate outside [-5, 3] (UCI band) must be rejected."""
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "emp.var.rate": 99.0})

    def test_to_raw_dict_uses_dot_notation_keys(self):
        """to_raw_dict() must restore UCI dot-notation for the feature pipeline."""
        obj = CustomerFeatures(**VALID_PAYLOAD)
        raw = obj.to_raw_dict()
        assert "emp.var.rate"   in raw
        assert "cons.price.idx" in raw
        assert "cons.conf.idx"  in raw
        assert "nr.employed"    in raw
        assert raw["age"] == 35

    def test_to_raw_dict_round_trips_numeric_fields(self):
        """Numeric aliases must survive the Python→dict round-trip exactly."""
        obj = CustomerFeatures(**VALID_PAYLOAD)
        raw = obj.to_raw_dict()
        assert raw["emp.var.rate"] == pytest.approx(VALID_PAYLOAD["emp.var.rate"])
        assert raw["euribor3m"]    == pytest.approx(VALID_PAYLOAD["euribor3m"])


# ---------------------------------------------------------------------------
# ComplaintQuery
# ---------------------------------------------------------------------------
class TestComplaintQueryValidation:

    def test_valid_question_passes(self):
        q = ComplaintQuery(question="What mortgage problems exist?")
        assert q.question == "What mortgage problems exist?"
        assert q.product is None

    def test_empty_question_raises(self):
        """An empty string must fail the min_length=1 constraint."""
        with pytest.raises(ValidationError):
            ComplaintQuery(question="")

    def test_missing_question_raises(self):
        """question is required -- missing it must raise ValidationError."""
        with pytest.raises(ValidationError):
            ComplaintQuery()  # type: ignore[call-arg]

    def test_optional_filters_are_none_by_default(self):
        q = ComplaintQuery(question="test query")
        assert q.product   is None
        assert q.company   is None
        assert q.date_from is None
        assert q.issue     is None

    def test_all_optional_filters_accepted(self):
        q = ComplaintQuery(
            question  = "payment issue",
            product   = "Mortgage",
            company   = "Wells Fargo",
            issue     = "Payment",
            date_from = "2023-01-01",
        )
        assert q.product   == "Mortgage"
        assert q.company   == "Wells Fargo"
        assert q.date_from == "2023-01-01"


# ---------------------------------------------------------------------------
# ComplaintAnswer
# ---------------------------------------------------------------------------
class TestComplaintAnswerModel:

    def test_sufficient_answer_construction(self):
        ans = ComplaintAnswer(
            answer               = "Customers report X.",
            retrieved_ids        = ["1001", "1002"],
            evidence_sufficiency = "sufficient",
            prompt_version       = "v1.0",
            refusal              = False,
        )
        assert ans.refusal is False
        assert len(ans.retrieved_ids) == 2

    def test_refusal_answer_allows_null_answer(self):
        """When refusal=True the answer field must be allowed to be None."""
        ans = ComplaintAnswer(
            answer               = None,
            retrieved_ids        = [],
            evidence_sufficiency = "insufficient",
            prompt_version       = "v1.0",
            refusal              = True,
        )
        assert ans.answer is None
        assert ans.retrieved_ids == []

    def test_refusal_always_has_empty_retrieved_ids(self):
        ans = ComplaintAnswer(
            answer=None, retrieved_ids=[],
            evidence_sufficiency="insufficient",
            prompt_version="v1.0", refusal=True,
        )
        assert ans.retrieved_ids == []
        assert ans.refusal is True
