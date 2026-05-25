"""
Pydantic v2 request/response models for the Customer Intelligence API.

Field naming convention:
  Request fields use underscores (Python-friendly).
  to_raw_dict() maps them back to UCI dot-notation for the feature pipeline.

Threshold bands (applied to predict_proba output):
  >= 0.6  -> "high"   (strong signal; contact immediately)
  0.3-0.6 -> "medium" (worth considering; queue for follow-up)
  < 0.3   -> "low"    (unlikely converter; skip unless low-cost channel)
  Rationale: 0.6 is ~2x the model's optimal F1 threshold (~0.3-0.5), ensuring
  high-band contacts have meaningful precision lift before committing call-centre
  resources. 0.3 guards against discarding borderline positives entirely.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Input schema
# -----------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Demographics
    age:        int   = Field(..., ge=17, le=98)
    job:        str
    marital:    str
    education:  str

    # Financial flags
    default:    str   = Field(..., pattern="^(yes|no|unknown)$")
    housing:    str   = Field(..., pattern="^(yes|no|unknown)$")
    loan:       str   = Field(..., pattern="^(yes|no|unknown)$")

    # Last contact
    contact:     str
    month:       str
    day_of_week: str
    duration:    int  = Field(..., ge=0)

    # Campaign context
    campaign:  int   = Field(..., ge=1)
    pdays:     int   = Field(..., ge=-1)
    previous:  int   = Field(..., ge=0)
    poutcome:  str

    # Economic indicators -- dot-notation names as JSON aliases
    emp_var_rate:    float = Field(..., alias="emp.var.rate",    ge=-5.0,  le=3.0)
    cons_price_idx:  float = Field(..., alias="cons.price.idx",  ge=90.0,  le=96.0)
    cons_conf_idx:   float = Field(..., alias="cons.conf.idx",   ge=-60.0, le=-20.0)
    euribor3m:       float = Field(..., ge=0.0,  le=6.0)
    nr_employed:     float = Field(..., alias="nr.employed",     ge=4000.0, le=6000.0)

    def to_raw_dict(self) -> dict:
        """Return a dict keyed by UCI column names (dot-notation) for the feature pipeline."""
        return {
            "age":           self.age,
            "job":           self.job,
            "marital":       self.marital,
            "education":     self.education,
            "default":       self.default,
            "housing":       self.housing,
            "loan":          self.loan,
            "contact":       self.contact,
            "month":         self.month,
            "day_of_week":   self.day_of_week,
            "duration":      self.duration,
            "campaign":      self.campaign,
            "pdays":         self.pdays,
            "previous":      self.previous,
            "poutcome":      self.poutcome,
            "emp.var.rate":  self.emp_var_rate,
            "cons.price.idx":self.cons_price_idx,
            "cons.conf.idx": self.cons_conf_idx,
            "euribor3m":     self.euribor3m,
            "nr.employed":   self.nr_employed,
        }


# -----------------------------------------------------------------------------
# Output schemas
# -----------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    probability:        float
    threshold_decision: Literal["high", "medium", "low"]
    model_version:      str
    run_id:             str
    latency_ms:         float
    request_id:         str


class BatchRecord(BaseModel):
    """Single row result in a batch response -- may carry an error instead of a prediction."""
    probability:        float | None                          = None
    threshold_decision: Literal["high", "medium", "low"] | None = None
    model_version:      str
    run_id:             str
    latency_ms:         float | None                          = None
    request_id:         str
    error:              str | None                            = None


class BatchScoreRequest(BaseModel):
    records: list[CustomerFeatures]


class BatchScoreResponse(BaseModel):
    results: list[BatchRecord]
    total:   int
    errors:  int


# -----------------------------------------------------------------------------
# RAG complaint intelligence schemas
# -----------------------------------------------------------------------------
class ComplaintQuery(BaseModel):
    """Input for POST /ask-complaints."""
    question:  str  = Field(..., min_length=1, description="Natural-language question about complaints")
    product:   str | None = Field(None, description="Filter by CFPB product category")
    company:   str | None = Field(None, description="Filter by company name (exact match)")
    date_from: str | None = Field(None, description="Earliest complaint date (YYYY-MM-DD)")
    issue:     str | None = Field(None, description="Filter by complaint issue type")


class ComplaintAnswer(BaseModel):
    """Output for POST /ask-complaints."""
    answer:               str | None       # None when refusal=True
    retrieved_ids:        list[str]        # complaint IDs used as evidence; [] on refusal
    evidence_sufficiency: str              # "sufficient" | "insufficient"
    prompt_version:       str
    refusal:              bool
