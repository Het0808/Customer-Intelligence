#!/usr/bin/env python3
"""
Data validation for Customer Intelligence Platform — Bank Marketing dataset.

Enforces:
  • Column presence and dtype
  • Null / missing value constraints (nullable=False on every column)
  • Duplicate row detection (DataFrame-level check)
  • 11 domain-specific business rules (age bounds, controlled vocabularies,
    non-negative counters, economic indicator ranges, binary target)

Exit codes:
  0 — validation passed
  1 — one or more checks failed  (also used for config/IO errors)

Usage:
    python -m src.data_pipeline.validate              # validate raw data
    python -m src.data_pipeline.validate --sample     # validate committed 500-row sample
    python -m src.data_pipeline.validate --inject-bad # inject a bad record first (demo/CI)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_ROOT         = Path(__file__).resolve().parents[2]
_BASE         = Path(os.getenv("DATA_DIR",           _ROOT / "data"))
RAW_DIR       = Path(os.getenv("RAW_DATA_DIR",       _BASE / "raw"))
SAMPLES_DIR   = Path(os.getenv("SAMPLES_DATA_DIR",   _BASE / "samples"))

BANK_RAW_FILE    = RAW_DIR    / "bank_marketing" / "bank-additional-full.csv"
BANK_SAMPLE_FILE = SAMPLES_DIR / "bank_marketing_sample.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Controlled vocabularies  (sourced from UCI bank-additional-names.txt)
# ─────────────────────────────────────────────────────────────────────────────
VALID_JOBS = frozenset({
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown",
})
VALID_MARITAL = frozenset({"divorced", "married", "single", "unknown"})
VALID_EDUCATION = frozenset({
    "basic.4y", "basic.6y", "basic.9y", "high.school",
    "illiterate", "professional.course", "university.degree", "unknown",
})
VALID_CONTACT    = frozenset({"cellular", "telephone"})
VALID_MONTHS     = frozenset({"jan", "feb", "mar", "apr", "may", "jun",
                               "jul", "aug", "sep", "oct", "nov", "dec"})
VALID_DAYS       = frozenset({"mon", "tue", "wed", "thu", "fri"})
VALID_POUTCOME   = frozenset({"failure", "nonexistent", "success"})
VALID_YESNO_UNK  = frozenset({"no", "yes", "unknown"})
VALID_TARGET     = frozenset({"no", "yes"})

# ─────────────────────────────────────────────────────────────────────────────
# Pandera schema
#
# coerce=True: convert column dtypes before checking so that CSV int columns
# read as float64 (due to missing-value upcasting) still pass type checks.
# ─────────────────────────────────────────────────────────────────────────────
BANK_SCHEMA = DataFrameSchema(
    coerce=True,
    columns={
        # ── Demographics ──────────────────────────────────────────────────────
        "age": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(17),   # Rule 1a — youngest in UCI data
                Check.less_than_or_equal_to(98),       # Rule 1b — oldest in UCI data
            ],
            nullable=False,
            description="Client age in years",
        ),
        "job": Column(
            str,
            checks=Check.isin(VALID_JOBS),             # Rule 2 — UCI controlled vocab
            nullable=False,
        ),
        "marital": Column(
            str,
            checks=Check.isin(VALID_MARITAL),          # Rule 3
            nullable=False,
        ),
        "education": Column(
            str,
            checks=Check.isin(VALID_EDUCATION),        # Rule 4
            nullable=False,
        ),
        "default": Column(str, checks=Check.isin(VALID_YESNO_UNK), nullable=False),
        "housing": Column(str, checks=Check.isin(VALID_YESNO_UNK), nullable=False),
        "loan":    Column(str, checks=Check.isin(VALID_YESNO_UNK), nullable=False),

        # ── Contact campaign ──────────────────────────────────────────────────
        "contact":      Column(str, checks=Check.isin(VALID_CONTACT), nullable=False),
        "month":        Column(str, checks=Check.isin(VALID_MONTHS),  nullable=False),
        "day_of_week":  Column(str, checks=Check.isin(VALID_DAYS),    nullable=False),
        "duration": Column(
            int,
            checks=Check.greater_than_or_equal_to(0), # Rule 5 — call duration ≥ 0 s
            nullable=False,
            description="Last contact duration in seconds",
        ),
        "campaign": Column(
            int,
            checks=Check.greater_than_or_equal_to(1), # Rule 6 — at least 1 contact made
            nullable=False,
            description="Number of contacts during this campaign",
        ),
        "pdays": Column(
            int,
            checks=Check.greater_than_or_equal_to(-1),# Rule 7 — -1 = not previously contacted
            nullable=False,
            description="Days since last contact from previous campaign (-1 = never)",
        ),
        "previous": Column(
            int,
            checks=Check.greater_than_or_equal_to(0), # Rule 8 — non-negative prior contacts
            nullable=False,
        ),
        "poutcome": Column(str, checks=Check.isin(VALID_POUTCOME), nullable=False),

        # ── Economic indicators ───────────────────────────────────────────────
        "emp.var.rate":   Column(float, nullable=False),
        "cons.price.idx": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(92.0), # Rule 9a — realistic CPI band
                Check.less_than_or_equal_to(95.0),    # Rule 9b   (UCI range: 92.2–94.8)
            ],
            nullable=False,
        ),
        "cons.conf.idx":  Column(float, nullable=False),
        "euribor3m":      Column(float, nullable=False),
        "nr.employed": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(4000.0), # Rule 10a — realistic NE band
                Check.less_than_or_equal_to(6000.0),    # Rule 10b  (UCI range: 4963–5228)
            ],
            nullable=False,
        ),

        # ── Target ────────────────────────────────────────────────────────────
        "y": Column(
            str,
            checks=Check.isin(VALID_TARGET),           # Rule 11 — binary, no 'unknown'
            nullable=False,
            description="Campaign subscription outcome",
        ),
    },
    checks=[
        # DataFrame-level: zero duplicate rows
        Check(
            lambda df: not df.duplicated().any(),
            error="Duplicate rows detected — dedup before training",
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_bank_data(use_sample: bool = False) -> pd.DataFrame:
    """
    Load the Bank Marketing dataset.

    Raw file uses ';' separator (UCI convention).
    Committed sample is written as standard comma-CSV by ingest.py.
    Falls back to the committed sample if raw is absent and use_sample=False.
    """
    if use_sample:
        path, sep = BANK_SAMPLE_FILE, ","
    else:
        path, sep = BANK_RAW_FILE, ";"

    if not path.exists():
        if not use_sample and BANK_SAMPLE_FILE.exists():
            log.warning(
                "Raw file not found at %s; falling back to committed sample.", BANK_RAW_FILE
            )
            return pd.read_csv(BANK_SAMPLE_FILE)
        log.error(
            "Data file not found: %s\n"
            "Run  python -m src.data_pipeline.ingest  first.",
            path,
        )
        raise SystemExit(1)

    return pd.read_csv(path, sep=sep)


# ─────────────────────────────────────────────────────────────────────────────
# Bad-record injection (demo + CI smoke test)
# ─────────────────────────────────────────────────────────────────────────────
def inject_bad_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append one deliberately invalid row to *df*.

    Violations injected:
      • age = 150        → Rule 1b (> 98)
      • job = "hacker"   → Rule 2 (not in vocabulary)
      • campaign = 0     → Rule 6 (must be ≥ 1)
      • y = "maybe"      → Rule 11 (not in {yes, no})
    """
    bad_row = {
        "age": 150,           # violates Rule 1b
        "job": "hacker",      # violates Rule 2
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 300,
        "campaign": 0,        # violates Rule 6
        "pdays": -1,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
        "y": "maybe",         # violates Rule 11
    }
    log.warning(
        "Injecting 1 bad row with violations: age=150, job='hacker', "
        "campaign=0, y='maybe'"
    )
    return pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Validation runner
# ─────────────────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> bool:
    """
    Run all checks against *df*.
    Returns True on pass, False on failure.
    Prints a human-readable failure_cases table when checks fail.
    """
    log.info("Validating %d rows × %d columns …", *df.shape)

    # Pre-flight null report (gives cleaner context than Pandera's default message)
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        log.warning("Null values before schema check:\n%s", nulls.to_string())

    try:
        # lazy=True collects ALL failures instead of stopping at the first
        BANK_SCHEMA.validate(df, lazy=True)
        log.info("✓  All checks passed.")
        return True
    except pa.errors.SchemaErrors as exc:
        n = len(exc.failure_cases)
        log.error("✗  Validation FAILED — %d failure case(s):", n)
        print(exc.failure_cases.to_string(index=False))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sample", action="store_true",
        help="Validate the committed 500-row sample (no download needed)",
    )
    p.add_argument(
        "--inject-bad", action="store_true",
        help="Append an intentionally invalid record before validating (schema smoke test)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_bank_data(use_sample=args.sample)
    log.info("Loaded %d rows from %s", len(df), "sample" if args.sample else "raw")

    if args.inject_bad:
        df = inject_bad_records(df)

    ok = validate(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
