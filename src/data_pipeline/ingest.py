#!/usr/bin/env python3
"""
Data ingestion for Customer Intelligence Platform.

Downloads:
  1. UCI Bank Marketing dataset  (bank-additional-full.csv, ~41 k rows)
  2. CFPB Consumer Complaint sample (10 000 records)
     Priority: CFPB Elasticsearch API -> bulk CSV -> synthetic fallback

Then writes 500-row committed samples to data/samples/ for CI and quick
local testing -- no network required after the first run.

Usage:
    python -m src.data_pipeline.ingest
    python -m src.data_pipeline.ingest --force         # re-download even if files exist
    python -m src.data_pipeline.ingest --sample-only   # refresh samples from existing raw
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config -- every value is overridable via .env or the environment.
# Never hard-code paths in downstream modules; import from src.config instead.
# -----------------------------------------------------------------------------
UCI_ZIP_URL = os.getenv(
    "UCI_BANK_URL",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
)
# CFPB has changed its public API several times; we try multiple endpoints in order.
CFPB_API_URLS = [
    url.strip()
    for url in os.getenv(
        "CFPB_API_URLS",
        ",".join([
            "https://api.consumerfinance.gov/data/complaints.json",
            "https://api.consumerfinance.gov/data/complaints",
        ]),
    ).split(",")
    if url.strip()
]
# Bulk CSV fallback -- may be a large file; we stream and sample.
CFPB_BULK_CSV_URL = os.getenv(
    "CFPB_BULK_CSV_URL",
    "https://files.consumerfinance.gov/f/documents/cfpb_complaints.csv.zip",
)

_ROOT         = Path(__file__).resolve().parents[2]
_BASE         = Path(os.getenv("DATA_DIR",           _ROOT / "data"))
RAW_DIR       = Path(os.getenv("RAW_DATA_DIR",       _BASE / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", _BASE / "processed"))
SAMPLES_DIR   = Path(os.getenv("SAMPLES_DATA_DIR",   _BASE / "samples"))

CFPB_SAMPLE_SIZE     = int(os.getenv("CFPB_SAMPLE_SIZE", "10000"))
COMMITTED_SAMPLE_ROWS = 500  # rows written to data/samples/ and committed to git

# Canonical file locations
BANK_RAW_FILE    = RAW_DIR    / "bank_marketing" / "bank-additional-full.csv"
CFPB_RAW_FILE    = RAW_DIR    / "cfpb"           / "complaints.csv"
BANK_SAMPLE_FILE = SAMPLES_DIR / "bank_marketing_sample.csv"
CFPB_SAMPLE_FILE = SAMPLES_DIR / "cfpb_sample.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_dirs() -> None:
    for d in (
        RAW_DIR / "bank_marketing",
        RAW_DIR / "cfpb",
        PROCESSED_DIR,
        SAMPLES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _download_bytes(url: str, desc: str) -> bytes:
    """Stream-download *url* with a tqdm progress bar; return raw bytes."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in resp.iter_content(chunk_size=8_192):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


# -----------------------------------------------------------------------------
# UCI Bank Marketing
# -----------------------------------------------------------------------------
def ingest_bank_marketing(force: bool = False) -> pd.DataFrame:
    """Download and extract the UCI Bank Marketing full dataset."""
    if BANK_RAW_FILE.exists() and not force:
        log.info(
            "Bank Marketing already at %s -- skipping (--force to re-download)",
            BANK_RAW_FILE,
        )
        return pd.read_csv(BANK_RAW_FILE, sep=";")

    log.info("Downloading UCI Bank Marketing zip from %s", UCI_ZIP_URL)
    raw_bytes = _download_bytes(UCI_ZIP_URL, "bank-additional.zip")

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        # The zip nests inside a bank-additional/ subfolder.
        target = next(
            name for name in zf.namelist() if "bank-additional-full.csv" in name
        )
        content = zf.read(target).decode("utf-8")

    BANK_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    BANK_RAW_FILE.write_text(content, encoding="utf-8")
    log.info("Saved %d bytes -> %s", len(content), BANK_RAW_FILE)

    df = pd.read_csv(io.StringIO(content), sep=";")
    log.info("Bank Marketing shape: %s", df.shape)
    return df


# -----------------------------------------------------------------------------
# CFPB Consumer Complaints -- three-tier fallback strategy
#   1. Elasticsearch API (CFPB's public JSON endpoint)
#   2. Bulk CSV download (streaming, take first N rows)
#   3. Synthetic data generator (always works; clearly labelled)
# -----------------------------------------------------------------------------
def _cfpb_try_api(size: int) -> pd.DataFrame | None:
    """
    Attempt to fetch complaints from the CFPB Elasticsearch API.
    Returns a DataFrame or None if the API is unreachable / returns non-JSON.
    Tries each URL in CFPB_API_URLS in order.
    """
    page_size = min(size, 10_000)

    for base_url in CFPB_API_URLS:
        records: list[dict] = []
        from_idx = 0
        try:
            while len(records) < size:
                fetch_n = min(page_size, size - len(records))
                params: dict = {
                    "size": fetch_n,
                    "from_": from_idx,
                    "no_aggs": "true",
                    "format": "json",
                }
                log.info("CFPB API [%s]  from=%d  n=%d", base_url, from_idx, fetch_n)
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()

                # The API should return JSON; if it returns HTML the endpoint has moved
                ct = resp.headers.get("content-type", "")
                if "html" in ct:
                    log.warning("CFPB API at %s returned HTML -- endpoint may have changed", base_url)
                    break

                payload = resp.json()
                hits = payload.get("hits", {}).get("hits", [])
                if not hits:
                    break
                records.extend(h.get("_source", h) for h in hits)
                from_idx += len(hits)

            if records:
                return pd.DataFrame(records[:size])

        except (requests.RequestException, ValueError) as exc:
            log.warning("CFPB API [%s] failed: %s", base_url, exc)

    return None


def _cfpb_try_bulk_csv(size: int) -> pd.DataFrame | None:
    """
    Stream-download the CFPB bulk CSV zip and return the first *size* rows.
    Returns None if the URL is unavailable (returns non-200 or times out).
    """
    log.info("Trying CFPB bulk CSV download from %s", CFPB_BULK_CSV_URL)
    try:
        resp = requests.get(CFPB_BULK_CSV_URL, stream=True, timeout=30)
        if resp.status_code != 200:
            log.warning("CFPB bulk CSV returned %d", resp.status_code)
            return None
        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            log.warning("CFPB bulk CSV URL returned HTML -- link has moved")
            return None

        raw_bytes = _download_bytes(CFPB_BULK_CSV_URL, "cfpb_complaints.csv.zip")
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, nrows=size, low_memory=False)
        log.info("Bulk CSV rows loaded: %d", len(df))
        return df

    except (requests.RequestException, zipfile.BadZipFile, StopIteration, Exception) as exc:
        log.warning("CFPB bulk CSV failed: %s", exc)
        return None


def _cfpb_synthetic(size: int) -> pd.DataFrame:
    """
    Generate a synthetic CFPB-schema-compatible dataset.

    All values are drawn from known CFPB vocabulary; no real complaint data.
    Clearly marked with source='synthetic' so downstream code can detect it.
    """
    log.warning(
        "All CFPB real-data sources failed -- generating %d synthetic records. "
        "Mark source='synthetic' column will be present.",
        size,
    )
    rng = random.Random(42)

    products = [
        "Mortgage", "Credit card", "Student loan", "Debt collection",
        "Checking or savings account", "Vehicle loan or lease",
        "Credit reporting", "Payday loan",
    ]
    issues = [
        "Account information incorrect",
        "Trouble during payment process",
        "Incorrect information on your report",
        "Communication tactics",
        "Managing an account",
        "Problem with a purchase shown on your statement",
        "Can't repay your loan",
    ]
    responses = [
        "Closed with explanation",
        "Closed with monetary relief",
        "Closed with non-monetary relief",
        "Closed without relief",
        "In progress",
    ]
    states = [
        "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    ]
    submitted_via = ["Web", "Phone", "Referral", "Postal mail", "Fax", "Email"]

    rows = []
    for i in range(size):
        year = rng.randint(2019, 2024)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        rows.append({
            "date_received": f"{year}-{month:02d}-{day:02d}",
            "product": rng.choice(products),
            "sub_product": "unknown",
            "issue": rng.choice(issues),
            "sub_issue": "unknown",
            "company": f"Company_{rng.randint(1, 200)}",
            "state": rng.choice(states),
            "zip_code": f"{rng.randint(10000, 99999)}",
            "submitted_via": rng.choice(submitted_via),
            "company_response_to_consumer": rng.choice(responses),
            "timely_response": rng.choice(["Yes", "No"]),
            "consumer_disputed": rng.choice(["Yes", "No", "N/A"]),
            "complaint_id": 1_000_000 + i,
            "source": "synthetic",
        })
    return pd.DataFrame(rows)


def ingest_cfpb(force: bool = False) -> pd.DataFrame:
    """
    Fetch CFPB_SAMPLE_SIZE complaint records, trying three sources in order:
      API -> bulk CSV -> synthetic fallback.
    """
    if CFPB_RAW_FILE.exists() and not force:
        log.info("CFPB raw already at %s -- skipping", CFPB_RAW_FILE)
        return pd.read_csv(CFPB_RAW_FILE, low_memory=False)

    df: pd.DataFrame | None = None

    df = _cfpb_try_api(CFPB_SAMPLE_SIZE)
    if df is None:
        df = _cfpb_try_bulk_csv(CFPB_SAMPLE_SIZE)
    if df is None:
        df = _cfpb_synthetic(CFPB_SAMPLE_SIZE)

    # Strip narrative regardless of source (PII-adjacent; large)
    if "complaint_what_happened" in df.columns:
        df = df.drop(columns=["complaint_what_happened"])

    CFPB_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFPB_RAW_FILE, index=False)
    log.info("Saved CFPB -> %s  shape=%s", CFPB_RAW_FILE, df.shape)
    return df


# -----------------------------------------------------------------------------
# Committed samples (max COMMITTED_SAMPLE_ROWS, regenerated on every run)
# -----------------------------------------------------------------------------
def write_samples(bank_df: pd.DataFrame, cfpb_df: pd.DataFrame) -> None:
    """
    Write 500-row stratified (y) bank sample and 500-row CFPB sample to
    data/samples/. These files ARE committed to git for CI and demos.
    """
    n = min(COMMITTED_SAMPLE_ROWS, len(bank_df))
    # Stratify on target so the sample preserves the ~11 % positive rate
    # Stratified on target label so the sample preserves the ~11 % positive rate.
    pos = bank_df[bank_df["y"] == "yes"]
    neg = bank_df[bank_df["y"] == "no"]
    n_pos = max(1, round(n * len(pos) / len(bank_df)))
    n_neg = n - n_pos
    bank_sample = pd.concat([
        pos.sample(n=min(n_pos, len(pos)), random_state=42),
        neg.sample(n=min(n_neg, len(neg)), random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    cfpb_sample = cfpb_df.sample(
        n=min(COMMITTED_SAMPLE_ROWS, len(cfpb_df)), random_state=42
    )
    if "complaint_what_happened" in cfpb_sample.columns:
        cfpb_sample = cfpb_sample.drop(columns=["complaint_what_happened"])

    # Samples are written as standard comma-CSV (not semicolon like UCI raw)
    bank_sample.to_csv(BANK_SAMPLE_FILE, index=False)
    cfpb_sample.to_csv(CFPB_SAMPLE_FILE, index=False)
    log.info("Wrote %d-row bank sample -> %s", len(bank_sample), BANK_SAMPLE_FILE)
    log.info("Wrote %d-row CFPB sample -> %s", len(cfpb_sample), CFPB_SAMPLE_FILE)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download even if raw files already exist",
    )
    p.add_argument(
        "--sample-only", action="store_true",
        help="Skip network calls; just refresh committed samples from existing raw files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dirs()

    if args.sample_only:
        if not BANK_RAW_FILE.exists() or not CFPB_RAW_FILE.exists():
            log.error(
                "--sample-only requires existing raw files. "
                "Run without --sample-only first."
            )
            raise SystemExit(1)
        bank_df = pd.read_csv(BANK_RAW_FILE, sep=";")
        cfpb_df = pd.read_csv(CFPB_RAW_FILE, low_memory=False)
    else:
        bank_df = ingest_bank_marketing(force=args.force)
        cfpb_df = ingest_cfpb(force=args.force)

    write_samples(bank_df, cfpb_df)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
