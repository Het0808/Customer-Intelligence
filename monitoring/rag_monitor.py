#!/usr/bin/env python3
"""
RAG monitoring metrics for the Customer Intelligence Platform.

Runs the 10 EVAL_CASES from src/rag/rag_eval.py against the live FAISS index,
tracking per-query retrieval performance.  Saves a JSON summary to:
  monitoring/reports/rag_metrics.json

Metrics:
  n_queries             -- total queries run (always 10)
  retrieval_hit_rate    -- fraction of cases where >= 1 expected ID was found
                           (refusal cases count as hits when refusal is correct)
  empty_retrieval_count -- queries that returned zero chunks
  avg_top1_score        -- mean cosine similarity of the top-ranked chunk
  refusal_rate          -- fraction of queries that triggered insufficient_evidence
  avg_retrieved_tokens  -- mean estimated token count across all retrieved chunks
                           (word count * 1.3 approximation)
  avg_latency_ms        -- mean per-query retrieval latency in milliseconds

Usage:
    python monitoring/rag_monitor.py
    python monitoring/rag_monitor.py --index-dir faiss_index
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.rag.retrieve import INDEX_DIR, load_retriever, retrieve
from src.rag.rag_eval import EVAL_CASES, _load_docstore, _lookup_expected_ids

# ── Constants ─────────────────────────────────────────────────────────────────
REPORT_DIR   = Path(__file__).resolve().parent / "reports"
METRICS_PATH = REPORT_DIR / "rag_metrics.json"

# Rough token estimate: average English word ≈ 1.3 tokens (GPT tokeniser heuristic)
_WORDS_TO_TOKENS = 1.3


# ── Helpers ───────────────────────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return round(len(text.split()) * _WORDS_TO_TOKENS)


# ── Runner ────────────────────────────────────────────────────────────────────
def run_rag_monitor(index_dir: Path = INDEX_DIR) -> dict:
    """
    Execute all EVAL_CASES and compute aggregate retrieval metrics.

    Returns the metrics dict.  Also writes the dict to METRICS_PATH as JSON.
    """
    if not (index_dir / "index.bin").exists():
        print(f"ERROR: FAISS index not found at {index_dir}", file=sys.stderr)
        print("Build it first:  python -m src.rag.build_index", file=sys.stderr)
        sys.exit(1)

    load_retriever(index_dir)
    docstore = _load_docstore(index_dir)

    hit_count     = 0
    empty_count   = 0
    refusal_count = 0
    top1_scores:  list[float] = []
    token_counts: list[int]   = []
    latencies_ms: list[float] = []

    print(f"\nRunning {len(EVAL_CASES)} eval queries against {index_dir} ...\n")

    for case in EVAL_CASES:
        expected_ids = _lookup_expected_ids(docstore, case.filter_criteria, case.n_expected)

        t0         = time.perf_counter()
        result     = retrieve(case.question, filters=case.filter_criteria, top_k=5)
        latency_ms = (time.perf_counter() - t0) * 1_000
        latencies_ms.append(latency_ms)

        if result.insufficient_evidence:
            refusal_count += 1
            empty_count   += 1
            # A correct refusal (EC-10) counts as a hit
            if not case.pass_if_contains:
                hit_count += 1
        else:
            if not result.chunks:
                empty_count += 1
            else:
                top1_scores.append(result.chunks[0].score)
                for chunk in result.chunks:
                    token_counts.append(_estimate_tokens(chunk.chunk_text))

                retrieved_ids = {c.complaint_id for c in result.chunks}
                if not expected_ids or (set(expected_ids) & retrieved_ids):
                    hit_count += 1

        status = "[PASS]" if (
            (result.insufficient_evidence and not case.pass_if_contains) or
            (not result.insufficient_evidence and case.pass_if_contains)
        ) else "[FAIL]"
        print(f"  {case.id} {status}  {case.description}  ({latency_ms:.0f} ms)")

    n = len(EVAL_CASES)
    metrics = {
        "n_queries":             n,
        "retrieval_hit_rate":    round(hit_count / n, 4),
        "empty_retrieval_count": empty_count,
        "avg_top1_score":        round(sum(top1_scores) / len(top1_scores), 4)
                                  if top1_scores else 0.0,
        "refusal_rate":          round(refusal_count / n, 4),
        "avg_retrieved_tokens":  round(sum(token_counts) / len(token_counts), 1)
                                  if token_counts else 0.0,
        "avg_latency_ms":        round(sum(latencies_ms) / n, 2),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nRAG metrics saved -> {METRICS_PATH}")

    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_metrics_summary(metrics: dict) -> None:
    w = 52
    print(f"\n{'=' * w}")
    print("  RAG Monitor -- Customer Intelligence Platform")
    print(f"{'=' * w}")
    rows = [
        ("Queries run",             metrics["n_queries"]),
        ("Retrieval hit rate",      f"{metrics['retrieval_hit_rate']:.1%}"),
        ("Empty retrievals",        metrics["empty_retrieval_count"]),
        ("Refusal rate",            f"{metrics['refusal_rate']:.1%}"),
        ("Avg top-1 cosine score",  f"{metrics['avg_top1_score']:.4f}"),
        ("Avg retrieved tokens",    f"{metrics['avg_retrieved_tokens']:.0f}"),
        ("Avg latency (ms)",        f"{metrics['avg_latency_ms']:.1f}"),
    ]
    for label, value in rows:
        print(f"  {label:<28}  {value}")
    print(f"{'=' * w}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", type=Path, default=INDEX_DIR,
                   help=f"FAISS index directory (default: {INDEX_DIR})")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    metrics = run_rag_monitor(index_dir=args.index_dir)
    print_metrics_summary(metrics)


if __name__ == "__main__":
    main()
