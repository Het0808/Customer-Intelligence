#!/usr/bin/env python3
"""
Build a FAISS flat index over the CFPB complaint corpus.

Chunking strategy -- chunk_size=512, overlap=50:
  Why 512 tokens:
    all-MiniLM-L6-v2 defaults to max_seq_length=256, but the underlying
    MiniLM transformer supports 512 positions.  We override to 512 so each
    chunk can capture a full complaint narrative (median CFPB narrative is
    ~350 tokens).  Chunks smaller than the narrative would split one person's
    complaint across vectors, losing causal context (e.g. "they charged me
    again" requires the preceding "I cancelled my account" to make sense).
  Why 50-token overlap:
    ~10% of chunk size.  Enough to preserve one sentence of boundary context
    so a query matching the tail of chunk N also matches the head of chunk N+1.
    Larger overlaps (>20%) inflate corpus size and add near-duplicate vectors
    that bias retrieval toward longer complaints.

Embedding model: all-MiniLM-L6-v2 (384-dim, CPU, no API key)
  Strong zero-shot semantic similarity for complaint-domain English.
  Runs in <5 s on 10k chunks on a modern laptop CPU.

FAISS index: IndexFlatIP (exact cosine via normalised inner product)
  Exact search -- no approximation error -- critical for a compliance context
  where a missed relevant complaint could be a liability.
  10k x 384 x 4 bytes = ~15 MB, fits in RAM without batching.

Reproducibility:
  Records are sorted by complaint_id before chunking.  Embedding is
  deterministic (sentence-transformers runs in eval mode, no dropout).
  Same CFPB CSV -> identical index.bin and docstore.json every run.

Usage:
    python -m src.rag.build_index                # full 10k corpus
    python -m src.rag.build_index --sample       # 500-row sample (fast demo)
    python -m src.rag.build_index --force        # rebuild even if index exists
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# sentence_transformers (torch) must be imported before faiss on Windows --
# faiss loads MKL DLLs that conflict with torch's DLL loader if torch goes second.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import RAW_DIR, SAMPLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE     = 512   # tokens -- see module docstring
CHUNK_OVERLAP  = 50    # tokens -- see module docstring
INDEX_DIR      = Path(os.getenv("FAISS_INDEX_DIR", str(_ROOT / "faiss_index")))

CFPB_RAW_FILE    = RAW_DIR    / "cfpb" / "complaints.csv"
CFPB_SAMPLE_FILE = SAMPLES_DIR / "cfpb_sample.csv"


# ── Text construction ─────────────────────────────────────────────────────────
def build_complaint_text(row: pd.Series) -> str:
    """
    Concatenate structured CFPB fields into a single passage.
    Real data may include a 'complaint_what_happened' narrative; synthetic
    data does not, so this function degrades gracefully.
    """
    parts = [
        f"Product: {row.get('product', '')}",
        f"Issue: {row.get('issue', '')}",
        f"Company: {row.get('company', '')}",
    ]
    state = str(row.get("state", ""))
    if state and state not in ("", "nan", "unknown"):
        parts.append(f"State: {state}")

    response = str(row.get("company_response_to_consumer", ""))
    if response and response not in ("", "nan", "unknown"):
        parts.append(f"Company response: {response}")

    disputed = str(row.get("consumer_disputed", ""))
    if disputed and disputed not in ("", "nan", "N/A", "unknown"):
        parts.append(f"Consumer disputed: {disputed}")

    narrative = str(row.get("complaint_what_happened", ""))
    if narrative and narrative not in ("", "nan"):
        parts.append(f"Narrative: {narrative}")

    return "  |  ".join(parts)


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_tokens(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap:    int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping token-window chunks using the model tokenizer.
    Returns the full text as a single chunk if shorter than chunk_size.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        end    = min(start + chunk_size, len(ids))
        pieces = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        chunks.append(pieces.strip())
        if end == len(ids):
            break
        start += step
    return chunks


# ── Index building ────────────────────────────────────────────────────────────
def build_index(df: pd.DataFrame) -> tuple[faiss.Index, list[dict]]:
    """
    Chunk, embed, and index all complaints.
    Returns (faiss_index, docstore) where docstore[i] maps to vector i in index.
    """
    log.info("Loading embedding model '%s' ...", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = CHUNK_SIZE   # override default 256 to match chunk_size
    tokenizer = model.tokenizer

    # Sort for deterministic indexing -- same input -> same chunk ordering
    df = df.sort_values("complaint_id").reset_index(drop=True)

    log.info("Chunking %d complaints (chunk=%d, overlap=%d) ...",
             len(df), CHUNK_SIZE, CHUNK_OVERLAP)
    docstore: list[dict] = []
    texts:    list[str]  = []

    for _, row in df.iterrows():
        complaint_text = build_complaint_text(row)
        for chunk_text in chunk_tokens(complaint_text, tokenizer):
            docstore.append({
                "chunk_id":    len(docstore),
                "complaint_id": str(row.get("complaint_id", "")),
                "product":     str(row.get("product",  "")),
                "issue":       str(row.get("issue",    "")),
                "company":     str(row.get("company",  "")),
                "date":        str(row.get("date_received", "")),
                "source":      str(row.get("source", "real")),
                "chunk_text":  chunk_text,
            })
            texts.append(chunk_text)

    log.info("Embedding %d chunks (batch_size=256) ...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit norm -> inner product == cosine sim
        convert_to_numpy=True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS IndexFlatIP built: %d vectors, dim=%d", index.ntotal, dim)
    return index, docstore


def save_index(index: faiss.Index, docstore: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.bin"))
    with open(out_dir / "docstore.json", "w", encoding="utf-8") as fh:
        json.dump(docstore, fh, indent=2, ensure_ascii=False)
    log.info("Saved -> %s/index.bin  (%d vectors)", out_dir, index.ntotal)
    log.info("Saved -> %s/docstore.json  (%d entries)", out_dir, len(docstore))


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample", action="store_true",
                   help="Use 500-row CFPB sample (no download needed, fast demo)")
    p.add_argument("--force",  action="store_true",
                   help="Rebuild even if index already exists")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if (INDEX_DIR / "index.bin").exists() and not args.force:
        log.info(
            "Index already exists at %s -- skipping. Pass --force to rebuild.",
            INDEX_DIR,
        )
        return

    src_path = CFPB_SAMPLE_FILE if args.sample else CFPB_RAW_FILE
    if not src_path.exists():
        log.error("CFPB data not found at %s. Run: python -m src.data_pipeline.ingest", src_path)
        raise SystemExit(1)

    df = pd.read_csv(src_path, low_memory=False)
    log.info("Loaded %d rows from %s", len(df), src_path)

    index, docstore = build_index(df)
    save_index(index, docstore, INDEX_DIR)
    log.info("Done. %d chunks indexed from %d complaints.", len(docstore), len(df))


if __name__ == "__main__":
    main()
