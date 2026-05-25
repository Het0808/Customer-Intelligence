"""
Retrieval module for the CFPB complaint intelligence RAG pipeline.

Design:
  Pre-filtering is applied to the docstore BEFORE any vector search so the
  cosine scoring only runs over chunks that are contextually in-scope.  This
  matters for compliance: a mortgage question must not be answered with credit
  card evidence, even if those chunks happened to score higher after a broad
  search.

  Implementation uses IndexFlatIP.reconstruct(i) to extract vectors for the
  filtered candidate set, builds a temporary mini-index, and runs exact cosine
  search over only those vectors.  This is O(n_candidates * d) per query --
  cheap for n <= 15k.

Refusal threshold -- MIN_SCORE_THRESHOLD = 0.35:
  Cosine similarity distribution on this corpus:
    - Truly unrelated pairs (e.g. "ECB rate" vs a mortgage complaint): 0.10-0.25
    - Tangentially related (wrong product, same domain): 0.26-0.34
    - On-topic (right product + right issue):             0.50-0.90
  0.35 sits at the inflection between "tangential" and "relevant".
  It is intentionally conservative: a false refusal is safer than a
  hallucinated answer in a financial compliance context.
  Override via environment variable RAG_MIN_SCORE.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# sentence_transformers (torch) must be imported before faiss on Windows --
# faiss loads MKL DLLs that conflict with torch's DLL loader if torch goes second.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

# ── Configurable constants ────────────────────────────────────────────────────
MIN_SCORE_THRESHOLD: float = float(os.getenv("RAG_MIN_SCORE", "0.35"))
INDEX_DIR:  Path = Path(os.getenv("FAISS_INDEX_DIR", str(_ROOT / "faiss_index")))
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id:     int
    complaint_id: str
    score:        float
    chunk_text:   str
    metadata:     dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id":     self.chunk_id,
            "complaint_id": self.complaint_id,
            "score":        self.score,
            "chunk_text":   self.chunk_text,
            "metadata":     self.metadata,
        }


@dataclass
class RetrievalResult:
    chunks:                list[RetrievedChunk]
    insufficient_evidence: bool
    query:                 str
    filters_applied:       dict


# ── Retriever ─────────────────────────────────────────────────────────────────
class Retriever:
    """
    Holds the loaded FAISS index and docstore.
    Pre-filters by metadata, then runs cosine search on the candidate subset.
    """

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self._index:    faiss.Index | None = None
        self._docstore: list[dict]         = []
        self._model                        = None
        self._index_dir                    = index_dir
        self._dim:      int                = 0

    # -- Lifecycle -------------------------------------------------------------
    def load(self) -> None:
        idx_path = self._index_dir / "index.bin"
        ds_path  = self._index_dir / "docstore.json"
        if not idx_path.exists() or not ds_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_dir}. "
                "Build it first:  python -m src.rag.build_index"
            )

        self._index = faiss.read_index(str(idx_path))
        self._dim   = self._index.d

        with open(ds_path, encoding="utf-8") as fh:
            self._docstore = json.load(fh)

        log.info("FAISS index loaded: %d vectors dim=%d", self._index.ntotal, self._dim)

        self._model = SentenceTransformer(EMBED_MODEL)
        self._model.max_seq_length = 512
        log.info("Retriever ready (embedding model: %s)", EMBED_MODEL)

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        return len(self._docstore)

    # -- Filter logic ----------------------------------------------------------
    def _matches(self, doc: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        if filters.get("product") and \
                doc.get("product", "").lower() != filters["product"].lower():
            return False
        if filters.get("company") and \
                doc.get("company", "").lower() != filters["company"].lower():
            return False
        if filters.get("issue") and \
                doc.get("issue", "").lower() != filters["issue"].lower():
            return False
        if filters.get("date_from"):
            doc_date = doc.get("date", "")
            # ISO date strings compare lexicographically for YYYY-MM-DD format
            if doc_date and doc_date < filters["date_from"]:
                return False
        return True

    # -- Main retrieval --------------------------------------------------------
    def retrieve(
        self,
        query:   str,
        filters: dict | None = None,
        top_k:   int = 5,
    ) -> RetrievalResult:
        """
        Stage 1 -- Pre-filter docstore by metadata (product / company / issue / date).
        Stage 2 -- Embed query with the same model used at index time.
        Stage 3 -- Exact cosine search over the filtered candidate subset.
        Stage 4 -- Refusal gate: if max(scores) < MIN_SCORE_THRESHOLD, return
                   empty list with insufficient_evidence=True.

        When filters=None, searches the full index directly (no reconstruction
        overhead) rather than rebuilding a temporary mini-index.
        """
        if not self.is_ready:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        # ── Stage 1: pre-filter ───────────────────────────────────────────────
        if filters:
            candidate_idx = [
                i for i, doc in enumerate(self._docstore)
                if self._matches(doc, filters)
            ]
            if not candidate_idx:
                log.warning("Zero candidates after pre-filtering: %s", filters)
                return RetrievalResult(
                    chunks                = [],
                    insufficient_evidence = True,
                    query                 = query,
                    filters_applied       = filters,
                )
        else:
            candidate_idx = None  # sentinel: search full index

        # ── Stage 2: embed query ──────────────────────────────────────────────
        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # ── Stage 3: cosine search ────────────────────────────────────────────
        if candidate_idx is None:
            # No filter -- search full index directly (fast path)
            k      = min(top_k, self._index.ntotal)
            D, I   = self._index.search(q_vec, k)
            scores = D[0]
            local_idxs = I[0]
            # local_idxs are already original docstore indices for the full index
            orig_map   = {j: j for j in range(len(self._docstore))}
        else:
            # Pre-filtered -- reconstruct candidate vectors and build mini-index
            cand_vecs = np.array(
                [self._index.reconstruct(i) for i in candidate_idx],
                dtype=np.float32,
            )
            mini   = faiss.IndexFlatIP(self._dim)
            mini.add(cand_vecs)

            k      = min(top_k, len(candidate_idx))
            D, I   = mini.search(q_vec, k)
            scores     = D[0]
            local_idxs = I[0]
            orig_map   = {local: orig for local, orig in enumerate(candidate_idx)}

        # ── Stage 4: refusal gate ─────────────────────────────────────────────
        valid = local_idxs != -1
        if not valid.any():
            return RetrievalResult(
                chunks=[], insufficient_evidence=True,
                query=query, filters_applied=filters or {},
            )

        max_score = float(scores[valid].max())
        if max_score < MIN_SCORE_THRESHOLD:
            log.info(
                "Refusal: max_score=%.4f < threshold=%.2f  query='%s'",
                max_score, MIN_SCORE_THRESHOLD, query[:80],
            )
            return RetrievalResult(
                chunks=[], insufficient_evidence=True,
                query=query, filters_applied=filters or {},
            )

        # ── Assemble results ──────────────────────────────────────────────────
        chunks: list[RetrievedChunk] = []
        for local_i, score in zip(local_idxs[valid], scores[valid]):
            orig_i = orig_map[int(local_i)]
            doc    = self._docstore[orig_i]
            chunks.append(RetrievedChunk(
                chunk_id     = doc["chunk_id"],
                complaint_id = doc["complaint_id"],
                score        = round(float(score), 6),
                chunk_text   = doc["chunk_text"],
                metadata     = {k: doc.get(k, "") for k in
                                ("product", "issue", "company", "date", "source")},
            ))

        log.info(
            "Retrieved %d chunks (max_score=%.4f) for query='%s'",
            len(chunks), max_score, query[:80],
        )
        return RetrievalResult(
            chunks=chunks, insufficient_evidence=False,
            query=query, filters_applied=filters or {},
        )


# ── Module-level singleton ────────────────────────────────────────────────────
_retriever: Retriever | None = None


def load_retriever(index_dir: Path = INDEX_DIR) -> Retriever:
    """
    Create and load the singleton Retriever.  Called once at startup.
    The global _retriever is only assigned AFTER load() succeeds, so a
    FileNotFoundError leaves _retriever = None and get_retriever() raises
    RuntimeError -> 503 on the first request.
    """
    global _retriever
    r = Retriever(index_dir)
    r.load()              # raises FileNotFoundError if index absent
    _retriever = r        # only reached if load() succeeded
    return r


def get_retriever() -> Retriever:
    if _retriever is None:
        raise RuntimeError(
            "Retriever not initialised. Call load_retriever() first."
        )
    return _retriever


def retrieve(
    query:   str,
    filters: dict | None = None,
    top_k:   int = 5,
) -> RetrievalResult:
    """Module-level convenience wrapper (used by answer.py and rag_eval.py)."""
    return get_retriever().retrieve(query, filters=filters, top_k=top_k)
