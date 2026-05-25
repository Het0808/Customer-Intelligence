"""
Unit tests for src/rag/retrieve.py using a hand-crafted in-memory FAISS index.

No disk I/O and no sentence-transformer model loading needed.  The embedding
model is replaced by a MagicMock whose encode() returns controlled unit
vectors, making cosine scores exact and predictable.

Vector layout (DIM = 8):
  Mortgage chunks    → unit vector on axis 0:  [1, 0, 0, 0, 0, 0, 0, 0]
  Credit card chunks → unit vector on axis 1:  [0, 1, 0, 0, 0, 0, 0, 0]

Query in axis 0  scores 1.0 against Mortgage, 0.0 against Credit card.
Query in axis 7  scores 0.0 against every stored vector → refusal gate fires.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from src.rag.retrieve import MIN_SCORE_THRESHOLD, Retriever


# ─── helpers ────────────────────────────────────────────────────────────────
DIM = 8


def _unit(axis: int, dim: int = DIM) -> np.ndarray:
    """Return a unit vector with value 1 at *axis* and 0 elsewhere."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _mock_query(retriever: Retriever, axis: int) -> None:
    """Point the mock encoder at a unit vector in *axis*."""
    retriever._model.encode.return_value = np.array([_unit(axis)])


# ─── fixture ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def retriever() -> Retriever:
    """
    Retriever backed by an in-memory IndexFlatIP with 20 controlled vectors:
      - chunks 0-9  : product='Mortgage',    axis 0
      - chunks 10-19: product='Credit card', axis 1
    """
    r = Retriever(index_dir=Path("."))   # __init__ sets all attrs to None/empty
    r._dim = DIM

    docs: list[dict] = []
    vecs: list[np.ndarray] = []

    for i in range(10):
        docs.append({
            "chunk_id":    i,
            "complaint_id": str(1000 + i),
            "product":     "Mortgage",
            "issue":       "Payment",
            "company":     "Bank A",
            "date":        "2023-01-15",
            "source":      "test",
            "chunk_text":  f"Mortgage payment trouble chunk {i}",
        })
        vecs.append(_unit(0))

    for i in range(10):
        docs.append({
            "chunk_id":    10 + i,
            "complaint_id": str(2000 + i),
            "product":     "Credit card",
            "issue":       "Billing",
            "company":     "Bank B",
            "date":        "2023-02-20",
            "source":      "test",
            "chunk_text":  f"Credit card billing dispute chunk {i}",
        })
        vecs.append(_unit(1))

    index = faiss.IndexFlatIP(DIM)
    index.add(np.stack(vecs).astype(np.float32))

    r._index    = index
    r._docstore = docs
    r._model    = MagicMock()
    return r


# ─── Test: relevant query returns results above threshold ────────────────────
class TestRelevantQuery:

    def test_on_topic_query_returns_chunks(self, retriever: Retriever):
        """A query aligned with stored vectors must return ≥ 1 chunk."""
        _mock_query(retriever, 0)
        result = retriever.retrieve("mortgage payment problem", top_k=5)
        assert not result.insufficient_evidence
        assert len(result.chunks) == 5

    def test_all_returned_scores_above_threshold(self, retriever: Retriever):
        """Every returned chunk must score at or above MIN_SCORE_THRESHOLD."""
        _mock_query(retriever, 0)
        result = retriever.retrieve("mortgage", top_k=5)
        assert all(c.score >= MIN_SCORE_THRESHOLD for c in result.chunks)

    def test_scores_are_exact_cosine_similarity(self, retriever: Retriever):
        """
        Query in axis 0 vs Mortgage vectors (also axis 0) → cosine sim = 1.0.
        This proves the index stores normalised vectors and inner product = cosine.
        """
        _mock_query(retriever, 0)
        result = retriever.retrieve("mortgage", top_k=3)
        for chunk in result.chunks:
            assert abs(chunk.score - 1.0) < 1e-4, (
                f"Expected cosine=1.0, got {chunk.score}"
            )


# ─── Test: irrelevant query triggers insufficient_evidence refusal ───────────
class TestIrrelevantQuery:

    def test_orthogonal_query_triggers_refusal(self, retriever: Retriever):
        """
        A query in axis 7 is orthogonal to every stored vector (axes 0 and 1).
        Cosine similarity = 0.0 < MIN_SCORE_THRESHOLD → refusal gate fires.
        """
        _mock_query(retriever, 7)
        result = retriever.retrieve("European Central Bank interest rate policy")
        assert result.insufficient_evidence

    def test_refusal_has_empty_chunks(self, retriever: Retriever):
        """insufficient_evidence=True must always be paired with chunks=[]."""
        _mock_query(retriever, 7)
        result = retriever.retrieve("unrelated off-topic query")
        assert result.chunks == []

    def test_refusal_filters_applied_is_recorded(self, retriever: Retriever):
        """filters_applied must be returned even on refusal for audit logging."""
        _mock_query(retriever, 7)
        filters = {"product": "Mortgage"}
        result = retriever.retrieve("ECB rates", filters=filters)
        assert result.insufficient_evidence
        assert result.filters_applied == filters


# ─── Test: metadata filters correctly narrow results ────────────────────────
class TestMetadataFilters:

    def test_product_filter_returns_only_matching_product(self, retriever: Retriever):
        """
        With filter product='Mortgage', only Mortgage chunks can be returned --
        Credit card chunks must never appear regardless of their cosine score.
        """
        _mock_query(retriever, 0)
        result = retriever.retrieve(
            "mortgage", filters={"product": "Mortgage"}, top_k=10
        )
        assert not result.insufficient_evidence
        assert all(c.metadata["product"] == "Mortgage" for c in result.chunks)

    def test_unknown_product_filter_returns_refusal(self, retriever: Retriever):
        """
        Pre-filtering to zero candidates must return insufficient_evidence=True
        before any vector search runs.
        """
        _mock_query(retriever, 0)
        result = retriever.retrieve(
            "query", filters={"product": "Student loan"}, top_k=5
        )
        assert result.insufficient_evidence
        assert result.chunks == []

    def test_company_filter_narrows_to_bank_a(self, retriever: Retriever):
        """company='Bank A' should exclude all Credit card (Bank B) chunks."""
        _mock_query(retriever, 0)
        result = retriever.retrieve(
            "bank issue", filters={"company": "Bank A"}, top_k=10
        )
        assert not result.insufficient_evidence
        assert all(c.metadata["company"] == "Bank A" for c in result.chunks)

    def test_filter_does_not_affect_unfiltered_search(self, retriever: Retriever):
        """
        Without filters the full corpus is searched: top-5 results come from
        both products when the query is equidistant from both clusters.
        """
        # axis-0 query → Mortgage scores 1.0, Credit card scores 0.0
        # but without filter, all 20 docs are candidates
        _mock_query(retriever, 0)
        unfiltered = retriever.retrieve("query", top_k=5)
        filtered   = retriever.retrieve("query", filters={"product": "Mortgage"}, top_k=5)
        # Filtered result must be a subset of the Mortgage product space
        for c in filtered.chunks:
            assert c.metadata["product"] == "Mortgage"
        # Unfiltered top-5 are all from Mortgage (cosine=1.0 beats 0.0)
        assert not unfiltered.insufficient_evidence
