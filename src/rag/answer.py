"""
Four-stage RAG answer chain for CFPB complaint intelligence queries.

Stages (each is an explicit function -- no implicit ordering):
  1. retrieve      -- fetch relevant chunks via retrieve.py
  2. build_prompt  -- format evidence + question into a versioned prompt template
  3. generate      -- call LLM (OpenAI or Ollama) with the constructed prompt
  4. log           -- structured log of every interaction for audit / monitoring

Hard constraint:
  If stage 1 returns insufficient_evidence=True, this module returns a
  structured refusal object IMMEDIATELY.  Stage 3 (LLM call) is NEVER
  reached.  This is enforced in code, not in comments.

LLM swapping via environment variables:
  LLM_PROVIDER   = "openai"  (default) | "ollama"
  LLM_MODEL      = "gpt-4o-mini" (default for openai) | "llama3.2" (ollama)
  LLM_BASE_URL   = "" (uses provider default) | "http://localhost:11434/v1"
  OPENAI_API_KEY = sk-...  (required when LLM_PROVIDER=openai)

Both providers are accessed via the openai Python package.  Ollama exposes
an OpenAI-compatible /v1 endpoint, so the same client object handles both.

Prompt version: PROMPT_VERSION = "v1.0"
  Bump this whenever the template changes so logged interactions remain
  attributable to the exact prompt that generated them.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.rag.retrieve import RetrievalResult, RetrievedChunk, retrieve

log = logging.getLogger(__name__)

# ── Prompt version (bump when template changes) ───────────────────────────────
PROMPT_VERSION = "v1.0"

# ── LLM configuration (all overridable via env) ───────────────────────────────
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
_LLM_MODEL    = os.getenv(
    "LLM_MODEL",
    "llama3.2" if _LLM_PROVIDER == "ollama" else "gpt-4o-mini",
)
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
_LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "30"))
_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "512"))

REFUSAL_MESSAGE = (
    "I cannot answer this question because the complaint corpus does not contain "
    "sufficient evidence.  Please try a more specific question or adjust the "
    "product/company/issue filters."
)

# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class AnswerResult:
    answer:               str | None
    retrieved_ids:        list[str]
    evidence_sufficiency: str            # "sufficient" | "insufficient"
    prompt_version:       str
    refusal:              bool
    model_name:           str
    latency_ms:           float
    token_count:          int
    generation_succeeded: bool
    request_id:           str           = field(default_factory=lambda: str(uuid.uuid4()))


# ── Stage 1: retrieve ─────────────────────────────────────────────────────────
def _stage_retrieve(
    question: str,
    filters:  dict | None,
    top_k:    int,
) -> RetrievalResult:
    return retrieve(question, filters=filters, top_k=top_k)


# ── Stage 2: build_prompt ─────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
You are a financial services compliance analyst reviewing CFPB consumer complaints.

Your task: answer the QUESTION below using ONLY the evidence provided.
- Cite each evidence item inline as [E1], [E2], [E3] etc. when you use it.
- Do NOT use any knowledge beyond the evidence below.
- If the evidence does not support a definitive answer, say so explicitly.
- Be concise and factual.

EVIDENCE:
{evidence_block}

QUESTION: {question}

ANSWER (cite [E1], [E2] ... as appropriate):"""


def _stage_build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = (
            f"[E{i}] Complaint {chunk.complaint_id} "
            f"(Product: {meta.get('product','')} | "
            f"Issue: {meta.get('issue','')} | "
            f"Company: {meta.get('company','')} | "
            f"Score: {chunk.score:.3f})"
        )
        lines.append(header)
        lines.append(f"     {chunk.chunk_text}")
        lines.append("")

    evidence_block = "\n".join(lines).rstrip()
    return _PROMPT_TEMPLATE.format(evidence_block=evidence_block, question=question)


# ── Stage 3: generate ─────────────────────────────────────────────────────────
def _get_client():
    """Return an openai.OpenAI client pointed at the configured provider."""
    from openai import OpenAI

    if _LLM_PROVIDER == "ollama":
        base_url = _LLM_BASE_URL or "http://localhost:11434/v1"
        return OpenAI(base_url=base_url, api_key="ollama")

    # OpenAI (or any compatible API)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.  For local inference set "
            "LLM_PROVIDER=ollama and ensure Ollama is running."
        )
    return OpenAI(
        api_key  = api_key,
        base_url = _LLM_BASE_URL or None,
        timeout  = _LLM_TIMEOUT,
    )


def _stage_generate(prompt: str) -> tuple[str, int]:
    """
    Call the LLM and return (answer_text, total_token_count).
    Raises RuntimeError on any API failure so the caller can log and surface
    a graceful error response rather than crashing the request.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model      = _LLM_MODEL,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = _MAX_TOKENS,
        temperature= 0.0,   # deterministic answers for compliance use
    )
    answer     = resp.choices[0].message.content or ""
    token_count = resp.usage.total_tokens if resp.usage else len(prompt.split())
    return answer.strip(), token_count


# ── Stage 4: log ──────────────────────────────────────────────────────────────
def _stage_log(
    request_id:          str,
    question:            str,
    prompt_version:      str,
    retrieval_params:    dict,
    model_name:          str,
    token_count:         int,
    latency_ms:          float,
    generation_succeeded: bool,
    refusal:             bool,
    n_chunks:            int,
) -> None:
    log.info(
        "RAG | request_id=%s prompt_version=%s model=%s "
        "n_chunks=%d tokens=%d latency_ms=%.1f "
        "succeeded=%s refusal=%s filters=%s",
        request_id, prompt_version, model_name,
        n_chunks, token_count, latency_ms,
        generation_succeeded, refusal, retrieval_params,
    )


# ── Public interface ──────────────────────────────────────────────────────────
def ask(
    question: str,
    filters:  dict | None = None,
    top_k:    int = 5,
) -> AnswerResult:
    """
    Full four-stage RAG pipeline.

    Stage 1 (retrieve) may trigger an early return with refusal=True.
    Stage 3 (generate) is NEVER called when retrieval is insufficient.
    """
    request_id = str(uuid.uuid4())
    t_start    = time.perf_counter()

    # ── Stage 1: retrieve ─────────────────────────────────────────────────────
    retrieval = _stage_retrieve(question, filters, top_k)

    # Hard refusal gate -- must be enforced BEFORE reaching the LLM
    if retrieval.insufficient_evidence:
        latency = (time.perf_counter() - t_start) * 1_000
        _stage_log(
            request_id=request_id, question=question,
            prompt_version=PROMPT_VERSION,
            retrieval_params={**(filters or {}), "top_k": top_k},
            model_name=_LLM_MODEL, token_count=0, latency_ms=latency,
            generation_succeeded=False, refusal=True, n_chunks=0,
        )
        return AnswerResult(
            answer               = None,
            retrieved_ids        = [],
            evidence_sufficiency = "insufficient",
            prompt_version       = PROMPT_VERSION,
            refusal              = True,
            model_name           = _LLM_MODEL,
            latency_ms           = round(latency, 2),
            token_count          = 0,
            generation_succeeded = False,
            request_id           = request_id,
        )

    retrieved_ids = [c.complaint_id for c in retrieval.chunks]

    # ── Stage 2: build_prompt ─────────────────────────────────────────────────
    prompt = _stage_build_prompt(question, retrieval.chunks)

    # ── Stage 3: generate ─────────────────────────────────────────────────────
    answer:      str  = REFUSAL_MESSAGE
    token_count: int  = 0
    succeeded:   bool = False

    try:
        answer, token_count = _stage_generate(prompt)
        succeeded = True
    except Exception as exc:  # noqa: BLE001
        log.error("LLM generation failed: %s", exc)
        answer = f"Generation error: {exc}"

    latency = (time.perf_counter() - t_start) * 1_000

    # ── Stage 4: log ──────────────────────────────────────────────────────────
    _stage_log(
        request_id=request_id, question=question,
        prompt_version=PROMPT_VERSION,
        retrieval_params={**(filters or {}), "top_k": top_k},
        model_name=_LLM_MODEL, token_count=token_count,
        latency_ms=latency, generation_succeeded=succeeded,
        refusal=False, n_chunks=len(retrieval.chunks),
    )

    return AnswerResult(
        answer               = answer,
        retrieved_ids        = retrieved_ids,
        evidence_sufficiency = "sufficient",
        prompt_version       = PROMPT_VERSION,
        refusal              = False,
        model_name           = _LLM_MODEL,
        latency_ms           = round(latency, 2),
        token_count          = token_count,
        generation_succeeded = succeeded,
        request_id           = request_id,
    )
