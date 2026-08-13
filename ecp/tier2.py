"""Tier-2 entailment backends.

Tier 2 is the only place ECP makes a *semantic* judgement, and it is the one
tier that is probabilistic. The contract is deliberately tiny:

    (claim_text, [cited_passages]) -> "entailed" | "not_entailed" | "partial"

Anything matching that signature works — a local NLI model, a hosted judge, a
human review queue. This module ships one reference backend (an LLM judge) so
the wiring is demonstrated rather than described; it is not a claim that an
LLM judge is the best available entailment model.

    from ecp import VerifiedPipeline, PipelineConfig, llm_judge_backend
    from ecp.llm import anthropic_llm

    judge = llm_judge_backend(anthropic_llm(model="claude-haiku-4-5-20251001"))
    cfg = PipelineConfig.production(audit_path="audit.jsonl", tier2=judge)

Fail-closed by design: if the judge errors, times out, or returns something
unparseable, the verdict is ``"not_entailed"``. Under ``tier2_policy="reject"``
that blocks the claim. A verification tier whose failure mode is "pass" is not
a verification tier.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger("ecp.tier2")

Tier2Backend = Callable[[str, list[str]], str]

VERDICTS = ("entailed", "not_entailed", "partial")

JUDGE_PROMPT = """You are an entailment judge. Decide whether the PASSAGES \
support the STATEMENT.

Answer with exactly one word, lowercase, nothing else:
  entailed      - the passages state or directly imply the statement
  partial       - the passages support part of the statement but not all of it
  not_entailed  - the passages do not support the statement, or say something else

Judge only what the passages say. Do not use outside knowledge. Plausibility is
not entailment: if the statement is reasonable but the passages do not actually
establish it, answer not_entailed.

STATEMENT: {claim}

PASSAGES:
{passages}

One word:"""


def llm_judge_backend(llm: Callable[[str], str], *,
                      treat_partial_as: Optional[str] = None) -> Tier2Backend:
    """Wrap any ``(prompt: str) -> str`` callable as a Tier-2 entailment judge.

    ``treat_partial_as`` optionally collapses ``"partial"`` into ``"entailed"``
    or ``"not_entailed"``. Leaving it ``None`` keeps three-valued output, which
    the verifier treats as a failed entailment (``partial`` is not support).

    The judge should be a *cheap, separate* model call. It never sees raw tool
    output — only the claim text and the cited passages the store already holds.
    """
    if treat_partial_as is not None and treat_partial_as not in ("entailed", "not_entailed"):
        raise ValueError("treat_partial_as must be 'entailed', 'not_entailed', or None")

    def judge(claim_text: str, passages: list[str]) -> str:
        if not passages:
            return "not_entailed"          # nothing to entail from
        body = "\n".join(f"- {p}" for p in passages)
        try:
            raw = llm(JUDGE_PROMPT.format(claim=claim_text, passages=body))
        except Exception as e:             # network, timeout, quota — anything
            logger.warning("tier2 judge failed (%s); failing closed to not_entailed", e)
            return "not_entailed"
        verdict = _parse_verdict(raw)
        if verdict is None:
            logger.warning("tier2 judge returned unparseable output %r; "
                           "failing closed to not_entailed", raw[:200])
            return "not_entailed"
        if verdict == "partial" and treat_partial_as is not None:
            return treat_partial_as
        return verdict

    return judge


def _parse_verdict(raw: str) -> Optional[str]:
    """Extract a verdict from judge output, or None if it is not recognisable.

    Checks ``not_entailed`` before ``entailed`` because the former contains the
    latter as a substring.
    """
    text = (raw or "").strip().lower()
    if not text:
        return None
    for token in ("not_entailed", "not entailed", "partial", "entailed"):
        if re.search(rf"\b{re.escape(token)}\b", text):
            return "not_entailed" if token.startswith("not") else token.replace(" ", "_")
    return None


def keyword_stub_backend(min_overlap: float = 0.6) -> Tier2Backend:
    """A dependency-free stand-in for tests and offline demos.

    Scores content-word overlap between the claim and its passages. This is NOT
    an entailment model and must not be used in production — it is here so the
    Tier-2 code path is exercisable without a model, and so the difference
    between "wired up" and "actually judging" stays visible.
    """
    def judge(claim_text: str, passages: list[str]) -> str:
        words = set(re.findall(r"[a-z]{4,}", claim_text.lower()))
        if not words:
            return "not_entailed"
        corpus = set(re.findall(r"[a-z]{4,}", " ".join(passages).lower()))
        overlap = len(words & corpus) / len(words)
        if overlap >= min_overlap:
            return "entailed"
        return "partial" if overlap >= min_overlap / 2 else "not_entailed"
    return judge
