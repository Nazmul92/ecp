"""Rendering verified claims into the final answer.

deterministic: templates only — the rendered text IS the verified text.
polished:      optional LLM rewrite, re-checked (Tier-1 numbers + causal trap)
               against the final prose; falls back to deterministic on violation.
"""
from __future__ import annotations

from typing import Callable, Optional

from .claims import Claim
from .verifier import CAUSAL_MARKERS, WORD_NUMBER_RE, Verifier, extract_numbers

ORDER = ["finding", "comparison", "observation", "inference", "gap", "recommendation"]

HEDGES = {
    "none": "This suggests that ",
    "soft": "One plausible reading is that ",
    "strong": "It is possible, though not established, that ",
}


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


class Renderer:
    def __init__(self, mode: str = "deterministic",
                 polish_llm: Optional[Callable[[str], str]] = None) -> None:
        if mode == "polished" and polish_llm is None:
            raise ValueError("polished mode requires polish_llm")
        self.mode = mode
        self.polish_llm = polish_llm

    # ------------------------------------------------------ deterministic
    def deterministic(self, claims: list[Claim], omitted: int = 0) -> str:
        ordered = sorted(claims, key=lambda c: ORDER.index(c.claim_type)
                         if c.claim_type in ORDER else len(ORDER))
        parts: list[str] = []
        for c in ordered:
            if c.claim_type == "inference":
                lead = HEDGES.get(c.hedge_level, HEDGES["none"])
                body = c.text[0].lower() + c.text[1:] if c.text else c.text
                parts.append(_sentence(lead + body))
            elif c.claim_type == "recommendation":
                parts.append(_sentence("Recommendation: " + c.text))
            else:
                parts.append(_sentence(c.text))
        if omitted:
            parts.append(f"({omitted} statement{'s' if omitted != 1 else ''} could not "
                         "be verified against the available evidence and "
                         f"{'were' if omitted != 1 else 'was'} omitted.)")
        return " ".join(parts)

    # ----------------------------------------------------------- polished
    def polished(self, claims: list[Claim], verifier: Verifier,
                 question: str, omitted: int = 0) -> str:
        assert self.polish_llm is not None
        base = self.deterministic(claims, omitted=0)
        prompt = (
            "Rewrite the following verified statements as one flowing, natural "
            "answer to the question. Rules: you may reorder and connect statements, "
            "but you may NOT add any new facts, numbers, or causal language "
            "(because / due to / led to / caused ...). Keep every number exactly "
            "as written. Keep hedged statements hedged. Output plain prose only.\n\n"
            f"QUESTION: {question}\n\nVERIFIED STATEMENTS:\n{base}"
        )
        prose = self.polish_llm(prompt).strip()

        # Post-render checks: numbers must survive, no new causal markers.
        allowed = set(extract_numbers(base))
        for num in extract_numbers(prose):
            if abs(num) <= verifier.small_int_ceiling and float(num).is_integer():
                continue
            if 1900 <= num <= 2100 and float(num).is_integer():
                continue
            if not any(verifier._close(num, a) for a in allowed):
                return self.deterministic(claims, omitted)      # fail-closed
        has_causal_claims = any(c.claim_type == "causal" for c in claims)
        if not has_causal_claims and CAUSAL_MARKERS.search(prose):
            return self.deterministic(claims, omitted)          # fail-closed
        if WORD_NUMBER_RE.search(prose) and not WORD_NUMBER_RE.search(base):
            return self.deterministic(claims, omitted)          # spelled-out qty introduced

        if omitted:
            prose += (f"\n\n({omitted} statement{'s' if omitted != 1 else ''} could not "
                      "be verified against the available evidence and "
                      f"{'were' if omitted != 1 else 'was'} omitted.)")
        return prose

    def render(self, claims: list[Claim], verifier: Verifier,
               question: str, omitted: int = 0) -> str:
        if not claims:
            return ("No statements could be verified against the available evidence, "
                    "so no answer is given. Please check the audit log.")
        if self.mode == "polished":
            return self.polished(claims, verifier, question, omitted)
        return self.deterministic(claims, omitted)
