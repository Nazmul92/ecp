"""05 — Wiring a Tier-2 entailment backend.

Tiers 0/1 and the causal gate are deterministic: they prove provenance and
arithmetic. They cannot tell you whether a sentence is *semantically* supported
by the passage it cites. That is Tier 2, and it is the one tier ECP does not
ship a model for.

This example runs the same unsupported claim through three configurations so
the difference is visible rather than described:

    A. no Tier-2                     -> deterministically clean, ACCEPTED
    B. no Tier-2 + prose_policy      -> downgraded to a hedged inference
    C. Tier-2 judge, policy=reject   -> REJECTED

Run offline (keyword stub, no model):     python examples/05_tier2_backend.py
Run against a local model via Ollama:     ECP_BACKEND=ollama python examples/05_tier2_backend.py

The claim below is the corpus case T01: 'The company is insolvent.' cited to a
survey passage about pricing. Nothing in it is fabricated numerically — which
is exactly why the deterministic tiers let it through.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecp import (CalcRegistry, EvidenceStore, PipelineConfig, Verifier,
                 llm_judge_backend)
from ecp.claims import Claim
from ecp.tier2 import keyword_stub_backend

CLAIM_TEXT = "The company is insolvent."
PASSAGE = "Customer survey: several respondents mentioned pricing concerns."


def world():
    s = EvidenceStore()
    t = s.add_text(PASSAGE, source_tool="survey", evidence_type="document_chunk")
    return s, CalcRegistry(s), t


def make_judge():
    """Pick a Tier-2 backend. Real deployments use a model; the stub keeps this
    example runnable in CI with no network and no weights."""
    if os.environ.get("ECP_BACKEND") == "ollama":
        from ecp.llm import ollama_llm
        model = os.environ.get("ECP_MODEL", "mistral:7b")
        print(f"Tier-2 backend: LLM judge on ollama/{model}\n")
        # format_json=False: the judge answers with one bare word, not JSON.
        return llm_judge_backend(ollama_llm(model=model, format_json=False))
    print("Tier-2 backend: keyword stub (NOT an entailment model - demo only)\n")
    return keyword_stub_backend()


def main() -> int:
    judge = make_judge()
    print(f"passage: {PASSAGE}")
    print(f"claim:   {CLAIM_TEXT}\n")

    # A. deterministic tiers only
    s, c, t = world()
    claim = Claim("S1", "observation", CLAIM_TEXT, cites=[t.evidence_id])
    a = Verifier(s, c).verify(claim)
    print(f"A. no Tier-2                    -> {a.status}")

    # B. no backend, but production prose policy
    s, c, t = world()
    claim = Claim("S1", "observation", CLAIM_TEXT, cites=[t.evidence_id])
    b = Verifier(s, c, prose_policy="downgrade").verify(claim)
    print(f"B. prose_policy=downgrade       -> {b.status} "
          f"(to {b.downgraded_to})")

    # C. Tier-2 backend, strict policy
    s, c, t = world()
    claim = Claim("S1", "observation", CLAIM_TEXT, cites=[t.evidence_id])
    cc = Verifier(s, c, tier2=judge, tier2_policy="reject").verify(claim)
    print(f"C. Tier-2 judge, policy=reject  -> {cc.status}")
    if cc.tier_results and cc.tier_results[-1].tier == "entailment":
        print(f"   verdict: {cc.tier_results[-1].reason}")

    print("\nProduction wiring:")
    print("  cfg = PipelineConfig.production(audit_path='audit.jsonl', tier2=judge)")
    cfg = PipelineConfig.production(audit_path="audit.jsonl", tier2=judge)
    print(f"  -> tier2_policy={cfg.tier2_policy!r}, prose_policy={cfg.prose_policy!r}")
    print("  (with a backend present, prose is judged rather than blanket-downgraded)")

    # A must accept and C must reject, or this example is not demonstrating
    # what it claims to demonstrate — so CI treats that as a failure.
    if a.status != "verified":
        print("\nUNEXPECTED: deterministic tiers should accept this claim")
        return 1
    if cc.status != "rejected":
        print("\nUNEXPECTED: Tier-2 with policy=reject should block this claim")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
