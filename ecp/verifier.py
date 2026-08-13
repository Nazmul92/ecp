"""Tiered verification. Tiers 0/1 + causal gate are deterministic and carry
the hard guarantees (G1–G4). Tier 2 (entailment) is probabilistic, pluggable,
and advisory by default — its fallibility never weakens G1–G4.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .calc import CalcRegistry
from .claims import Claim, CLAIM_TYPES, FACTUAL_TYPES
from .errors import ConfigError, validate_choice
from .store import EvidenceStore

# _ADV: up to two intervening words so adverb insertion can't defeat a marker
# ("driven primarily by", "due in large part to", "led directly to").
# up to three intervening non-conjunction words; conjunctions excluded so
# "data driven and by Q3" does not false-positive
_ADV = r"(?:\s+(?!and\b|or\b|but\b|nor\b)\w+){0,3}"
CAUSAL_MARKERS = re.compile(
    r"\b(because|caused|causes|causing"
    rf"|led{_ADV}\s+to|leads{_ADV}\s+to|leading{_ADV}\s+to"
    rf"|due{_ADV}\s+to|owing{_ADV}\s+to"
    r"|drove|drives"
    rf"|driven{_ADV}\s+by"
    rf"|result(?:ed|s|ing){_ADV}\s+(?:in|from)"
    r"|as a result of|thanks to|attributable to|attributed to"
    r"|triggered|triggers|explains|explained by"
    rf"|fueled|fuelled|fuels|stems?{_ADV}\s+from|stemmed{_ADV}\s+from"
    r"|responsible for|underpinned|underpins|sparked"
    r"|prompted by|prompted|gave rise to|brought about"
    r"|consequence of|effect of|impact of|induced|induces)\b",
    re.IGNORECASE,
)
# NOTE: lexical gating catches known markers only; it is a tripwire, not a
# semantic guarantee. Novel causal phrasings can pass — pair with Tier-2 for
# entailment-level coverage. (Scope of G4 is stated accordingly in README.)

# Numbers not glued to letters (skips Q2, H1, F001); commas, decimals, %, sign.
# (?<![A-Za-z]-) keeps letter-hyphen-digit identifiers (Tier-2, GPT-4,
# COVID-19) out of extraction while numeric ranges (12-15) still yield
# both tokens.
NUMBER_RE = re.compile(r"(?<![\w.])(?<![A-Za-z]-)-?\d[\d,]*(?:\.\d+)?%?(?![\w])")

# Spelled-out numbers with quantitative weight. Factual claims must use digits
# (enforced) so every quantity is machine-verifiable; single casual small words
# ("one plausible reading") are not matched.
_WNUM = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
         r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
         r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
         r"thousand|million|billion)")
# NOTE: fraction words (half/quarter/third) are deliberately excluded — they
# collide with temporal phrases ("one quarter after Q1", "the first half").
_UNITISH = (r"(?:percent|per\s?cent|points?|units?|dollars?|users?|customers?|"
            r"accounts?|times|fold|bps|basis\s+points)")
WORD_NUMBER_RE = re.compile(
    rf"\b(?:{_WNUM}(?:[\s-]+(?:point[\s-]+)?{_WNUM})+"      # multi-word: "twelve point three"
    rf"|{_WNUM}\s+{_UNITISH}"                                # "twelve percent"
    rf"|(?:hundred|thousand|million|billion))\b",
    re.IGNORECASE,
)

SCI_NOTATION_RE = re.compile(r"\b\d+(?:\.\d+)?[eE][+-]?\d+\b")

# textual unit words adjacent to a number ("100 dollars", "12 points") — a
# bounded lexicon so a sentence cannot relabel a value's unit while the
# asserted_value declares the honest one.
TEXT_UNIT_RE = re.compile(
    r"(percent|per\s?cent|points?|units?|dollars?|usd|users?|customers?|"
    r"accounts?|bps|basis\s+points|cases?|tests?|items?|orders?)\b",
    re.IGNORECASE)


def _adjacent_text_unit(text: str, num_end: int) -> Optional[str]:
    m = re.match(r"\s+", text[num_end:])
    rest = text[num_end + (m.end() if m else 0):]
    um = TEXT_UNIT_RE.match(rest)
    return um.group(1) if um else None


DECREASE_WORDS = re.compile(
    r"\b(fell|fall|falls|fallen|drop|drops|dropped|declin\w+|decreas\w+|down|"
    r"shrank|shrunk|contract\w+|lower|lost|loss|reduc\w+|dip|dipped|slid|slump\w+)\b",
    re.IGNORECASE)
INCREASE_WORDS = re.compile(
    r"\b(rose|rise|rises|risen|grew|grow|grows|grown|growth|increas\w+|up|"
    r"gain\w*|climb\w+|higher|expand\w+|jump\w+|surge\w+|improv\w+)\b",
    re.IGNORECASE)

YEAR_CONTEXT = re.compile(
    # period-before-year ("Q1 2026", "H1 2026") and year-before-period
    # ("2026 Q2", "2026 H1") must be symmetric: h[12] was previously missing
    # from this first branch, so "H1 2026" false-rejected while "Q1 2026" and
    # "2026 H1" both passed.
    r"(?:\b(?:in|since|by|until|from|during|through|of|year|fy|q[1-4]|h[12])\s+(\d{4})\b"
    r"|\b(\d{4})\s*(?:q[1-4]\b|h[12]\b))",
    re.IGNORECASE)


_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_LITERAL = re.compile(
    r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b"                     # 2026-08-15, 2026-08
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"                       # 15/08/2026
    r"|\b\d{4}/\d{1,2}/\d{1,2}\b"                         # 2026/08/15
    rf"|\b\d{{1,2}}\s+{_MONTHS}\.?,?\s+\d{{4}}\b"         # 15 August 2026
    rf"|\b{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}\b",        # August 15, 2026
    re.IGNORECASE)


def _date_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of date literals.

    Dates are not quantities and cannot be bound to an ``asserted_value``:
    ``AssertedValue.value`` is a float, so there is no way to express
    '2026-08-15' as evidence. Without this, every number inside a date
    ('2026', '8', '15') reads as an ungrounded quantity and any claim stating a
    delivery date, invoice date or period is rejected -- which is most of the
    claims a real support or ops agent wants to make.

    Exemption is by SPAN, not by value: '15' is exempt because it sits inside
    '2026-08-15', not everywhere it happens to appear in the sentence.

    Scope, stated plainly: ECP does NOT verify dates. A date inside a claim is
    passed through unchecked, exactly like the year exemption. If a date is
    load-bearing for you, cite it as text evidence and check it with Tier 2.
    """
    return [m.span() for m in DATE_LITERAL.finditer(text)]


def _year_tokens(text: str) -> set[float]:
    """Integers 1900–2100 that appear in a date-like context ('in 2026', 'FY 2025',
    '2026 Q2', 'H1 2026'). Bare 4-digit quantities ('shipped 1950 units') are NOT
    exempt."""
    out: set[float] = set()
    for m in YEAR_CONTEXT.finditer(text):
        tok = m.group(1) or m.group(2)
        y = float(tok)
        if 1900 <= y <= 2100:
            out.add(y)
    return out


_UNIT_ALIASES = {"%": "percent", "pct": "percent", "per cent": "percent",
                 "unit": "units", "usd": "dollars", "$": "dollars",
                 "point": "points", "bp": "bps", "basis points": "bps"}


def _norm_unit(u: str) -> str:
    u = u.strip().lower()
    return _UNIT_ALIASES.get(u, u)


def extract_number_tokens(text: str) -> list[tuple[float, bool]]:
    """All numeric tokens as (value, is_percent)."""
    out: list[tuple[float, bool]] = []
    for tok in NUMBER_RE.findall(text):
        is_pct = tok.endswith("%")
        tok = tok.rstrip("%").replace(",", "")
        try:
            out.append((float(tok), is_pct))
        except ValueError:
            pass
    return out


def extract_numbers(text: str) -> list[float]:
    return [v for v, _ in extract_number_tokens(text)]


@dataclass
class TierResult:
    tier: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"tier": self.tier, "pass": self.passed, "reason": self.reason}


@dataclass
class VerificationResult:
    claim_id: str
    status: str                       # verified | rejected | downgraded
    tier_results: list[TierResult] = field(default_factory=list)
    repairable: bool = True
    repair_hint: str = ""
    downgraded_to: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("verified", "downgraded")

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "tier_results": [t.to_dict() for t in self.tier_results],
            "repairable": self.repairable,
            "repair_hint": self.repair_hint,
            "downgraded_to": self.downgraded_to,
        }


# Tier-2 backend contract: (claim_text, [cited_evidence_texts]) -> "entailed" | "not_entailed" | "partial"
Tier2Backend = Callable[[str, list[str]], str]


class Verifier:
    def __init__(self, store: EvidenceStore, calcs: CalcRegistry, *,
                 rel_tol: float = 0.005, abs_tol: float = 0.01,
                 causal_gate: bool = True,
                 tier2: Optional[Tier2Backend] = None,
                 tier2_policy: str = "downgrade",   # downgrade | reject | annotate
                 prose_policy: str = "allow",       # allow | downgrade (no-Tier-2 prose)
                 allow_model_output_evidence: bool = True,
                 small_int_ceiling: int = 0) -> None:
        # Policy strings are validated here as well as in PipelineConfig:
        # Verifier is public API and is constructed directly by the benchmark
        # harness and by anyone embedding the checks without the pipeline.
        # An unknown tier2_policy used to fall through to 'annotate' (pass).
        self.store = store
        self.calcs = calcs
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.causal_gate = causal_gate
        self.tier2 = tier2
        self.tier2_policy = validate_choice("tier2_policy", tier2_policy,
                                            ("downgrade", "reject", "annotate"))
        self.prose_policy = validate_choice("prose_policy", prose_policy,
                                            ("allow", "downgrade"))
        if small_int_ceiling < 0:
            raise ConfigError(f"small_int_ceiling must be >= 0 (got {small_int_ceiling!r})")
        self.allow_model_output_evidence = allow_model_output_evidence
        # 0 (default) = every number must ground (G2, strict). Teams may raise
        # this to tolerate small ordinal integers, accepting the G2 weakening.
        self.small_int_ceiling = small_int_ceiling

    # ------------------------------------------------------------ helpers
    def _ref_unit(self, ref: str) -> Optional[str]:
        if self.calcs.exists(ref):
            return self.calcs.get(ref).unit
        ev = self.store.get(ref)
        return ev.unit if ev else None

    def _ref_exists(self, ref: str) -> bool:
        return self.store.exists(ref) or self.calcs.exists(ref)

    def _ref_value(self, ref: str) -> Optional[float]:
        if self.calcs.exists(ref):
            return self.calcs.get(ref).result
        ev = self.store.get(ref)
        if ev and isinstance(ev.value, (int, float)) and not isinstance(ev.value, bool):
            return float(ev.value)
        return None

    def _close(self, a: float, b: float) -> bool:
        """Exact tolerance match. No sign forgiveness — see _matches for the
        direction-aware sign-flip path."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def _matches(self, asserted: float, evidence: float, text: str) -> bool:
        """Value match with direction-aware sign flip.

        'decreased by 12%' (asserted +12.28) vs pct_change = -12.28 is fine
        BECAUSE the text says decrease and the evidence is negative. 'rose
        12.28%' against -12.28 is a direction inversion and must fail (G2)."""
        if self._close(asserted, evidence):
            return True
        if not self._close(asserted, -evidence):
            return False
        # magnitudes agree, signs differ: text direction must match evidence sign
        if evidence < 0 and DECREASE_WORDS.search(text) and not INCREASE_WORDS.search(text):
            return True
        if evidence > 0 and INCREASE_WORDS.search(text) and not DECREASE_WORDS.search(text):
            return True
        return False

    # -------------------------------------------------------------- tiers
    def _tier0(self, c: Claim) -> TierResult:
        if c.claim_type not in CLAIM_TYPES:
            return TierResult("structural", False, f"Unknown claim_type '{c.claim_type}'")
        if not c.text:
            return TierResult("structural", False, "Empty claim text")
        if c.claim_type in FACTUAL_TYPES and not c.cites:
            return TierResult("structural", False,
                              f"'{c.claim_type}' claims require at least one citation")
        for ref in c.cites:
            if not self._ref_exists(ref):
                return TierResult("structural", False, f"Cited '{ref}' does not exist")
        if not self.allow_model_output_evidence and c.claim_type in FACTUAL_TYPES:
            kinds = [self.store.get(r).evidence_type for r in c.cites if self.store.get(r)]
            if kinds and all(k == "model_output" for k in kinds):
                return TierResult("structural", False,
                                  "Claim rests only on model_output evidence (disallowed by profile)")
        return TierResult("structural", True)

    def _tier1(self, c: Claim) -> TierResult:
        # Numeric grounding applies to EVERY claim type. Inference and
        # recommendation may speculate about meaning, but any quantity they
        # state must still resolve to cited evidence — otherwise fabricated
        # facts just migrate into the "interpretive" claim types.

        # (a0) quantities must be written as plain digits so they can be verified
        m = WORD_NUMBER_RE.search(c.text)
        if m:
            return TierResult("value", False,
                              f"spelled-out quantity '{m.group(0)}' cannot be verified; "
                              "write all numbers as digits")
        m = SCI_NOTATION_RE.search(c.text)
        if m:
            return TierResult("value", False,
                              f"scientific notation '{m.group(0)}' cannot be verified; "
                              "write plain digits")

        # (a) asserted_values must resolve to cited refs with matching values
        for av in c.asserted_values:
            if av.from_ not in c.cites:
                return TierResult("value", False,
                                  f"asserted value {av.value} points to '{av.from_}' which is not cited")
            ref_val = self._ref_value(av.from_)
            if ref_val is None:
                return TierResult("value", False,
                                  f"'{av.from_}' has no numeric value to support {av.value}")
            if not self._matches(av.value, ref_val, c.text):
                return TierResult("value", False,
                                  f"asserted {av.value} != {av.from_} value {ref_val}"
                                  + (" (direction of change contradicts evidence sign)"
                                     if self._close(av.value, -ref_val) else ""))
            # unit consistency: if the evidence declares a unit, the assertion
            # must declare a compatible one (omitting the unit is not a bypass)
            ref_unit = self._ref_unit(av.from_)
            if ref_unit:
                if not av.unit:
                    return TierResult("value", False,
                                      f"asserted value {av.value} omits a unit but "
                                      f"{av.from_} is in '{ref_unit}'; declare the unit")
                if _norm_unit(av.unit) != _norm_unit(ref_unit):
                    return TierResult("value", False,
                                      f"unit mismatch: asserted '{av.unit}' vs "
                                      f"{av.from_} unit '{ref_unit}'")
            elif av.unit:
                return TierResult("value", False,
                                  f"asserted unit '{av.unit}' but {av.from_} is "
                                  "unitless; a unit cannot be invented for a "
                                  "dimensionless value")

        # (b) every number in the text must be BOUND to an asserted_value
        #     (which ties it to a cited ref and a declared unit); percent tokens
        #     need percent-united backing; a textual unit word next to the
        #     number must agree with the declared unit
        bound: list[tuple[float, Optional[str]]] = [
            (av.value, av.unit or self._ref_unit(av.from_)) for av in c.asserted_values]
        years = _year_tokens(c.text)
        date_spans = _date_spans(c.text)
        for tok in NUMBER_RE.finditer(c.text):
            raw = tok.group(0)
            is_pct = raw.endswith("%")
            try:
                num = float(raw.rstrip("%").replace(",", ""))
            except ValueError:
                continue
            if abs(num) <= self.small_int_ceiling and float(num).is_integer():
                continue          # 0 by default: every number must bind (G2)
            if any(s <= tok.start() and tok.end() <= e for s, e in date_spans):
                continue          # inside a date literal; dates are not quantities
            if num in years:
                continue          # date-context years only ('in 2026'), not bare quantities
            match_unit, supported = None, False
            for gv, gu in bound:
                if not self._matches(num, gv, c.text):
                    continue
                if is_pct and (gu is None or _norm_unit(gu) != "percent"):
                    continue      # '3%' cannot be backed by a non-percent value
                match_unit, supported = gu, True
                break
            if not supported:
                return TierResult("value", False,
                                  f"number {num}{'%' if is_pct else ''} in text is not "
                                  "bound to any asserted_value"
                                  + (" with unit %" if is_pct else "")
                                  + "; declare it in asserted_values with its source ref")
            tu = _adjacent_text_unit(c.text, tok.end())
            if tu and match_unit and _norm_unit(tu) != _norm_unit(match_unit):
                return TierResult("value", False,
                                  f"text says '{num} {tu}' but the bound value's unit is "
                                  f"'{match_unit}'")
            if tu and match_unit is None:
                return TierResult("value", False,
                                  f"text attaches unit '{tu}' to {num} but the bound "
                                  "value/calc has no unit")

        # (b2) a factual claim with no quantities is pure prose: its citations
        #     must include text-kind evidence something could entail it from —
        #     a numeric value cannot support a qualitative sentence
        if c.claim_type in FACTUAL_TYPES and not c.asserted_values \
                and not extract_numbers(c.text):
            kinds = [self.store.get(r).kind for r in c.cites if self.store.get(r)]
            if kinds and not any(k in ("text", "causal") for k in kinds):
                return TierResult("value", False,
                                  "qualitative claim cites only numeric evidence, which "
                                  "cannot support it; cite text evidence or restate "
                                  "numerically")

        # (c) recompute every cited calculation (G3)
        for ref in c.cites:
            if self.calcs.exists(ref) and not self.calcs.recompute(ref):
                return TierResult("value", False, f"calculation {ref} failed recomputation")

        return TierResult("value", True)

    def _causal(self, c: Claim) -> TierResult:
        if not self.causal_gate:
            return TierResult("causal_gate", True, "gate disabled")
        if c.claim_type == "causal" or c.causal:
            has_causal_ev = any(
                (self.store.get(r) and self.store.get(r).supports_causality)
                for r in c.cites
            )
            if not has_causal_ev:
                return TierResult("causal_gate", False,
                                  "causal claim without evidence marked supports_causality")
        elif c.claim_type in {"finding", "observation", "comparison"}:
            m = CAUSAL_MARKERS.search(c.text)
            if m:
                return TierResult("causal_gate", False,
                                  f"causal language ('{m.group(0)}') in a non-causal claim")
        return TierResult("causal_gate", True)

    def _tier2(self, c: Claim) -> Optional[TierResult]:
        if self.tier2 is None or c.claim_type not in FACTUAL_TYPES:
            return None
        texts = []
        for ref in c.cites:
            ev = self.store.get(ref)
            if ev and ev.kind in ("text", "causal"):
                texts.append(str(ev.value))
        if not texts:
            return None
        verdict = self.tier2(c.text, texts)
        if verdict == "entailed":
            return TierResult("entailment", True)
        return TierResult("entailment", False, f"tier2 verdict: {verdict}")

    # --------------------------------------------------------------- main
    def verify(self, claim: Claim) -> VerificationResult:
        tiers: list[TierResult] = []

        t0 = self._tier0(claim)
        tiers.append(t0)
        if not t0.passed:
            return VerificationResult(claim.claim_id, "rejected", tiers,
                                      repair_hint=t0.reason)

        t1 = self._tier1(claim)
        tiers.append(t1)
        if not t1.passed:
            return VerificationResult(claim.claim_id, "rejected", tiers,
                                      repair_hint=t1.reason +
                                      ". Use only values present in the evidence table, "
                                      "or request a calculation.")

        tc = self._causal(claim)
        tiers.append(tc)
        if not tc.passed:
            return VerificationResult(claim.claim_id, "rejected", tiers,
                                      repair_hint=tc.reason +
                                      ". Restate as an observation or inference, "
                                      "or drop the causal framing.")

        t2 = self._tier2(claim)
        if t2 is not None:
            tiers.append(t2)
            if not t2.passed:
                if claim.claim_type == "causal":
                    # the causal gate checks provenance, not the relation;
                    # a failed entailment on a causal claim is never advisory
                    return VerificationResult(claim.claim_id, "rejected", tiers,
                                              repair_hint="cited causal evidence does not "
                                              "entail this specific causal relationship")
                if self.tier2_policy == "reject":
                    return VerificationResult(claim.claim_id, "rejected", tiers,
                                              repair_hint="Cited passage does not entail "
                                              "the sentence; restate closer to the source.")
                if self.tier2_policy == "downgrade":
                    return VerificationResult(claim.claim_id, "downgraded", tiers,
                                              downgraded_to="inference")
                # annotate: verified with a flag in tier_results

        if (self.prose_policy == "downgrade" and self.tier2 is None
                and claim.claim_type in FACTUAL_TYPES and claim.claim_type != "causal"
                and not claim.asserted_values and not extract_numbers(claim.text)):
            # production-without-Tier-2: qualitative prose is not presented as
            # verified fact; it survives as a hedged inference
            return VerificationResult(claim.claim_id, "downgraded", tiers,
                                      downgraded_to="inference")
        return VerificationResult(claim.claim_id, "verified", tiers)
