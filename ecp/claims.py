"""Claim objects: the only thing the synthesis LLM is allowed to produce."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

CLAIM_TYPES = {
    "finding",         # value-level factual statement
    "observation",     # categorical/textual fact from evidence
    "comparison",      # relation between two evidence values
    "causal",          # X caused Y — requires causal evidence
    "inference",       # labeled interpretation; rendered hedged
    "recommendation",  # opinion/advice; rendered in distinct register
    "gap",             # "the evidence does not establish X" — always legal
}

FACTUAL_TYPES = {"finding", "observation", "comparison", "causal"}


@dataclass
class AssertedValue:
    value: float
    unit: Optional[str] = None
    from_: str = ""                       # evidence_id or calc_id ("from" in JSON)

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "from": self.from_}


@dataclass
class Claim:
    claim_id: str
    claim_type: str
    text: str
    cites: list[str] = field(default_factory=list)
    asserted_values: list[AssertedValue] = field(default_factory=list)
    causal: bool = False
    hedge_level: str = "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asserted_values"] = [av.to_dict() for av in self.asserted_values]
        return d

    def content_hash(self) -> str:
        """Semantic identity: excludes claim_id, which may be renumbered across repair rounds."""
        import hashlib
        d = self.to_dict()
        d.pop("claim_id", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class ClaimParseError(ValueError):
    pass


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_claims(llm_output: str) -> tuple[list[Claim], list[dict]]:
    """Parse LLM output into Claims. Returns (claims, calc_requests).

    Accepts either a JSON array of claim objects, or an object of the form
    {"claims": [...], "request_calcs": [{"operation":..., "inputs":[...]}]}.
    """
    raw = _strip_fences(llm_output)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClaimParseError(f"Synthesis output is not valid JSON: {e}") from e

    calc_requests: list[dict] = []
    if isinstance(data, dict):
        calc_requests = data.get("request_calcs", []) or []
        data = data.get("claims", [])
    if not isinstance(data, list):
        raise ClaimParseError("Expected a JSON array of claims.")

    claims: list[Claim] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ClaimParseError(f"Claim #{i} is not an object.")
        try:
            avs = []
            for av in item.get("asserted_values", []) or []:
                if not isinstance(av, dict) or "value" not in av:
                    raise ClaimParseError(
                        f"Claim #{i}: asserted_values entries need a numeric 'value'.")
                avs.append(AssertedValue(value=float(av["value"]),
                                         unit=av.get("unit"),
                                         from_=str(av.get("from", ""))))
            cites = item.get("cites", []) or []
            if not isinstance(cites, list):
                raise ClaimParseError(f"Claim #{i}: 'cites' must be a list.")
            claims.append(Claim(
                claim_id=item.get("claim_id") or f"S-{i:03d}",
                claim_type=str(item.get("claim_type", "")).lower(),
                text=str(item.get("text", "")).strip(),
                cites=[str(x) for x in cites],
                asserted_values=avs,
                causal=bool(item.get("causal", False)),
                hedge_level=str(item.get("hedge_level", "none")),
            ))
        except ClaimParseError:
            raise
        except Exception as e:               # any other malformation fails closed
            raise ClaimParseError(f"Claim #{i} is malformed: {e}") from e
    return claims, calc_requests
