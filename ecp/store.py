"""EvidenceStore: the only source of facts the synthesis LLM is allowed to see."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .version import __version__ as ECP_VERSION

# Tokens that would let tool-supplied text impersonate the prompt's evidence
# fence. The live fence also carries a per-prompt random nonce (see
# pipeline._context), so forging it requires guessing the nonce; stripping the
# base tokens as well is defence in depth, and costs nothing.
FENCE_TOKEN_RE = re.compile(
    r"\b(?:begin|end)[ _\-]*evidence\b(?:[ \t]+[0-9a-fA-F]{6,})?", re.IGNORECASE)


def sanitize_for_prompt(text: str) -> str:
    """Make tool-supplied text safe to place inside the evidence table.

    Two concrete attacks are closed here:

    1. Fence impersonation — text containing 'END EVIDENCE' would otherwise
       terminate the data region early and let the remainder read as
       instructions.
    2. Row forgery — a newline in an evidence value would otherwise let the
       text emit what looks like an additional 'E-999 [value] ...' row, i.e.
       evidence the store never issued.

    This is a boundary fix, not an injection cure: content inside the fence can
    still *say* anything. What it can no longer do is change the shape of the
    prompt. Claims derived from it still have to survive verification.
    """
    text = FENCE_TOKEN_RE.sub("[redacted-fence-token]", text)
    return re.sub(r"\s+", " ", text).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _full_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _short_hash(payload: Any) -> str:
    return _full_hash(payload)[:8]


@dataclass
class Source:
    tool: str
    call_id: str = ""
    ref: str = ""
    retrieved_at: str = field(default_factory=_now)


@dataclass
class Evidence:
    evidence_id: str
    kind: str                      # value | text | record | causal
    label: str
    value: Any
    unit: Optional[str]
    source: Source
    evidence_type: str             # database_result | api_response | document_chunk |
                                   # computation | human_input | model_output
    confidence: float = 1.0
    supports_causality: bool = False
    raw: Any = None                # original payload, audit only — never shown to the LLM

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ecp_version"] = ECP_VERSION
        return d


class EvidenceStore:
    """Holds every fact the agent 'knows'. If it isn't in here, it doesn't exist."""

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._counter = 0
        self._lock = threading.Lock()      # id allocation is the only mutation race

    # A lock is a runtime primitive, not state. Without these two methods the
    # store is unpicklable ("cannot pickle '_thread.lock' object"), which breaks
    # every LangGraph checkpointer (MemorySaver, SqliteSaver, PostgresSaver) the
    # moment the store travels through graph state — and therefore breaks
    # human-in-the-loop, resumability and time-travel in any graph that mounts
    # the ECP nodes. The lock is recreated on load.
    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- add
    def _new_id(self, prefix: str = "E") -> str:
        with self._lock:
            self._counter += 1
            return f"{prefix}-{self._counter:03d}"

    def add_value(self, label: str, value: Any, unit: Optional[str] = None, *,
                  source_tool: str, call_id: str = "", ref: str = "",
                  evidence_type: str = "database_result",
                  supports_causality: bool = False, confidence: float = 1.0,
                  raw: Any = None) -> Evidence:
        ev = Evidence(self._new_id(), "value", label, value, unit,
                      Source(source_tool, call_id, ref), evidence_type,
                      confidence, supports_causality, raw)
        self._items[ev.evidence_id] = ev
        return ev

    def add_text(self, text: str, *, label: str = "", source_tool: str,
                 call_id: str = "", ref: str = "",
                 evidence_type: str = "document_chunk",
                 supports_causality: bool = False, confidence: float = 1.0,
                 raw: Any = None) -> Evidence:
        ev = Evidence(self._new_id(), "causal" if supports_causality else "text",
                      label or (text[:60] + ("..." if len(text) > 60 else "")),
                      text, None, Source(source_tool, call_id, ref),
                      evidence_type, confidence, supports_causality, raw)
        self._items[ev.evidence_id] = ev
        return ev

    def ingest_json(self, payload: Any, *, source_tool: str, call_id: str = "",
                    path_labels: Optional[dict[str, str]] = None,
                    evidence_type: str = "api_response") -> list[Evidence]:
        """Auto mode: every scalar leaf becomes evidence with a path-derived label.

        Works everywhere; label quality is worse than curated add_value calls.
        Prefer curated ingestion for your core tools.
        """
        path_labels = path_labels or {}
        out: list[Evidence] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif node is not None:
                label = path_labels.get(path, path)
                if isinstance(node, (int, float)) and not isinstance(node, bool):
                    out.append(self.add_value(label, node, source_tool=source_tool,
                                              call_id=call_id, ref=path,
                                              evidence_type=evidence_type, raw=payload))
                else:
                    out.append(self.add_text(str(node), label=label,
                                             source_tool=source_tool, call_id=call_id,
                                             ref=path, evidence_type=evidence_type,
                                             raw=payload))
        walk(payload, "")
        return out

    # -------------------------------------------------------------- query
    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._items.get(evidence_id)

    def exists(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def manifest(self) -> list[str]:
        return list(self._items.keys())

    def manifest_hash(self) -> str:
        """Full SHA-256 over every evidence record (audit-grade)."""
        return _full_hash([e.to_dict() for e in self._items.values()])

    def snapshot(self) -> list[dict]:
        """Durable copies of every evidence record for embedding in proofs.
        Excludes 'raw' (can be large); everything needed to reconstruct what
        E-001 meant months later is included."""
        out = []
        for e in self._items.values():
            d = e.to_dict()
            d.pop("raw", None)
            out.append(d)
        return out

    # ------------------------------------------------------------- render
    def _row(self, e: Evidence) -> str:
        label = sanitize_for_prompt(str(e.label))
        if e.kind == "value":
            val = f"{label} = {e.value}" + (f" {e.unit}" if e.unit else "")
        else:
            val = f'{label}: "{sanitize_for_prompt(str(e.value))}"'
        causal = " causal:yes" if e.supports_causality else ""
        return f"{e.evidence_id}  [{e.kind}]  {val}  ({sanitize_for_prompt(e.source.tool)}{causal})"

    def render_table(self, *, max_chars: Optional[int] = None) -> tuple[str, dict]:
        """Render the evidence table and report what the model actually saw.

        Returns ``(text, stats)`` with stats
        ``{total, included, omitted, truncated, chars}``.

        Auto-ingestion turns every scalar leaf of a tool result into a row, so
        this table is the one unbounded input to the prompt. ``max_chars`` caps
        it. Truncation is never silent: the omitted count goes into the table
        itself (so the model can answer with a 'gap' claim instead of
        confabulating over missing rows), into ``result.metrics``, and into the
        proof object (so an auditor knows the answer was formed on a subset).
        """
        rows = [self._row(e) for e in self._items.values()]
        total = len(rows)
        header = "EVIDENCE"
        full = "\n".join([header] + rows)
        if max_chars is None or len(full) <= max_chars:
            return full, {"total": total, "included": total, "omitted": 0,
                          "truncated": False, "chars": len(full)}

        # Truncating. The notice is part of the table the model reads, so its
        # length is reserved from the budget rather than added on top of it —
        # otherwise a "cap" would be exceeded by exactly the text announcing
        # that the cap was hit.
        def notice(omitted: int) -> str:
            return (f"[{omitted} of {total} evidence rows omitted: the evidence "
                    "table exceeded the configured context budget. You are seeing "
                    "a subset. If the question needs the omitted rows, say so with "
                    "a 'gap' claim rather than guessing.]")

        budget = max_chars - len(notice(total)) - 1      # worst-case notice width
        kept: list[str] = []
        size = len(header)
        for row in rows:
            if size + len(row) + 1 > budget:
                break
            kept.append(row)
            size += len(row) + 1
        omitted = total - len(kept)
        text = "\n".join([header] + kept + [notice(omitted)])
        return text, {"total": total, "included": len(kept), "omitted": omitted,
                      "truncated": True, "chars": len(text)}

    def table(self) -> str:
        """Compact evidence table — the ONLY view of the world the LLM receives."""
        return self.render_table()[0]
