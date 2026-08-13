"""CalcRegistry: derived values are computed by code, registered, and recomputable.

The LLM never performs arithmetic. It may *request* a calculation; the runtime
executes it and the result becomes citable evidence (a calc_id works anywhere
an evidence_id does).
"""
from __future__ import annotations

import math
import pickle
import statistics
import threading
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from .store import EvidenceStore


class CalcError(ValueError):
    """A calculation could not be performed (bad inputs, division by zero,
    non-finite result). Raised from register(); recompute() returns False instead."""


def _pct_change(old: float, new: float) -> float:
    return (new - old) / old * 100.0


OPS: dict[str, Callable[..., float]] = {
    "sum": lambda *xs: sum(xs),
    "mean": lambda *xs: statistics.fmean(xs),
    "diff": lambda a, b: b - a,
    "ratio": lambda a, b: a / b,
    "pct_change": _pct_change,          # inputs: [old, new]
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "count": lambda *xs: float(len(xs)),
}


@dataclass
class Calculation:
    calc_id: str
    operation: str
    inputs: list[str]                    # evidence_ids (or calc_ids)
    result: float
    unit: Optional[str] = None
    expression: str = ""
    computed_by: str = "ecp.calc"
    recomputable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CalcRegistry:
    def __init__(self, store: EvidenceStore) -> None:
        self.store = store
        self._calcs: dict[str, Calculation] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._custom_ops: dict[str, Callable[..., float]] = {}

    # See EvidenceStore.__getstate__ — same reason: a CalcRegistry travelling
    # through LangGraph state must survive checkpoint serialization.
    def __getstate__(self) -> dict:
        # Custom ops are the one thing here that may not be serializable:
        # pickle stores a function by qualified name, so a module-level `def`
        # survives while a lambda or a closure does not.
        #
        # Fail here rather than dropping the op. A restored registry missing
        # its op would recompute to False (G3), silently turning an already
        # verified calculation into a rejected claim on resume — a correctness
        # change disguised as a verification result. An error at save time is
        # the honest failure.
        for name, fn in self._custom_ops.items():
            try:
                pickle.dumps(fn)
            except Exception as e:
                raise TypeError(
                    f"custom calc op {name!r} cannot be serialized ({e.__class__.__name__}), "
                    "so this CalcRegistry cannot be checkpointed. Register the op as a "
                    "module-level `def` instead of a lambda or closure — pickle stores "
                    "functions by qualified name. If the op genuinely cannot be made "
                    "module-level, keep the registry out of checkpointed graph state and "
                    "rebuild it per run."
                ) from e
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def register_op(self, name: str, fn: Callable[..., float]) -> None:
        """Register a custom pure function as an available operation.

        Use a module-level `def` if this registry will ever be checkpointed —
        see __getstate__. Built-in ops carry no such constraint; they are
        looked up by name at recompute time, not stored.
        """
        self._custom_ops[name] = fn

    def _op(self, name: str) -> Callable[..., float]:
        if name in self._custom_ops:
            return self._custom_ops[name]
        if name in OPS:
            return OPS[name]
        raise KeyError(f"Unknown operation '{name}'")

    def _resolve(self, ref: str) -> float:
        if ref in self._calcs:
            return self._calcs[ref].result
        ev = self.store.get(ref)
        if ev is None:
            raise KeyError(f"Unknown input '{ref}'")
        if not isinstance(ev.value, (int, float)) or isinstance(ev.value, bool):
            raise TypeError(f"Input '{ref}' is not numeric (kind={ev.kind})")
        return float(ev.value)

    def _input_unit(self, ref: str) -> Optional[str]:
        if ref in self._calcs:
            return self._calcs[ref].unit
        ev = self.store.get(ref)
        return ev.unit if ev else None

    def _derive_unit(self, operation: str, inputs: list[str]) -> Optional[str]:
        """Units are a property of the operation and its inputs — never of who
        asked for the calculation. pct_change is always %; ratio and count are
        dimensionless; aggregations carry a unit only when every input agrees."""
        if operation == "pct_change":
            return "%"
        if operation in ("ratio", "count"):
            return None
        units = {self._input_unit(r) for r in inputs}
        return units.pop() if len(units) == 1 else None

    def register(self, operation: str, inputs: list[str],
                 unit: Optional[str] = None) -> Calculation:
        fn = self._op(operation)
        vals = [self._resolve(r) for r in inputs]
        if operation in OPS:
            unit = self._derive_unit(operation, inputs)   # built-ins: caller unit IGNORED
        # custom ops: the registering code supplies the unit (it wrote the fn)
        try:
            result = fn(*vals)
        except (ArithmeticError, ValueError, TypeError, statistics.StatisticsError) as e:
            raise CalcError(f"{operation}({', '.join(inputs)}) failed: {e}") from e
        if not isinstance(result, (int, float)) or isinstance(result, bool) \
                or math.isnan(result) or math.isinf(result):
            raise CalcError(f"{operation}({', '.join(inputs)}) produced "
                            f"non-finite result {result!r}")
        with self._lock:
            self._counter += 1
            n = self._counter
        calc = Calculation(
            calc_id=f"C-{n:03d}",
            operation=operation,
            inputs=list(inputs),
            result=round(result, 6),
            unit=unit,
            expression=f"{operation}({', '.join(inputs)})",
        )
        self._calcs[calc.calc_id] = calc
        return calc

    def recompute(self, calc_id: str) -> bool:
        """G3: recompute a calculation from its inputs and confirm the stored result.

        Stored results are rounded to 6 decimals at registration, so the fresh
        value is rounded identically before comparison.
        """
        calc = self._calcs.get(calc_id)
        if calc is None:
            return False
        try:
            vals = [self._resolve(r) for r in calc.inputs]
            fresh = round(self._op(calc.operation)(*vals), 6)
        except Exception:
            return False                    # unresolvable/failed recompute = unverified
        return math.isclose(fresh, calc.result, rel_tol=1e-9, abs_tol=1e-9)

    def get(self, calc_id: str) -> Optional[Calculation]:
        return self._calcs.get(calc_id)

    def exists(self, ref: str) -> bool:
        return ref in self._calcs

    def all(self) -> list[Calculation]:
        return list(self._calcs.values())

    def table(self) -> str:
        if not self._calcs:
            return ""
        lines = ["CALCULATIONS"]
        for c in self._calcs.values():
            u = f" {c.unit}" if c.unit else ""
            lines.append(f"{c.calc_id}  [calc]  {c.expression} = {c.result}{u}")
        return "\n".join(lines)
