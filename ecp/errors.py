"""Shared error types and config validation.

Lives in its own module so both ``verifier`` and ``pipeline`` can raise the
same error without an import cycle.

Why this exists: every policy field in this library is a plain string, and an
unrecognised value used to fall through to the most permissive branch. In a
verification runtime a typo that silently disables a check is the worst
available failure mode, so unknown values are rejected at construction time.
"""
from __future__ import annotations

from typing import Iterable


class ConfigError(ValueError):
    """A configuration value is not one of the values the runtime understands.

    Raised at construction time (``PipelineConfig``, ``Verifier``) rather than
    at verification time, so a misconfigured deployment fails on startup
    instead of quietly passing claims in production.
    """


class EvidenceOverflowError(RuntimeError):
    """The evidence table exceeded ``max_evidence_chars`` and the configured
    policy is ``on_evidence_overflow="error"``.

    Deliberately NOT a ConfigError: the configuration is valid, this run's
    evidence simply did not fit. Callers that wrap startup in
    ``except ConfigError`` should not also swallow a runtime overflow.
    """


def validate_choice(field: str, value: object, allowed: Iterable[str]) -> str:
    """Return ``value`` if it is in ``allowed``; raise ConfigError otherwise.

    Matching is exact and case-sensitive on purpose: silently accepting
    ``"Enforce"`` would mean the set of valid values differs between fields
    that happen to fail safe and fields that fail open.
    """
    allowed = sorted(allowed)
    if value not in allowed:
        raise ConfigError(
            f"{field}={value!r} is not a recognised value; "
            f"expected one of {', '.join(repr(a) for a in allowed)}"
        )
    return str(value)
