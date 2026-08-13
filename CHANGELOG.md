# Changelog

## 0.4.1 — production hardening

No new verification tiers. This release closes the gap between "the checks are
correct" and "the runtime around them is safe to operate," found in a review
round that separated documentation caveats from actual runtime properties.

### Behaviour changes (read before upgrading)

- **Unknown policy values now raise `ConfigError` at construction.** Previously
  any unrecognised `tier2_policy` fell through to the `annotate` branch, so a
  typo silently disabled Tier-2 enforcement while the config still read as
  strict. `mode`, `render_mode`, `tier2_policy`, `prose_policy` and
  `on_evidence_overflow` are validated in both `PipelineConfig` and `Verifier`.
  If you were passing an invalid string, you were not getting the behaviour you
  thought you were; you will now get an exception at startup.
- **`record_hash` is computed after redaction.** It previously covered the
  pre-redaction record, so a persisted redacted record did not verify against
  its own hash — silently. Verification is now
  `sha256(json.dumps(record_without_hash, sort_keys=True, default=str))`.
  Hashes for records written by earlier versions will not match under the new
  recipe.
- **The evidence table is capped by default** at `max_evidence_chars=24000`.
  Set it to `None` for the previous uncapped behaviour. Truncation is reported
  in the table itself, in `result.metrics` and in `proof.evidence_stats`.
- **A calc request always triggers a re-prompt.** When the model returned claims
  *and* `request_calcs` together, the calcs were registered but the claims were
  returned immediately — so the model could never cite the C-ids it had just
  asked for, making the documented "request now, cite in a later pass" contract
  unreachable. Any round that registers a new calculation now re-prompts. This
  costs one extra call in that case and is bounded by `max_calc_rounds`.

### Security

- **The evidence fence is no longer forgeable.** Delimiters carry a fresh random
  nonce per prompt, and tool-supplied text is stripped of fence-like tokens and
  flattened to a single line. Closes two concrete attacks: terminating the
  evidence region early with a literal `END EVIDENCE`, and forging additional
  evidence rows with an embedded newline. This bounds prompt *structure*, not
  prompt *content* — see PRODUCTION.md §9.

### Integration surface (found auditing it for the new guide)

- **`EvidenceStore` and `CalcRegistry` are now serializable.** A `threading.Lock`
  made both unpicklable, so mounting the ECP nodes in a LangGraph app broke every
  checkpointer (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) the moment the
  store travelled through graph state — taking human-in-the-loop, resumability
  and time travel with it. `__getstate__`/`__setstate__` drop the lock and
  recreate it on load; a lock is a runtime primitive, not state.
- `ecp_ingest_node` tolerates `tool_results: None` — a TypedDict-shaped state
  usually has the key present and `None` before the tool node runs, and
  `.get(key, [])` handed that straight to a for-loop.
- `ecp_answer_node` raises naming the state contract instead of a bare
  `KeyError`, since the usual cause is an edge wired past `ecp_ingest_node`.
- `anthropic_llm` defaults updated: `claude-opus-5` (was a stale model) and
  `max_tokens=16000` (was 2000 — enough to truncate a claim set into
  unparseable JSON, which surfaces as an empty answer rather than an error).

### Additions

- Integration guide in the README plus `examples/06_integration.py`, which CI
  runs — every documented pattern executes, including a checkpoint round trip.
- `ecp.tier2` with `llm_judge_backend` (reference LLM judge, fails closed to
  `not_entailed`) and `keyword_stub_backend` (offline demo only, not an
  entailment model). Example: `examples/05_tier2_backend.py`.
- `proof.evidence_stats` and three `metrics` keys recording what the synthesis
  model actually saw.
- `ConfigError` and `EvidenceOverflowError` are exported from `ecp`. The latter
  is not a subclass of the former: a startup `except ConfigError` must not
  swallow a runtime overflow.
- `PRODUCTION.md`: deployment checklist and API stability statement.
- `LICENSE` (MIT), `py.typed`, CI workflow, `CHANGELOG.md`.

### Fixes

- Version metadata is single-sourced from `ecp/version.py`. Evidence records
  previously serialized `ecp_version: "0.2"` regardless of package version;
  CI now fails if the two drift.
- `pipeline.last_audit` is initialised to `None` instead of raising
  `AttributeError` before the first run.
- Malformed or failing calc requests are surfaced on `result.errors` and logged
  instead of being silently swallowed.
- Duplicate calc requests resolve to the existing calc rather than minting a
  second identical `C-id`.
- Audit files are opened with an explicit UTF-8 encoding.
- **Half-year date context no longer false-rejects.** `YEAR_CONTEXT` accepted
  `Q1 2026` and `2026 H1` but not `H1 2026` — `h[12]` was missing from the
  period-before-year branch — so a correct observation was rejected as an
  unbound quantity. This was visible in `examples/01_sales_agent_demo.py`, where
  it also drove the demo into an extra repair round and an exhausted MockLLM.
  Corpus case V11 and two tests now cover both orders.
- **Date literals no longer false-reject.** `2026-08-15` produced three unbound
  numeric tokens, so any claim stating a delivery, invoice or period date was
  rejected — which is most of what an ops or support agent says. ISO, slash and
  month-name dates are now exempt by SPAN (`15` is exempt inside `2026-08-15`,
  not elsewhere in the sentence). Dates remain unverified, and that is now
  stated in the README limitations.
- README test count and DESIGN roadmap headings corrected; the shipped
  `arm_comparison_record.json` regenerated against current code.

### Known limitations unchanged

No shipped entailment model (Tier-2 is bring-your-own). JSONL audit is not
tamper-evident without external chaining, and is single-writer on Windows.
No held-out, repeated, live-model end-to-end benchmark. `examples/04_ecp_real_agent.py`
generates `live_run_record.json` on demand (not checked in): a single transcript
with measured cost, not a benchmark.

## 0.4.0

Value binding, textual unit agreement, scientific-notation rejection, derived
calculation units, qualitative-claim evidence-kind checks, Tier-2 on every
factual claim type, honest verification levels, `PipelineConfig.production`,
pluggable audit sink/redactor, 35-case corpus.

## 0.3.0

Tier-1 numeric grounding on every claim type, mandatory units, percent-token
grounding, small-integer exemption removed, durable proof objects with full
hashes and per-sentence verification levels, `audit_required`, labelled
benchmark harness.
