# Running ECP in production

This document is the short version of "what you must decide before this library
is load-bearing." It assumes you have read the guarantees and non-guarantees in
[README.md](README.md).

## API stability

ECP is `0.x`. The public API may change between minor versions, and proof-object
fields may be added (existing fields are not removed without a version bump note
in the release).

**Intended stable surfaces** — changes here will be called out explicitly:

| Surface | Status |
|---|---|
| `VerifiedPipeline.run()` | stable |
| `EvidenceStore.add_value` / `add_text` / `ingest_json` | stable |
| `CalcRegistry.register` / `register_op` | stable |
| `PipelineConfig.production()` | stable |
| `PipelineResult` fields `text`, `proof`, `metrics`, `errors` | stable |
| Tier-2 backend contract `(claim, passages) -> verdict` | stable |
| Everything under `ecp.verifier` internals (`_tier1`, regexes) | unstable |
| Proof object shape | additive-only |

Pin an exact version if you depend on proof-object parsing:
`ecp-runtime==0.4.1`.

---

## Checklist

### 1. Use the production profile

```python
from ecp import PipelineConfig
cfg = PipelineConfig.production(audit_path="audit.jsonl", tier2=judge)
```

This sets `mode="enforce"`, deterministic rendering, `audit_required=True`,
`allow_model_output_evidence=False`, and `tier2_policy="reject"`. If you build a
config by hand instead, know what each of those is doing before you drop one.

Policy strings are validated at construction: an unrecognised value raises
`ConfigError` at startup rather than silently selecting the permissive branch.
If your config loader passes strings from YAML or env vars, that error is your
early-warning system — do not catch and default it.

### 2. Curate ingestion for your core tools

Auto-ingestion (`ingest_json`, or passing `tool_results=` to `run()`) is an
onboarding path, not a destination. It labels evidence with JSON paths, and
label quality drives claim quality. For every tool that matters, use
`add_value` / `add_text` with a real label, the unit, and the exact query:

```python
store.add_value("Q2 2026 unit sales", 100, "units",
                source_tool="sales_db", ref="SELECT SUM(qty) WHERE q='2026Q2'")
```

The `ref` is what makes a proof object readable a year later.

### 3. Keep deterministic rendering for anything regulated

`render_mode="polished"` runs a second model over the verified statements. The
prose is re-checked for numbers, causal markers and word-numbers, but it is not
a verbatim transcript of the verified claims — proofs carry
`prose_is_verbatim_claims: false` and an auditor has to reconcile against
`proof.sentences`. For regulated reporting, keep `deterministic`.

### 4. Provide a durable audit sink

`audit_path` writes JSONL. That is a development and single-node convenience,
not an audit system. Know the three limits:

- **Concurrency.** On POSIX, concurrent workers are serialised with `flock`. On
  **Windows there is no locking at all** — `fcntl` does not exist. Audit records
  carry a full evidence snapshot and stage transcript, so they are large enough
  that concurrent appends can interleave and corrupt lines. Treat `audit_path`
  as **single-writer only on Windows**; for multi-worker Windows deployments use
  `audit_sink`.
- **Tamper evidence.** Each record hashes itself, which detects in-place edits.
  Records are **not chained**, so deleting or reordering whole records leaves no
  trace. If you need that, ship to append-only storage (S3 Object Lock, a WORM
  volume, your SIEM) and chain or timestamp there.
- **Growth.** Records are large and unbounded in count. Rotate or expire them.

```python
def sink(record: dict) -> None:
    db.execute("INSERT INTO ecp_audit (answer_id, record) VALUES (%s, %s)",
               (record["proof"]["answer_id"], json.dumps(record)))

cfg = PipelineConfig.production(audit_path=None, tier2=judge)
cfg.audit_sink = sink        # audit_required=True makes a sink failure fail the run
```

Verifying a record hash (note: hash covers the **redacted** record, i.e. exactly
what was persisted):

```python
stored = record.pop("record_hash")
assert stored == hashlib.sha256(
    json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
```

### 5. Redact before you persist

`audit_redactor` runs before hashing and before the sink. Evidence values and
the question are both in the record; if either can contain personal or
regulated data, redact there rather than downstream.

### 6. Decide your Tier-2 posture

This is the decision people skip. Deterministic tiers prove provenance and
arithmetic; they do **not** prove that a sentence is semantically supported.
Pick one, explicitly:

| Posture | Config | What you get |
|---|---|---|
| No Tier-2, prose allowed | `prose_policy="allow"` | qualitative prose presented as verified on citation resolution alone. Weakest. |
| No Tier-2, prose hedged | `prose_policy="downgrade"` (production default) | qualitative factual prose is downgraded to hedged inference |
| Tier-2, advisory | `tier2=judge, tier2_policy="downgrade"` | failed entailment becomes a hedged inference |
| Tier-2, strict | `tier2=judge, tier2_policy="reject"` | failed entailment blocks the claim |

See [examples/05_tier2_backend.py](examples/05_tier2_backend.py) for the wiring
and the difference between all three, and `ecp.tier2.llm_judge_backend` for a
reference judge. Backends fail closed: an erroring or unparseable judge returns
`not_entailed`.

Note the asymmetry that is deliberate: a failed entailment on a `causal` claim
is **never** advisory. It is rejected regardless of policy.

### 7. Set the evidence budget

`max_evidence_chars` (default 24000) caps the rendered evidence table. The table
is the one unbounded input to the prompt — auto-ingesting a large tool result
can otherwise blow the context window or the bill.

- `on_evidence_overflow="truncate"` (default): omitted rows are announced inside
  the table so the model can answer with a `gap` claim, counted in
  `result.metrics["evidence_truncated"]`, and recorded in
  `proof.evidence_stats`.
- `on_evidence_overflow="error"`: raises `EvidenceOverflowError`. Use this where
  a partial answer is worse than no answer. It is deliberately not a
  `ConfigError` — the config is valid, this run's evidence just did not fit — so
  wrapping startup in `except ConfigError` will not swallow it.

Set the budget from your model's context window and your cost ceiling; the
default is a starting point, not a recommendation for your deployment.

### 8. Watch the metrics

`result.metrics` is built for a dashboard. The two that matter operationally:

- `rejection_rate` climbing means your evidence and your questions have drifted
  apart — usually a retrieval problem, not a verifier problem.
- `evidence_truncated` firing means answers are being formed on a subset.

`result.errors` is non-empty whenever a backend or parse failure was swallowed
into fail-closed behaviour. An empty answer with a populated `errors` list is
the system working, but you still want it alarmed.

### 9. Understand what the fence does and does not do

Evidence is wrapped in `BEGIN EVIDENCE <nonce>` / `END EVIDENCE <nonce>` with a
fresh random nonce per prompt, and tool text is stripped of fence-like tokens
and flattened to one line. That closes prompt-*structure* attacks: tool output
cannot terminate the evidence region early or forge an extra evidence row.

It does not make injected text harmless. Content inside the fence can still
argue with the model, and ECP's answer is only as good as your evidence.
What limits the damage is that any resulting claim still has to cite real
evidence and survive Tier 0/1 and the causal gate. Treat tool output as
untrusted input everywhere else in your stack too.

### 10. Latency and cost

ECP adds 1–2 LLM calls over a naive agent (synthesis, plus repair rounds only
when claims fail), and one more per Tier-2 judged claim if you enable a judge.
That is the right trade for reports, filings and analyses. It is the wrong trade
for chat-speed UX.

---

## Roll out in observe mode first

```python
PipelineConfig(mode="observe")   # nothing blocked; result.observe_report shows would-be rejections
```

Run a week. Read the would-reject report. Fix the evidence-quality problems it
exposes — most first-week rejections are missing units and uncurated labels, not
model misbehaviour. Then flip to `enforce`. This is the linter playbook, and it
is the only rollout we recommend.
