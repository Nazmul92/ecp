# ECP v0.4 — Production Design
## A post-tool verification runtime for agentic systems

This is the final design. It supersedes the v0.1 "Evidence Context Protocol" draft:
same architecture underneath, repositioned from *protocol* to *runtime*, with the
Pipeline as the single adoption primitive and observe-mode as the rollout path.
A working reference implementation accompanies this document (`ecp/`, stdlib-only).

---

## 1. Positioning

**What it is:** a verification runtime any agent inserts between tool execution
and final-answer generation.

**What it is not:** a protocol. No tool author has to change anything; ordinary
tool results are ingested through adapters. If external teams adopt the library
and a wire format becomes useful, a spec can be extracted later — from working
code with users, not before.

**The problem statement (unchanged):** tool calling is solved; final-answer
generation is where fabrication happens. Every mainstream framework hands raw
tool output to the LLM and trusts the prose that comes back.

**The core architectural rule:** *the synthesis LLM only ever sees the evidence
table, never raw tool output.* If a fact is not in the EvidenceStore, the model
cannot cite it, and uncited factual claims do not survive verification.

---

## 2. Runtime architecture

```
User Question
     │
Agent Planning + Tool Calls        (your framework — untouched)
     │
┌──────────────────── ECP runtime boundary ────────────────────┐
│ 1 Ingestion        tool results → EvidenceObjects            │
│ 2 CalcRegistry     derived values; LLM requests, code computes│
│ 3 Synthesis        LLM emits structured Claims only          │
│ 4 Verification     Tier0 → Tier1 → causal gate → Tier2       │
│ 5 Repair loop      fail-closed; passing claims frozen        │
│ 6 Rendering        deterministic | polished (re-checked)     │
│ 7 Audit            JSONL provenance record                   │
└───────────────────────────────────────────────────────────────┘
     │
Final Answer + ProofObject
```

Control-flow principle (the DIIS invariant): deterministic code owns the loop;
the LLM is a constrained subroutine invoked at exactly two points — claim
synthesis and optional prose polishing.

---

## 3. Guarantees and non-guarantees

Deterministic guarantees, enforced by code paths with no model in them:

| # | Guarantee | Enforced by |
|---|---|---|
| G1 | Every factual claim cites evidence that exists | Tier 0 |
| G2 | Every number matches cited evidence/calc within tolerance | Tier 1 |
| G3 | Every cited calculation recomputes correctly | Tier 1 + CalcRegistry |
| G4 | Causal claims require evidence marked `supports_causality` | Causal gate + lexical trap |
| G5 | Every sentence has retrievable provenance | ProofObject + audit log |

Explicit non-guarantees, stated in the README: evidence truth (garbage in,
cited garbage out); perfect prose entailment (Tier 2 is probabilistic); domain
correctness of reasoning; framing/emphasis neutrality.

---

## 4. Data model (five objects)

**EvidenceObject** — `evidence_id, kind (value|text|record|causal), label, value,
unit, source{tool, call_id, ref, retrieved_at}, evidence_type, confidence,
supports_causality, raw`. `evidence_type=model_output` marks LLM-derived facts
as second-class; strict profiles reject claims resting only on them. `raw` is
audit-only and never shown to the synthesis model.

**Calculation** — `calc_id, operation, inputs[], result, unit, expression`.
Registered by code or by LLM *request*; always recomputable; citable anywhere an
evidence_id is. Built-in ops: `sum, mean, diff, ratio, pct_change, min, max,
count`, plus `register_op` for custom pure functions.

**Claim** — `claim_id, claim_type, text, cites[], asserted_values[{value, unit,
from}], causal, hedge_level`. The only artifact the synthesis LLM may produce.

**VerificationResult** — `status (verified|rejected|downgraded), tier_results[],
repair_hint, downgraded_to`.

**ProofObject** — sentences with claim_ids and citations, rejected claims,
evidence manifest + hash, verifier config hash, repair iteration count. Ships
alongside the prose; this is the audit surface.

### Claim taxonomy (the crux)

| Type | Verification rule |
|---|---|
| `finding` | Tier-1: every value grounded in cited evidence/calc |
| `observation` | Tier-1 for categoricals; Tier-2 entailment for prose |
| `comparison` | Values from both cited inputs grounded |
| `causal` | Requires cited evidence with `supports_causality: true` |
| `inference` | Allowed; rendered with hedging; marked in provenance |
| `recommendation` | Never rendered as fact; distinct register |
| `gap` | Always legal; zero citations required |

`inference` and `gap` are the pressure valves. Give interpretation a labeled,
legal home and the model stops smuggling it into findings; reward admitting
ignorance and it stops filling gaps with fabrication. The synthesis prompt says
so explicitly.

---

## 5. Verification tiers

**Tier 0 — structural (deterministic).** Schema validity; cited IDs exist;
factual types require citations; model_output-only citation policy.

**Tier 1 — value-level (deterministic).** Three checks: (a) every
`asserted_value` points at a *cited* ref whose value matches within
`rel_tol/abs_tol` (sign flips accepted only when the sentence's direction word agrees with the evidence sign — "decreased by 12%" vs `pct_change = -12` passes, "rose 12%" against the same value is rejected); asserted units must be declared and compatible whenever the cited evidence declares one; percent-suffixed tokens must ground against percent-united evidence;
(b) every number extracted from the text matches some grounded value — with two
pragmatic exemptions learned from implementation: ordinary-language small
Tier-1 applies to EVERY claim type — inference and recommendation may interpret, but any quantity they state must ground. Exemptions: none by default (`small_int_ceiling=0`; raising it is an explicit G2 relaxation) and year tokens only in date context ("in 2026", "FY 2025", "Q1 2026", "H1 2026" — bare 4-digit quantities are verified); (c) every cited calculation
is recomputed from its inputs (G3), with recomputation rounding matched to
storage rounding.

**Causal gate (deterministic).** `causal` claims need causal evidence; plus a
lexical trap — causal markers (*because, due to, led to, caused, drove,
resulted in, attributable to…*) in a `finding/observation/comparison` bounce the
claim with a repair hint suggesting `inference` or `gap`. The trap does not
apply to `inference` claims, which are allowed hedged causal speculation.

**Tier 2 — entailment (probabilistic, optional, pluggable).** Contract:
`(claim_text, [cited_passages]) → entailed | not_entailed | partial`. Backends:
NLI model or LLM judge. Policy: `downgrade` (default — failed observations
become hedged inferences), `reject` (strict profiles), or `annotate`. Tier-2
fallibility never weakens G1–G4, which run regardless.

---

## 6. Repair loop (fail-closed)

Max `max_repairs` rounds (default 2). Each round: rejected claims + machine-
generated repair hints go back to the model; **passing claims are frozen by
semantic content hash** (claim_id excluded — models renumber) and restored if
the model mutates or drops them; results are deduped by the same hash. After
the last round, surviving rejects are omitted and the rendered answer carries an
explicit omission note. Worst case is a shorter, blunter answer — never a
fabricated one. If a repair round returns unparseable output, the loop exits
fail-closed with the current passing set.

---

## 7. Rendering

**deterministic** (default): per-type templates, ordered findings → comparisons
→ observations → inferences → gaps → recommendations; hedge prefixes for
inferences; "Recommendation:" register. The rendered text *is* the verified text.

**polished** (optional): a second LLM call (can be a cheaper model) rewrites the
verified statements under a no-new-facts/no-new-causality constraint. The final
prose is then re-checked — Tier-1 number survival and the causal-marker trap —
and any violation falls back to deterministic rendering. Polish can never
weaken the guarantees.

---

## 8. The one primitive: VerifiedPipeline

```python
result = VerifiedPipeline(llm=..., config=PipelineConfig(...)).run(
    question, store=store, calcs=calcs)        # curated (recommended)
result = pipeline.run(question, tool_results=tool_results)   # zero-config
```

`llm` is any `(str) -> str` callable — no SDK lock-in; adapters ship for
Anthropic, Ollama (fully local, the DIEX path), and a scripted MockLLM for tests.
`PipelineResult` carries `text, proof, verified_claims, rejected,
repair_iterations, observe_report`.

**Calc-request loop:** the synthesis model may return `request_calcs` instead of
doing arithmetic; the runtime executes them, extends the calc table, and
re-prompts (bounded by `max_calc_rounds`).

### Configuration profiles

```
strict      tier2_policy=reject,  allow_model_output_evidence=False, deterministic render
balanced    tier2_policy=downgrade, deterministic render               (default)
NOTE: deterministic rendering is the production default. polished prose is
re-checked (numbers / causal markers / word-numbers) but is not a verbatim
claim transcript; proofs carry prose_is_verbatim_claims=false and auditors
must reconcile against proof.sentences. Use deterministic where G5 alignment
is required.
observe     mode=observe — nothing blocked; observe_report shows would-be rejections
```

All policy fields are validated at construction (`ConfigError` on an unknown
value). An unrecognised `tier2_policy` previously fell through to the annotate
branch, i.e. a typo silently disabled Tier-2 enforcement while the config still
read as strict. Fields with no safe default — `max_evidence_chars`,
`on_evidence_overflow` — are documented in PRODUCTION.md rather than guessed at
here, because the right value depends on the deployment's context window and
cost ceiling.

Observe mode is the adoption path: run a week in observe, review the
would-reject report, then flip to enforce. Warn first, block later — the linter
playbook.

---

## 9. Integration patterns

**LangGraph (explicit nodes — primary pattern):**

```
plan → tools → ecp_ingest_node(curate) → ecp_answer_node(pipeline) → END
```

The adapter nodes are plain callables over the state dict — no framework
imports in the library — so they mount on LangGraph or any graph runner.
State contract: `question`, `tool_results` in; `ecp_store`, `ecp_result`,
`final_answer` out.

**wrap() (convenience, honestly scoped):** `ecp.adapters.wrap(agent_fn, pipeline)`
for request/response agents exposing a clean seam (question in → tool_results
out). Streaming or interleaved-answer agents should mount the nodes explicitly;
framework invisibility is not promised in general.

**Ingestion modes:** curated (`add_value/add_text` with real labels and the
exact SQL/ref — required for core tools, since label quality drives claim
quality) and auto (`ingest_json`, every scalar leaf becomes evidence with a
path label — the zero-config on-ramp and the long tail).

**MCP relationship:** none required. ECP consumes whatever MCP returns. Native
ECP-shaped envelopes import losslessly if a tool ever emits them — soft upgrade
path, not a dependency.

---

## 10. Audit

Every run appends one JSONL record: proof object plus a stage transcript
(synthesis output, per-round verification results). Any sentence in any answer
is traceable to the tool call and query that grounded it, months later
(G5). `pipeline.last_audit` exposes the same record in-process (`None` before
the first run).

The record is hashed *after* redaction, so `record_hash` verifies against
exactly what was persisted. Scope of that hash: it detects in-place edits to a
record. It is not tamper-evidence for the log — records are not chained, so
whole-record deletion or reordering is invisible. JSONL is also single-writer on
Windows, where `fcntl` is unavailable and no lock is taken. Both limits are
reasons `audit_sink` exists; see PRODUCTION.md §4.

---

## 11. Benchmark methodology (measure before you claim)

Corpus: analytical questions over a known database with canonical-SQL ground
truth (the existing 100-query Northwind stress-test set fits). Two arms: raw
agent vs the same agent with ECP in enforce mode. Scoring instrument: the
Verifier itself in observe mode, applied identically to both arms.
Metrics: fabricated numeric facts, unsupported causal claims, incorrect
calculations, provenance coverage, and answer completeness (the recall cost of
enforcement — report it; enforcement is not free). Expect a nonzero residual in
the ECP arm (word-form numbers, extraction edges) and report it. No number
appears in any public material before it is measured: an unverified benchmark
claim in a verification library is a self-refutation.

---

## 12. Roadmap

**v0.4.1 (this release):** store, calc registry, claims, Tier 0/1 + causal gate
across all claim types, strict unit/percent grounding and value binding, repair
loop, deterministic + polished rendering, observe mode, durable audit records
(evidence + calc snapshots, full SHA-256 hashes, per-record hash over the
persisted redacted record, fail-closed audit_required mode), per-sentence
verification levels, labelled 36-case benchmark harness, adapters, 80-case
adversarial + hardening test suite. Production hardening: validated policy
enums, nonced/sanitized evidence fence, configurable evidence budget,
calc-request re-prompt, `ecp.tier2` reference backends, CI gate, MIT license,
PRODUCTION.md.

**v0.5:** Tier-2 NLI backend (local DeBERTa-class model) shipped rather than
bring-your-own; number-word normalizer for Tier-1; held-out repeated live-model
benchmark publication; `ecp audit` CLI; audit record chaining.

**Later, only with ≥3 external adopters:** wire-format spec extraction;
non-Python SDKs; native-envelope tool guidance.

## 13. Known limitations

Evidence truth is out of scope. Tier-1 extraction misses word-form numbers.
Tier-2 is probabilistic and defaults to downgrade. Latency +1–2 LLM calls —
suited to reports, not chat-speed UX. ECP governs grounding, not framing:
adversarial prompts can still shape which verified facts get emphasized.
