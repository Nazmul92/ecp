# ECP

**A post-tool verification runtime for AI agents.**

ECP sits between tool execution and final-answer generation. It turns tool
results into structured evidence, asks the model for cited claims, verifies
those claims with deterministic code, and returns a final answer with a
machine-readable proof object.

```text
tool results -> evidence store -> claims -> verifier -> answer + proof
```

**Why it exists:** tool calls can be correct while the final LLM answer still
fabricates a number, overstates causality, or blends facts across tool results.
ECP protects that last step.

**Current status:** beta runtime core. Stdlib-only, no required dependencies,
works with any `(prompt: str) -> str` model callable and any agent framework.

## At A Glance

| Capability | What ECP does |
|---|---|
| Evidence grounding | Every factual claim must cite evidence that exists. |
| Number verification | Every visible number must match cited evidence or a recomputed calculation. |
| Calculation safety | Arithmetic is performed by Python code, not by the model. |
| Causal guard | Causal claims require evidence explicitly marked causal. |
| Fail-closed output | Unsupported claims are repaired or dropped before rendering. |
| Auditability | Final answers include a proof object mapping sentences to evidence. |

## The Problem

Suppose a manager asks an AI agent:

> *"How did sales perform in Q2, why did they change, and what will happen next?"*

The agent calls three tools.

### 1. Three tools return correct data

**Tool 1 — Sales database**

```json
{
  "q1_sales": 114,
  "q2_sales": 100,
  "unit": "products"
}
```

**Tool 2 — Customer survey**

```json
{
  "finding": "Several customers mentioned pricing concerns."
}
```

**Tool 3 — Forecast service**

```json
{
  "forecast_available": false
}
```

All three tools worked correctly.

### 2. The normal agent can still hallucinate

A normal agent gives the raw tool results to an LLM:

```text
three tool results  ->  LLM  ->  final answer
```

The LLM might answer:

> "Q2 sales were 100 products, down 14% from Q1. The decline was caused by
> pricing concerns, but sales should recover next quarter."

This sounds reasonable, but most of it is unsupported:

- **100 products** is correct.
- **14%** is wrong. The actual decline is **12.28%**.
- The survey mentioned pricing, but did not prove pricing **caused** the decline.
- The forecast service returned **no forecast**, so "sales should recover" was invented.

The tools were correct. The hallucination happened when the LLM synthesized
their results.

## How ECP Solves It

ECP sits between the tools and the final answer:

```text
Three tools
    ↓
Evidence Store
    ↓
LLM creates cited claims
    ↓
ECP verifies each claim
    ↓
Final answer + proof
```

### 3. ECP converts tool results into evidence

```text
E-001: Q1 sales = 114 products
        source: sales database

E-002: Q2 sales = 100 products
        source: sales database

E-003: Several customers mentioned pricing concerns
        source: customer survey

E-004: Forecast unavailable
        source: forecast service
```

Each piece of information receives an evidence ID and source.

The LLM no longer receives uncontrolled raw results. It receives this organized
evidence table.

### 4. Calculations are performed by code

The LLM should not calculate the percentage itself. ECP calculates it:

```text
C-001 = percentage_change(E-001, E-002)
      = (100 - 114) / 114 × 100
      = -12.28%
```

Now 12.28% becomes a registered, recomputable calculation.

### 5. The LLM produces cited claims

Instead of writing unrestricted prose, the LLM produces structured claims:

```json
{
  "claim_type": "comparison",
  "text": "Sales declined 12.28% from Q1 to Q2.",
  "cites": ["C-001"],
  "asserted_values": [
    {
      "value": -12.28,
      "unit": "%",
      "from": "C-001"
    }
  ]
}
```

It can also create:

```json
{
  "claim_type": "observation",
  "text": "Several customers mentioned pricing concerns.",
  "cites": ["E-003"]
}
```

Because the evidence does not prove the cause or future forecast, the model
should produce a gap claim:

```json
{
  "claim_type": "gap",
  "text": "The evidence does not establish the cause of the decline or provide a future forecast.",
  "cites": []
}
```

### 6. ECP verifies every claim

If the model tries to say:

> "Sales declined 14%."

ECP rejects it because 14% does not match calculation `C-001`.

If the model says:

> "Pricing concerns caused the decline."

ECP rejects it as a factual causal claim because the survey evidence was not
marked as causal evidence.

If the model says:

> "Sales will recover next quarter."

ECP cannot find supporting forecast evidence, so the claim is rejected or
converted into an explicit gap.

### 7. ECP returns the final answer

The final answer becomes:

> "Q2 sales were 100 products, down from 114 in Q1 — a decline of 12.28%.
> Several customers mentioned pricing concerns. The available evidence does not
> establish the cause of the decline or provide a future forecast."

It also returns a proof object (abbreviated):

```json
{
  "sentences": [
    {
      "text": "Q2 sales were 100 products.",
      "cites": ["E-002"]
    },
    {
      "text": "Sales declined 12.28%.",
      "cites": ["C-001"]
    },
    {
      "text": "Several customers mentioned pricing concerns.",
      "cites": ["E-003"]
    }
  ]
}
```

The core idea is simple:

> Tools collect facts. ECP converts those facts into evidence, makes the LLM
> cite that evidence, checks the resulting claims, and only then produces the
> final answer.

ECP does not guarantee that the tools returned true information. It ensures that
the agent's final answer does not silently go beyond the available evidence.

## Install

```bash
pip install -e .
pip install -e ".[langgraph]"  # optional LangGraph example dependency
```

The core package has no runtime dependencies.

## Quick Start

```python
from ecp import VerifiedPipeline
from ecp.llm import anthropic_llm  # or ollama_llm, or any (str) -> str callable

pipeline = VerifiedPipeline(llm=anthropic_llm())

result = pipeline.run(
    "Where is my order?",
    tool_results=[
        {
            "tool": "order_db",
            "call_id": "order-1",
            "result": {
                "status": "shipped",
                "carrier": "UPS",
                "delivery_date": "2026-08-15",
            },
        }
    ],
)

print(result.text)
print(result.proof)
```

Zero-config mode auto-ingests scalar values from tool JSON. It is useful for
getting started. For production paths, curate your important tool outputs with
clear labels, units and provenance.

```python
from ecp import EvidenceStore, CalcRegistry

store = EvidenceStore()
q1 = store.add_value(
    "2026Q1 revenue",
    1240000,
    "USD",
    source_tool="orders_db",
    ref="SELECT revenue_usd FROM orders WHERE quarter='2026Q1'",
)
q2 = store.add_value(
    "2026Q2 revenue",
    1088000,
    "USD",
    source_tool="orders_db",
    ref="SELECT revenue_usd FROM orders WHERE quarter='2026Q2'",
)

calcs = CalcRegistry(store)
calcs.register("pct_change", [q1.evidence_id, q2.evidence_id])

result = pipeline.run(
    "Did revenue change from Q1 to Q2?",
    store=store,
    calcs=calcs,
)
```

## Integration

ECP does not replace your agent. Your planner still plans, your tools still run,
and your framework still owns the workflow. ECP takes over the final answer.

```text
your planner -> your tools -> ECP ingest -> ECP verified answer
```

The only required tool-result shape is:

```python
tool_results = [
    {
        "tool": "order_db",
        "call_id": "order-1",
        "result": {"status": "shipped", "delivery_date": "2026-08-15"},
    }
]
```

Every pattern below is executable in
[examples/06_integration.py](examples/06_integration.py), which CI runs.

### LangGraph

The adapters are plain callables over a state dict, so ECP does not import
LangGraph. Mount the two ECP nodes after your tool loop:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from ecp import PipelineConfig, VerifiedPipeline
from ecp.adapters import ecp_answer_node, ecp_ingest_node
from ecp.llm import anthropic_llm


class AgentState(TypedDict, total=False):
    question: str
    tool_results: list[dict]
    ecp_store: object
    ecp_result: object
    final_answer: str


def tool_node(state: AgentState) -> dict:
    return {
        "tool_results": [
            {
                "tool": "order_db",
                "call_id": "order-1",
                "result": {
                    "status": "shipped",
                    "delivery_date": "2026-08-15",
                },
            }
        ]
    }


pipeline = VerifiedPipeline(
    llm=anthropic_llm(),
    config=PipelineConfig.production(audit_path="audit/ecp.jsonl"),
)

builder = StateGraph(AgentState)
builder.add_node("tools", tool_node)
builder.add_node("ecp_ingest", ecp_ingest_node())
builder.add_node("ecp_answer", ecp_answer_node(pipeline))

builder.add_edge(START, "tools")
builder.add_edge("tools", "ecp_ingest")
builder.add_edge("ecp_ingest", "ecp_answer")
builder.add_edge("ecp_answer", END)

graph = builder.compile()
state = graph.invoke({"question": "Where is my order?"})

print(state["final_answer"])
print(state["ecp_result"].proof)
```

State contract:

| Input | Output |
|---|---|
| `question` | `ecp_store` |
| `tool_results` | `ecp_result` |
| | `final_answer` |

`EvidenceStore`, `CalcRegistry` and `PipelineResult` are serializable, so
LangGraph checkpointers such as `MemorySaver`, `SqliteSaver` and `PostgresSaver`
can persist graph state with ECP mounted.

### Curated Ingestion

Auto-ingestion is the on-ramp. Curated ingestion is the production path:

```python
from ecp.adapters import ecp_ingest_node


def curate_order(store, tool_results):
    for tr in tool_results:
        result = tr["result"]
        store.add_text(
            result["status"],
            label="Order fulfilment status",
            source_tool=tr["tool"],
            call_id=tr.get("call_id", ""),
            ref="SELECT status FROM orders WHERE id=?",
        )
        store.add_text(
            result["delivery_date"],
            label="Carrier delivery estimate",
            source_tool=tr["tool"],
            call_id=tr.get("call_id", ""),
            ref="SELECT delivery_date FROM orders WHERE id=?",
        )


builder.add_node("ecp_ingest", ecp_ingest_node(curate=curate_order))
```

Use `add_value` for numbers and `add_text` for prose. The `ref` field is what
makes a proof object readable months later.

### Existing Agents

If your agent already returns tool results, wrap it:

```python
from ecp.adapters import wrap


def my_agent(question: str) -> dict:
    return {"tool_results": [...], "curate": curate_order}


verified_agent = wrap(my_agent, pipeline)
result = verified_agent("Can I get a refund?")

print(result.text)
print(result.proof)
```

Streaming or interleaved-answer agents should mount the nodes explicitly. ECP
verifies a finished claim set; it cannot honestly stream a sentence before that
sentence has been checked.

### Any Model

ECP only requires a callable:

```python
def my_llm(prompt: str) -> str:
    return call_my_model_api(prompt)


pipeline = VerifiedPipeline(llm=my_llm)
```

Adapters ship for Anthropic and Ollama in `ecp.llm`. Local models, hosted APIs,
vLLM, LM Studio and in-house model servers all work the same way.

## Production Profile

```python
from ecp import PipelineConfig

config = PipelineConfig.production(
    audit_path="audit/ecp.jsonl",
    tier2=None,  # pass an entailment backend for strict semantic prose checks
)
```

The production profile sets:

- `mode="enforce"`
- deterministic rendering
- mandatory audit persistence
- no `model_output`-only evidence
- strict Tier-2 rejection when a backend is present
- qualitative prose downgraded to hedged inference when no Tier-2 backend is present

Deployment details are in [PRODUCTION.md](PRODUCTION.md).

## Benchmarks

The shipped benchmark measures verifier discrimination on a labelled synthetic
36-case corpus. It is a regression gate for the deterministic verifier, not a
live-agent hallucination benchmark.

```bash
python tests/test_ecp.py       # 81 adversarial + hardening cases
python benchmark/harness.py    # labelled corpus; fails on any in-scope miss
```

Current corpus result: **22/22 in-scope bad claims** rejected, 0 false accepts,
0 false rejects — drawn from 33 cases from one synthetic evidence world. Treat it
as a regression gate, not a generalization estimate.

| Metric | Result |
|---|---:|
| In-scope bad claims rejected | 22 / 22 |
| Valid claims accepted | 11 / 11 |
| False accepts | 0 |
| False rejects | 0 |
| Tier-2 residual | 3 cases |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/verifier-confusion-matrix-dark.svg">
  <img alt="Confusion matrix: 22 invalid claims rejected, 11 valid claims accepted, 0 false accepts, 0 false rejects" src="docs/figures/verifier-confusion-matrix.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rejection-coverage-by-tier-dark.svg">
  <img alt="Rejection coverage by tier: structural 2 of 2, value 17 of 17, causal gate 3 of 3, Tier-2 0 of 3" src="docs/figures/rejection-coverage-by-tier.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/two-arm-illustration-dark.svg">
  <img alt="Statement fate per answer: the baseline ships 7 statements of which 6 are unsupported; ECP ships 5, each labeled grounded, citation-resolved or interpretive" src="docs/figures/two-arm-illustration.svg">
</picture>

The two-arm figure is a constructed illustration, not a live benchmark. The
baseline arm is hand-written to show how unsupported claims appear in fluent
prose. The ECP arm is shorter on purpose: enforcement buys grounding by
dropping claims that cannot be proven.

Regenerate figures with:

```bash
python benchmark/figures.py
```

## Examples

| File | Purpose |
|---|---|
| [examples/01_sales_agent_demo.py](examples/01_sales_agent_demo.py) | Offline demo: catches a fabricated number and unsupported causality. |
| [examples/02_langgraph_agent.py](examples/02_langgraph_agent.py) | ECP mounted after a LangGraph tool loop. |
| [examples/03_analyst_agent.py](examples/03_analyst_agent.py) | Production-shaped SQLite analyst agent. |
| [examples/04_ecp_real_agent.py](examples/04_ecp_real_agent.py) | Live local Ollama run: naive answer vs ECP answer. |
| [examples/05_tier2_backend.py](examples/05_tier2_backend.py) | Tier-2 backend wiring and policy modes. |
| [examples/06_integration.py](examples/06_integration.py) | Executable integration guide. |

## Guarantees

Deterministic guarantees, enforced without a model in the loop:

| | Guarantee |
|---|---|
| G1 | Every factual claim cites evidence that exists. |
| G2 | Every number in a claim matches cited evidence or a recomputed calculation. |
| G3 | Every cited calculation recomputes correctly. |
| G4 | Causal claims require evidence explicitly marked causal. |
| G5 | Every rendered claim sentence has retrievable provenance. |

## Non-Guarantees

ECP is intentionally explicit about what it does not prove:

- **Evidence truth.** If a tool returns garbage, ECP can still faithfully cite it.
- **Semantic entailment without Tier 2.** Bring an entailment backend for strict prose support.
- **Novel causal phrasing.** The deterministic causal gate catches known markers, not every possible phrasing.
- **Labeled interpretation.** `inference`, `recommendation` and `gap` claims may carry
  no citations at all. That is deliberate — they are the legal home for speculation,
  and the proof marks them `interpretive` — but it means a final answer can contain
  *"This suggests that…"* or *"Recommendation: …"* with no evidence behind it.
  Verified facts and labeled interpretation are different things; treat them
  differently downstream.
- **Dates.** A date cannot bind to a numeric asserted value, so dates pass through
  unchecked rather than verified. Cite one as text evidence and use Tier 2 if it is
  load-bearing.
- **Word-form numbers.** Digits are verified; *"a fifth"* is not a digit. Spelled-out
  quantities are rejected outright rather than checked.
- **Audit durability.** The JSONL log is not tamper-evident without external chaining,
  and is single-writer on Windows. Use `audit_sink` for multi-worker deployments —
  see [PRODUCTION.md](PRODUCTION.md).
- **Domain judgment.** ECP checks grounding, arithmetic and provenance; it does not decide business correctness.
- **Chat-speed latency.** Verification adds model calls and is best suited to reports, analyses and workflows where wrong claims have cost.

## Rollout

Start in observe mode:

```python
from ecp import PipelineConfig

config = PipelineConfig(mode="observe")
```

Observe mode renders normally but reports what would have been rejected. Run it
against real traffic, fix ingestion labels and units, then switch to
`mode="enforce"`.

## Project Docs

- [PRODUCTION.md](PRODUCTION.md): deployment checklist, audit sinks, evidence budgets and Tier-2 posture.
- [DESIGN.md](DESIGN.md): architecture and verification design.
- [CHANGELOG.md](CHANGELOG.md): release history.

## License

MIT - see [LICENSE](LICENSE).
