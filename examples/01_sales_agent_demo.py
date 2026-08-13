"""Example 1 — Sales analyst agent (fully runnable, no API key needed).

The MockLLM is scripted to do exactly what real models do wrong:
  * invent a number (14%) not supported by evidence,
  * assert causality ("because competitors lowered prices") with zero causal evidence.

Watch the runtime reject both, feed the reasons back, and produce a
grounded answer with a proof object.

Run:  python examples/01_sales_agent_demo.py
"""
import json
import sys
sys.path.insert(0, ".")

from ecp import EvidenceStore, CalcRegistry, VerifiedPipeline, PipelineConfig
from ecp.llm import MockLLM

# ----------------------------------------------------------------- "tools"
def sales_db(quarter: str) -> int:
    return {"Q1": 114, "Q2": 100}[quarter]

def market_api() -> dict:
    return {"sector": "retail hardware", "demand_trend": "declining through H1 2026"}

# --------------------------------------------------- 1. curated ingestion
store = EvidenceStore()
q1 = store.add_value("Q1 2026 unit sales", sales_db("Q1"), "units",
                     source_tool="sales_db", ref="SELECT SUM(qty) WHERE q='2026Q1'")
q2 = store.add_value("Q2 2026 unit sales", sales_db("Q2"), "units",
                     source_tool="sales_db", ref="SELECT SUM(qty) WHERE q='2026Q2'")
mk = store.add_text(market_api()["demand_trend"], label="Sector demand trend",
                    source_tool="market_api", evidence_type="api_response")

calcs = CalcRegistry(store)
pct = calcs.register("pct_change", [q1.evidence_id, q2.evidence_id], unit="%")
# pct_change(114, 100) = -12.28%

# --------------------------------- 2. scripted LLM (bad first draft, fixed repair)
BAD_DRAFT = json.dumps({"claims": [
    {   # fabricated number: evidence says -12.28, model says 14
        "claim_type": "finding",
        "text": "Sales decreased by 14% in Q2.",
        "cites": [pct.calc_id],
        "asserted_values": [{"value": -14, "unit": "%", "from": pct.calc_id}],
        "causal": False,
    },
    {   # unsupported causality — the classic hallucination. The model even
        # mislabels it as non-causal; it is rejected anyway. Note which check
        # fires first: Tier 1 catches it as a qualitative claim citing only a
        # numeric value, before the causal gate ever sees it. Fix the citation
        # and the lexical causal trap catches it on the next pass.
        "claim_type": "finding",
        "text": "Sales fell because competitors lowered their prices.",
        "cites": [q2.evidence_id],
        "asserted_values": [],
        "causal": False,
    },
    {   # this one is fine
        "claim_type": "observation",
        "text": "Sector demand was declining through H1 2026.",
        "cites": [mk.evidence_id],
        "asserted_values": [],
        "causal": False,
    },
]})

REPAIRED = json.dumps({"claims": [
    {
        "claim_type": "finding",
        "text": "Unit sales decreased by 12.28% from Q1 to Q2 2026.",
        "cites": [pct.calc_id],
        "asserted_values": [{"value": -12.28, "unit": "%", "from": pct.calc_id}],
        "causal": False,
    },
    {
        "claim_type": "observation",
        "text": "Sector demand was declining through H1 2026.",
        "cites": [mk.evidence_id],
        "asserted_values": [],
        "causal": False,
    },
    {
        "claim_type": "inference",
        "text": "the demand decline contributed to the drop in sales",
        "cites": [mk.evidence_id, pct.calc_id],
        "asserted_values": [],
        "causal": False,
        "hedge_level": "soft",
    },
    {
        "claim_type": "gap",
        "text": "The available evidence does not establish the specific cause of the decline; competitor pricing data was not retrieved.",
        "cites": [],
        "asserted_values": [],
        "causal": False,
    },
]})

llm = MockLLM([BAD_DRAFT, REPAIRED])

# ------------------------------------------------------------ 3. pipeline
pipeline = VerifiedPipeline(llm=llm, config=PipelineConfig(
    mode="enforce", max_repairs=2, render_mode="deterministic",
    audit_path="audit/demo.jsonl"))

result = pipeline.run("Why did sales fall in Q2 2026?", store=store, calcs=calcs)

# ------------------------------------------------------------- 4. output
print("=" * 72)
print("FINAL ANSWER")
print("=" * 72)
print(result.text)
print()
print("=" * 72)
print(f"REPAIR ITERATIONS: {result.repair_iterations}")
print("=" * 72)
print("WHAT WAS CAUGHT IN THE FIRST DRAFT:")
for stage in pipeline.last_audit["transcript"]:
    if stage["stage"] == "verify_0":
        for r in stage["results"]:
            failed = [t for t in r["tier_results"] if not t["pass"]][0]
            print(f"  {r['claim_id']} rejected [{failed['tier']}]: {failed['reason']}")
print()
print("=" * 72)
print("PROOF OBJECT")
print("=" * 72)
print(json.dumps(result.proof, indent=2))
