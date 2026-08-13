"""Example 2 — LangGraph agent with ECP mounted after the tool loop.

    pip install langgraph
    export ANTHROPIC_API_KEY=...
    python examples/02_langgraph_agent.py

Graph:
    plan -> tools -> ecp_ingest -> ecp_answer -> END

Your existing plan/tool nodes are untouched. ECP replaces only the step
where the LLM would have written the final answer from raw tool output.
"""
import sys
sys.path.insert(0, ".")
from typing import Any, TypedDict

from ecp import EvidenceStore, VerifiedPipeline, PipelineConfig
from ecp.adapters import ecp_ingest_node, ecp_answer_node
from ecp.llm import anthropic_llm

from langgraph.graph import StateGraph, END


# ----------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    question: str
    tool_results: list[dict]
    ecp_store: Any
    ecp_result: Any
    final_answer: str


# ---------------------------------------------------- your existing nodes
def plan_node(state: AgentState) -> dict:
    # In a real agent this is your LLM tool-selection step. Kept static here.
    return {}


def tool_node(state: AgentState) -> dict:
    """Simulated tool executions — swap in your real MCP / DB / API calls."""
    return {"tool_results": [
        {"tool": "sales_db", "call_id": "tc_001",
         "result": {"q1_unit_sales": 114, "q2_unit_sales": 100}},
        {"tool": "market_api", "call_id": "tc_002",
         "result": {"sector": "retail hardware",
                    "demand_trend": "declining through H1 2026"}},
    ]}


# --------------------------------------------------- curated ingestion map
def curate(store: EvidenceStore, tool_results: list[dict]) -> None:
    """Curated ingestion for core tools: good labels = good claims.

    Anything you don't curate can still be auto-ingested; here we curate all.
    """
    for tr in tool_results:
        r, cid = tr["result"], tr["call_id"]
        if tr["tool"] == "sales_db":
            store.add_value("Q1 2026 unit sales", r["q1_unit_sales"], "units",
                            source_tool="sales_db", call_id=cid)
            store.add_value("Q2 2026 unit sales", r["q2_unit_sales"], "units",
                            source_tool="sales_db", call_id=cid)
        elif tr["tool"] == "market_api":
            store.add_text(f'{r["sector"]} demand: {r["demand_trend"]}',
                           label="Sector demand trend",
                           source_tool="market_api", call_id=cid)


# ------------------------------------------------------------------ graph
pipeline = VerifiedPipeline(
    llm=anthropic_llm("claude-sonnet-4-6"),
    config=PipelineConfig(mode="enforce", max_repairs=2,
                          render_mode="polished",
                          audit_path="audit/langgraph.jsonl"),
    polish_llm=anthropic_llm("claude-haiku-4-5-20251001"),   # cheap model polishes
)

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("tools", tool_node)
graph.add_node("ecp_ingest", ecp_ingest_node(curate=curate))
graph.add_node("ecp_answer", ecp_answer_node(pipeline))
graph.set_entry_point("plan")
graph.add_edge("plan", "tools")
graph.add_edge("tools", "ecp_ingest")
graph.add_edge("ecp_ingest", "ecp_answer")
graph.add_edge("ecp_answer", END)
app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"question": "Why did sales fall in Q2 2026?"})
    print("ANSWER:\n", out["final_answer"])
    print("\nPROOF SENTENCES:")
    for s in out["ecp_result"].proof["sentences"]:
        print(f"  [{s['type']:<12}] {s['text']}  <- {s['cites']}")
    if out["ecp_result"].rejected:
        print("\nREJECTED:")
        for vr in out["ecp_result"].rejected:
            print(f"  {vr.claim_id}: {vr.repair_hint}")
