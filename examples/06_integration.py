"""06 — The integration patterns from the README, executable.

    python examples/06_integration.py

Everything here runs offline with a scripted MockLLM, so it is a test of the
integration surface rather than a demo of a model. It exists because a README
integration guide that nobody runs drifts silently: this file is the same code,
and CI executes it.

Three patterns, in the order most people need them:

    1. graph nodes      ecp_ingest_node + ecp_answer_node    (LangGraph et al.)
    2. curated nodes    same, with real labels and the exact query
    3. wrap()           for request/response agents with a clean seam

The LangGraph section uses plain dicts and direct calls instead of importing
langgraph, because the adapters are plain callables over a state dict -- that is
the point of them. Wiring the same two callables into a real StateGraph is the
README snippet.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecp import PipelineConfig, VerifiedPipeline
from ecp.adapters import ecp_answer_node, ecp_ingest_node, wrap
from ecp.llm import MockLLM

QUESTION = "Where is my order and when does it arrive?"

# The only shape ECP requires from your agent.
TOOL_RESULTS = [{
    "tool": "order_db",
    "call_id": "order-1",
    "result": {"status": "shipped", "carrier": "UPS",
               "delivery_date": "2026-08-15", "packages": 2},
}]


def scripted(*evidence_ids: str) -> MockLLM:
    """A model that cites correctly. Real models need the repair loop; this file
    is testing the integration surface, not the verifier (tests/ does that)."""
    status, date = evidence_ids[0], evidence_ids[1]
    return MockLLM([json.dumps({"claims": [
        {"claim_type": "observation", "text": "The order has shipped.",
         "cites": [status]},
        {"claim_type": "observation",
         "text": "The order is scheduled to arrive on 2026-08-15.",
         "cites": [date]},
    ]})])


def show(name: str, result) -> None:
    print(f"\n--- {name} " + "-" * (58 - len(name)))
    print(result.text)
    print(f"  claims={len(result.verified_claims)} rejected={len(result.rejected)} "
          f"levels={[s['verification_level'] for s in result.proof['sentences']]}")


# ---------------------------------------------------------------- 1. graph nodes
def pattern_graph_nodes() -> None:
    """What ecp_ingest_node / ecp_answer_node do inside any graph runner.

    State contract: question + tool_results in; ecp_store, ecp_result and
    final_answer out. A StateGraph calls these; here we call them directly so
    the example has no framework dependency.
    """
    state = {"question": QUESTION, "tool_results": TOOL_RESULTS}
    state.update(ecp_ingest_node()(state))            # auto-ingestion

    store = state["ecp_store"]
    ids = {e.label: e.evidence_id for e in store.all()}
    pipeline = VerifiedPipeline(llm=scripted(ids["status"], ids["delivery_date"]))

    state.update(ecp_answer_node(pipeline)(state))
    show("1. graph nodes (auto-ingested)", state["ecp_result"])
    print(f"  final_answer key present: {'final_answer' in state}")


# ------------------------------------------------------------ 2. curated nodes
def curate_order(store, tool_results) -> None:
    """Curated ingestion: real labels and the call that produced each fact.

    Auto-ingestion labels this evidence 'status' and 'delivery_date' from the
    JSON path. Curation is what makes a proof object readable a year later.
    """
    for tr in tool_results:
        r, tool, cid = tr["result"], tr["tool"], tr.get("call_id", "")
        store.add_text(r["status"], label="Order fulfilment status",
                       source_tool=tool, call_id=cid,
                       ref="SELECT status FROM orders WHERE id=?")
        store.add_text(r["delivery_date"], label="Carrier delivery estimate",
                       source_tool=tool, call_id=cid,
                       ref="SELECT delivery_date FROM orders WHERE id=?")
        store.add_value("Packages in shipment", r["packages"], "packages",
                        source_tool=tool, call_id=cid,
                        ref="SELECT COUNT(*) FROM packages WHERE order_id=?")


def pattern_curated() -> None:
    state = {"question": QUESTION, "tool_results": TOOL_RESULTS}
    state.update(ecp_ingest_node(curate=curate_order)(state))

    store = state["ecp_store"]
    ids = {e.label: e.evidence_id for e in store.all()}
    pipeline = VerifiedPipeline(
        llm=scripted(ids["Order fulfilment status"], ids["Carrier delivery estimate"]))

    state.update(ecp_answer_node(pipeline)(state))
    show("2. curated ingestion", state["ecp_result"])
    ev = state["ecp_result"].proof["evidence_snapshot"][0]
    print(f"  provenance: {ev['label']!r} <- {ev['source']['tool']} "
          f"{ev['source']['ref']!r}")


# -------------------------------------------------------------------- 3. wrap()
def pattern_wrap() -> None:
    """For agents that already return tool results behind one function call."""
    def my_agent(question: str) -> dict:
        return {"tool_results": TOOL_RESULTS, "curate": curate_order}

    # wrap() builds the store itself, so the ids are known only inside the run;
    # a fixed-id scripted model is fine because curate_order is deterministic.
    pipeline = VerifiedPipeline(llm=scripted("E-001", "E-002"))
    verified_agent = wrap(my_agent, pipeline)
    show("3. wrap() an existing agent", verified_agent(QUESTION))


# ------------------------------------------------------- 4. any llm is a callable
def pattern_any_llm() -> None:
    calls = []

    def my_llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"claims": [
            {"claim_type": "gap",
             "text": "The evidence does not establish a refund policy.",
             "cites": []}]})

    r = VerifiedPipeline(llm=my_llm).run(QUESTION, tool_results=TOOL_RESULTS)
    show("4. any (str)->str callable", r)
    print(f"  backend received {len(calls)} prompt(s); no SDK involved")


def pattern_checkpointed_state() -> None:
    """What a LangGraph checkpointer does to the state between the two nodes.

    MemorySaver / SqliteSaver / PostgresSaver all serialize graph state, so the
    store has to survive a round trip or persistence breaks the moment the ECP
    nodes are mounted. Pickling here stands in for the checkpointer.
    """
    import pickle

    state = {"question": QUESTION, "tool_results": TOOL_RESULTS}
    state.update(ecp_ingest_node(curate=curate_order)(state))

    checkpointed = pickle.loads(pickle.dumps(state))     # <- the checkpointer
    store = checkpointed["ecp_store"]
    ids = {e.label: e.evidence_id for e in store.all()}
    pipeline = VerifiedPipeline(
        llm=scripted(ids["Order fulfilment status"], ids["Carrier delivery estimate"]))

    checkpointed.update(ecp_answer_node(pipeline)(checkpointed))
    show("5. resumed from a checkpoint", checkpointed["ecp_result"])
    print(f"  evidence survived the round trip: {len(store.all())} items, "
          f"ref={store.get(ids['Order fulfilment status']).source.ref!r}")


def main() -> int:
    print("=" * 72)
    print("ECP integration patterns (offline, scripted model)")
    print("=" * 72)
    pattern_graph_nodes()
    pattern_curated()
    pattern_wrap()
    pattern_any_llm()
    pattern_checkpointed_state()

    # Production profile note: without a Tier-2 backend, qualitative prose is
    # downgraded to hedged inference rather than presented as verified fact.
    print("\n--- production profile, no Tier-2 backend " + "-" * 30)
    state = {"question": QUESTION, "tool_results": TOOL_RESULTS}
    state.update(ecp_ingest_node(curate=curate_order)(state))
    ids = {e.label: e.evidence_id for e in state["ecp_store"].all()}
    cfg = PipelineConfig.production(audit_path="audit/integration.jsonl")
    p = VerifiedPipeline(
        llm=scripted(ids["Order fulfilment status"], ids["Carrier delivery estimate"]),
        config=cfg)
    r = p.run(QUESTION, store=state["ecp_store"])
    print(r.text)
    print(f"  prose_policy={cfg.prose_policy!r} -> levels="
          f"{[s['verification_level'] for s in r.proof['sentences']]}")
    print("  (add a Tier-2 backend to have prose judged instead of downgraded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
