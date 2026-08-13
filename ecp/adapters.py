"""Framework adapters.

The adapters are plain callables over a state dict, so they need no framework
imports — LangGraph, or any graph runner, can mount them directly:

    graph.add_node("ecp_ingest", ecp_ingest_node())
    graph.add_node("ecp_answer", ecp_answer_node(pipeline))
    graph.add_edge("tools", "ecp_ingest")
    graph.add_edge("ecp_ingest", "ecp_answer")

State contract:
    state["question"]      : str
    state["tool_results"]  : list[{"tool": str, "call_id": str, "result": Any}]
    state["ecp_curate"]    : optional callable(store, tool_results) for curated ingestion
    -> adds state["ecp_store"], state["ecp_result"], state["final_answer"]
"""
from __future__ import annotations

from typing import Callable, Optional

from .pipeline import PipelineResult, VerifiedPipeline
from .store import EvidenceStore


STATE_CONTRACT = ("ECP node state contract — in: 'question' (str), 'tool_results' "
                  "(list of {tool, call_id, result}); out: 'ecp_store', "
                  "'ecp_result', 'final_answer'.")


def ecp_ingest_node(curate: Optional[Callable[[EvidenceStore, list[dict]], None]] = None):
    def node(state: dict) -> dict:
        store = EvidenceStore()
        # `or []` rather than a default: a TypedDict-shaped graph state usually
        # has the key present and set to None before the tool node runs, which
        # `state.get("tool_results", [])` would hand straight to a for-loop.
        tool_results = state.get("tool_results") or []
        curator = state.get("ecp_curate") or curate
        if curator:
            curator(store, tool_results)                 # curated: best label quality
        else:
            for tr in tool_results:                      # auto: works everywhere
                store.ingest_json(tr.get("result"), source_tool=tr.get("tool", "tool"),
                                  call_id=tr.get("call_id", ""))
        return {"ecp_store": store}
    return node


def ecp_answer_node(pipeline: VerifiedPipeline):
    def node(state: dict) -> dict:
        # Name the contract instead of raising a bare KeyError: the usual cause
        # is an edge wired straight from the tool node to ecp_answer, skipping
        # ecp_ingest, and "KeyError: 'ecp_store'" does not say that.
        for key in ("question", "ecp_store"):
            if key not in state:
                raise KeyError(
                    f"{key!r} missing from state. {STATE_CONTRACT} "
                    + ("Did you forget to run ecp_ingest_node first?"
                       if key == "ecp_store" else ""))
        store: EvidenceStore = state["ecp_store"]
        result: PipelineResult = pipeline.run(state["question"], store=store)
        return {"ecp_result": result, "final_answer": result.text}
    return node


def wrap(agent_fn: Callable[..., dict], pipeline: VerifiedPipeline):
    """Convenience wrapper for request/response agents that expose a clean seam.

    agent_fn(question, **kw) must return
        {"tool_results": [...], "curate": optional callable}
    i.e. the agent does planning + tool calls; ECP takes over final answering.

    Note: this fits report-style request/response agents. Streaming or
    interleaved-answer agents should mount the nodes explicitly instead.
    """
    def verified_agent(question: str, **kw) -> PipelineResult:
        out = agent_fn(question, **kw)
        store = EvidenceStore()
        curate = out.get("curate")
        if curate:
            curate(store, out.get("tool_results", []))
            return pipeline.run(question, store=store)
        return pipeline.run(question, tool_results=out.get("tool_results", []))
    return verified_agent
