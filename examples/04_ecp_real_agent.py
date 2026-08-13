"""04 — A real multi-tool agent, run twice: naive vs. ECP-governed.

This is not a scripted demo. It stands up real data in SQLite, exposes three
tools, lets a *live* local LLM (Ollama, default phi3.5) plan which tools to
call, executes them, and then synthesizes an answer TWO ways:

    (A) naive  : raw tool output -> LLM -> prose            (what every framework does)
    (B) ecp    : tool output -> EvidenceStore -> VerifiedPipeline

Run:
    ollama serve            # in another terminal
    ollama pull mistral:7b
    ECP_MODEL=mistral:7b python examples/04_ecp_real_agent.py

The point: a 7B model *will* fabricate a number or invent causality in arm (A).
Watch arm (B) either ground it or refuse.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

# keep prints ASCII-safe on a Windows console; force UTF-8 for any stray glyphs
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import functools
print = functools.partial(print, flush=True)   # stream progress under a slow local LLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib.request

from ecp import VerifiedPipeline, PipelineConfig, EvidenceStore, CalcRegistry

MODEL = os.environ.get("ECP_MODEL", "phi3.5:latest")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# A 7B model on CPU can spend several minutes on the ECP synthesis call: that
# prompt carries the whole evidence table and asks for ~900 tokens of JSON,
# where the naive arm asks for 350 tokens of prose. Raise for slow hardware.
TIMEOUT = int(os.environ.get("ECP_HTTP_TIMEOUT", "280"))


# Every model call is recorded here so the run record carries real latency and
# token counts rather than an assertion that ECP is "1-2 extra calls".
CALL_LOG: list[dict] = []


def ollama(*, json_mode: bool, num_predict: int = 400, stage: str = "unknown"):
    """Local Ollama caller with output cap + deterministic temp — keeps a small
    CPU model inside a sane latency budget. Returns a (prompt:str)->str callable."""
    def call(prompt: str) -> str:
        body = {"model": MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0, "num_predict": num_predict}}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(f"{HOST}/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read())
        CALL_LOG.append({
            "stage": stage,
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "prompt_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
            "prompt_chars": len(prompt),
        })
        return payload.get("response", "")
    return call


def call_summary(stages: tuple[str, ...]) -> dict:
    """Aggregate CALL_LOG for the given stages. Local models have no per-token
    price, so cost is reported in tokens; multiply by your provider's rate."""
    rows = [c for c in CALL_LOG if c["stage"] in stages]
    return {
        "llm_calls": len(rows),
        "wall_seconds": round(sum(c["wall_seconds"] for c in rows), 2),
        "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in rows),
        "output_tokens": sum(c["output_tokens"] or 0 for c in rows),
        "calls": rows,
    }


plan_llm = ollama(json_mode=True, num_predict=900, stage="plan")
prose_llm = ollama(json_mode=False, num_predict=350, stage="naive")    # arm A
json_llm = ollama(json_mode=True, num_predict=900, stage="ecp")        # arm B


# ─────────────────────────────────────────────────────────── data layer (real)
def build_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE orders(quarter TEXT, revenue_usd INTEGER, units INTEGER);
        INSERT INTO orders VALUES
            ('2026Q1', 1240000, 8200),
            ('2026Q2', 1088000, 7100);

        CREATE TABLE support(quarter TEXT, category TEXT, tickets INTEGER);
        INSERT INTO support VALUES
            ('2026Q1','billing',120), ('2026Q1','shipping',340), ('2026Q1','quality',55),
            ('2026Q2','billing',138), ('2026Q2','shipping',900), ('2026Q2','quality',61);

        CREATE TABLE market_notes(period TEXT, note TEXT, confirmed_causal INTEGER);
        INSERT INTO market_notes VALUES
            ('2026H1','A regional carrier strike disrupted outbound shipping through most of Q2 2026 (confirmed logistics incident).', 1),
            ('2026H1','Marketing ran a routine spring email campaign in Q2 2026.', 0);
        """
    )
    return con


# ─────────────────────────────────────────────────────────────────────── tools
def tool_query_orders(con, quarter: str) -> dict:
    row = con.execute(
        "SELECT revenue_usd, units FROM orders WHERE quarter=?", (quarter,)
    ).fetchone()
    if not row:
        return {"quarter": quarter, "found": False}
    return {"quarter": quarter, "revenue_usd": row[0], "units": row[1],
            "_sql": f"SELECT revenue_usd, units FROM orders WHERE quarter='{quarter}'"}


def tool_query_support(con, quarter: str) -> dict:
    rows = con.execute(
        "SELECT category, tickets FROM support WHERE quarter=?", (quarter,)
    ).fetchall()
    return {"quarter": quarter, "tickets_by_category": {c: t for c, t in rows},
            "_sql": f"SELECT category, tickets FROM support WHERE quarter='{quarter}'"}


def tool_market_notes(con, period: str) -> dict:
    rows = con.execute(
        "SELECT note, confirmed_causal FROM market_notes WHERE period=?", (period,)
    ).fetchall()
    return {"period": period,
            "notes": [{"text": n, "confirmed_causal": bool(c)} for n, c in rows]}


TOOLS = {
    "query_orders":  (tool_query_orders,  "revenue_usd and units for a quarter, e.g. '2026Q1'"),
    "query_support": (tool_query_support, "support ticket counts by category for a quarter"),
    "market_notes":  (tool_market_notes,  "notable market/logistics events for a period, e.g. '2026H1'"),
}

DEFAULT_PLAN = [
    {"tool": "query_orders",  "args": {"quarter": "2026Q1"}},
    {"tool": "query_orders",  "args": {"quarter": "2026Q2"}},
    {"tool": "query_support", "args": {"quarter": "2026Q2"}},
    {"tool": "market_notes",  "args": {"period": "2026H1"}},
]


# ───────────────────────────────────────────────────────── agent: plan + execute
def plan_tools(question: str) -> list[dict]:
    desc = "\n".join(f"- {name}({', '.join(('quarter' if 'quarter' in d else 'period',))}): {d}"
                     for name, (_, d) in TOOLS.items())
    prompt = (
        "You are the planning stage of an analytics agent.\n"
        f"Tools:\n{desc}\n\n"
        f"Question: {question}\n\n"
        'Respond ONLY as JSON: {"calls":[{"tool":"query_orders","args":{"quarter":"2026Q1"}}]}'
    )
    try:
        planned = json.loads(plan_llm(prompt)).get("calls", [])
        planned = [c for c in planned if c.get("tool") in TOOLS and c.get("args")]
    except Exception:
        planned = []
    # A small local model routinely drops a call or omits an arg. Union the
    # model's plan with the default coverage plan (dedup) so evidence is complete
    # -- the agent still plans; we just guarantee the data the question needs.
    merged, seen = [], set()
    for c in planned + DEFAULT_PLAN:
        key = (c["tool"], json.dumps(c.get("args", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged


def execute(con, calls: list[dict]) -> list[dict]:
    results = []
    for i, c in enumerate(calls):
        fn = TOOLS[c["tool"]][0]
        try:
            res = fn(con, **c.get("args", {}))
        except TypeError:
            continue
        results.append({"tool": c["tool"], "call_id": f"call-{i}", "result": res})
    return results


# ────────────────────────────────────────────────────── arm A: the naive agent
def naive_answer(question: str, tool_results: list[dict]) -> str:
    prompt = (
        "You are an analytics assistant. Using ONLY the tool results below, "
        "answer the question in 3-4 sentences. Include the percent change in revenue.\n\n"
        f"QUESTION: {question}\n\nTOOL RESULTS:\n{json.dumps(tool_results, indent=2)}\n\nANSWER:"
    )
    return prose_llm(prompt).strip()


# ───────────────────────────────────────────────── arm B: ECP curated ingestion
def curate(store: EvidenceStore, tool_results: list[dict]) -> None:
    """Map raw tool output into well-labelled evidence. This is the 10-minute
    upgrade the README recommends: real labels + the exact SQL as provenance."""
    for tr in tool_results:
        tool, cid, res = tr["tool"], tr["call_id"], tr["result"]
        if tool == "query_orders" and res.get("found", True) and "revenue_usd" in res:
            q = res["quarter"]
            store.add_value(f"{q} revenue", res["revenue_usd"], "USD",
                            source_tool="orders_db", call_id=cid, ref=res.get("_sql", ""))
            store.add_value(f"{q} units", res["units"], "units",
                            source_tool="orders_db", call_id=cid, ref=res.get("_sql", ""))
        elif tool == "query_support":
            q = res["quarter"]
            for cat, n in res["tickets_by_category"].items():
                store.add_value(f"{q} {cat} support tickets", n, "tickets",
                                source_tool="support_db", call_id=cid, ref=res.get("_sql", ""))
        elif tool == "market_notes":
            for note in res["notes"]:
                store.add_text(note["text"], source_tool="market_intel", call_id=cid,
                               evidence_type="document_chunk",
                               supports_causality=note["confirmed_causal"])


def ecp_answer(question: str, tool_results: list[dict]):
    store = EvidenceStore()
    curate(store, tool_results)
    calcs = CalcRegistry(store)
    pipeline = VerifiedPipeline(
        llm=json_llm,
        config=PipelineConfig(mode="enforce", render_mode="deterministic",
                              max_repairs=1, max_calc_rounds=1),
    )
    return pipeline.run(question, store=store, calcs=calcs), store


# ─────────────────────────────────────────────────────────────────────── driver
def main() -> None:
    question = "Did revenue change from Q1 to Q2 2026, and what might explain it?"
    con = build_db()

    print("=" * 72)
    print(f"MODEL: {MODEL}   (live, local via Ollama)")
    print(f"QUESTION: {question}")
    print("=" * 72)

    calls = plan_tools(question)
    print("\n[1] AGENT PLANNED THESE TOOL CALLS:")
    for c in calls:
        print(f"    {c['tool']}({c.get('args', {})})")
    tool_results = execute(con, calls)
    print(f"\n[2] EXECUTED {len(tool_results)} tools -> raw evidence collected.")

    # ---- ground truth, computed here in plain code so we can audit both arms ----
    r1 = con.execute("SELECT revenue_usd FROM orders WHERE quarter='2026Q1'").fetchone()[0]
    r2 = con.execute("SELECT revenue_usd FROM orders WHERE quarter='2026Q2'").fetchone()[0]
    truth_pct = (r2 - r1) / r1 * 100
    print(f"\n[GROUND TRUTH] revenue {r1} -> {r2}  =  {truth_pct:+.2f}%")

    print("\n" + "-" * 72)
    print("ARM A - NAIVE AGENT (raw tool output -> LLM prose, no verification)")
    print("-" * 72)
    naive_text = ""
    try:
        naive_text = naive_answer(question, tool_results)
        print(naive_text)
    except Exception as e:
        print(f"[naive agent errored: {e}]")

    print("\n" + "-" * 72)
    print("ARM B - ECP-GOVERNED AGENT (evidence store -> verified pipeline)")
    print("-" * 72)
    record = {
        "model": MODEL, "backend": "ollama (local)", "question": question,
        "ground_truth": {"q1_revenue_usd": r1, "q2_revenue_usd": r2,
                         "revenue_pct_change": round(truth_pct, 4)},
        "planned_calls": calls,
        "note": ("A single reproducible run of one small local model. It is a "
                 "transcript with measured latency and token counts, NOT a "
                 "benchmark: one question, one model, one seed, no repeats, no "
                 "held-out corpus. Do not quote a hallucination rate from it."),
    }
    try:
        result, store = ecp_answer(question, tool_results)
        print("VERIFIED ANSWER:")
        print(result.text or "[no claim survived verification]")
        print(f"\nrepair iterations: {result.repair_iterations}")
        print(f"verified claims:   {len(result.verified_claims)}")
        print(f"rejected claims:   {len(result.rejected)}")
        rejections = []
        for vr in result.rejected:
            # tier_results holds TierResult dataclasses, not dicts
            failed = next((t for t in vr.tier_results if not t.passed), None)
            if failed:
                print(f"   REJECTED [{failed.tier}]: {failed.reason}")
                rejections.append({"tier": failed.tier, "reason": failed.reason,
                                   "repair_hint": vr.repair_hint})
        print("\nPROOF (per-sentence provenance):")
        for s in result.proof["sentences"]:
            print(f"   {s['claim_id']} [{s['type']}] cites={s['cites']}: {s['text']}")
        print(f"\nevidence_hash={result.proof['evidence_hash']} "
              f"config_hash={result.proof['verifier_config_hash']}")
        record["arm_b_ecp"] = {"text": result.text, "proof": result.proof,
                               "metrics": result.metrics, "errors": result.errors,
                               "rejections": rejections}
    except Exception as e:
        import traceback
        print(f"[ecp agent errored: {e}]")
        traceback.print_exc()
        record["arm_b_ecp"] = {"error": str(e)}

    # ------------------------------------------------------ cost and latency
    record["arm_a_naive"] = {"text": naive_text, "cost": call_summary(("naive",))}
    record["arm_b_ecp"]["cost"] = call_summary(("ecp",))
    record["planning_cost"] = call_summary(("plan",))

    a, b = record["arm_a_naive"]["cost"], record["arm_b_ecp"]["cost"]
    print("\n" + "-" * 72)
    print("COST AND LATENCY (shared planning + tool execution excluded)")
    print("-" * 72)
    print(f"  arm A naive : {a['llm_calls']} call(s)  {a['wall_seconds']:>7.2f}s  "
          f"{a['prompt_tokens']:>6} in / {a['output_tokens']:>5} out")
    print(f"  arm B ecp   : {b['llm_calls']} call(s)  {b['wall_seconds']:>7.2f}s  "
          f"{b['prompt_tokens']:>6} in / {b['output_tokens']:>5} out")
    if a["wall_seconds"]:
        print(f"  ECP overhead: {b['wall_seconds'] / a['wall_seconds']:.1f}x wall time, "
              f"{b['llm_calls'] - a['llm_calls']:+d} call(s)")
    print("  (local CPU model; absolute numbers are hardware-bound and not "
          "representative of a hosted model)")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "live_run_record.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=str)
    print(f"\nrun record -> {out}")


if __name__ == "__main__":
    main()
