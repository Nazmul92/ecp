"""Example 3 — Production-shaped BI analyst agent (real LLM, full loop).

Shows the pattern for a real deployment:
  1. Agent plans and calls tools (SQLite here, standing in for your warehouse).
  2. Curated ingestion into an EvidenceStore.
  3. VerifiedPipeline with the observe -> enforce rollout switch.
  4. Proof object persisted to a JSONL audit log.

Backends: set ECP_BACKEND=anthropic (needs ANTHROPIC_API_KEY) or
          ECP_BACKEND=ollama    (local model — the DIEX-style path).

    python examples/03_analyst_agent.py "Why did revenue fall in Q2?"
"""
import json
import os
import sqlite3
import sys
sys.path.insert(0, ".")

from ecp import EvidenceStore, CalcRegistry, VerifiedPipeline, PipelineConfig
from ecp.llm import anthropic_llm, ollama_llm

# ------------------------------------------------------------ demo warehouse
def make_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE sales(quarter TEXT, region TEXT, revenue REAL)")
    con.executemany("INSERT INTO sales VALUES (?,?,?)", [
        ("2026Q1", "ON", 61000), ("2026Q1", "BC", 53000),
        ("2026Q2", "ON", 52000), ("2026Q2", "BC", 48000),
    ])
    con.execute("CREATE TABLE churn(quarter TEXT, churned_accounts INTEGER)")
    con.executemany("INSERT INTO churn VALUES (?,?)",
                    [("2026Q1", 14), ("2026Q2", 31)])
    return con


# --------------------------------------------------------------- agent tools
def run_sql(con: sqlite3.Connection, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def gather_evidence(question: str, con: sqlite3.Connection,
                    store: EvidenceStore, calcs: CalcRegistry) -> None:
    """Deterministic evidence-gathering plan (the DIIS pattern: code drives,
    canonical SQL is the ground truth, and each result is registered with the
    exact query as its ref so the audit trail is replayable).
    """
    q1 = run_sql(con, "SELECT SUM(revenue) FROM sales WHERE quarter='2026Q1'")[0][0]
    q2 = run_sql(con, "SELECT SUM(revenue) FROM sales WHERE quarter='2026Q2'")[0][0]
    e_q1 = store.add_value("Q1 2026 total revenue", q1, "CAD",
                           source_tool="warehouse",
                           ref="SELECT SUM(revenue) FROM sales WHERE quarter='2026Q1'")
    e_q2 = store.add_value("Q2 2026 total revenue", q2, "CAD",
                           source_tool="warehouse",
                           ref="SELECT SUM(revenue) FROM sales WHERE quarter='2026Q2'")
    calcs.register("pct_change", [e_q1.evidence_id, e_q2.evidence_id], unit="%")

    for region in ("ON", "BC"):
        r1 = run_sql(con, f"SELECT revenue FROM sales WHERE quarter='2026Q1' AND region='{region}'")[0][0]
        r2 = run_sql(con, f"SELECT revenue FROM sales WHERE quarter='2026Q2' AND region='{region}'")[0][0]
        a = store.add_value(f"Q1 2026 revenue ({region})", r1, "CAD", source_tool="warehouse")
        b = store.add_value(f"Q2 2026 revenue ({region})", r2, "CAD", source_tool="warehouse")
        calcs.register("pct_change", [a.evidence_id, b.evidence_id], unit="%")

    c1 = run_sql(con, "SELECT churned_accounts FROM churn WHERE quarter='2026Q1'")[0][0]
    c2 = run_sql(con, "SELECT churned_accounts FROM churn WHERE quarter='2026Q2'")[0][0]
    e_c1 = store.add_value("Q1 2026 churned accounts", c1, None, source_tool="warehouse")
    e_c2 = store.add_value("Q2 2026 churned accounts", c2, None, source_tool="warehouse")
    calcs.register("pct_change", [e_c1.evidence_id, e_c2.evidence_id], unit="%")
    # Note: churn correlates with the revenue drop, but nothing here is marked
    # supports_causality — so the model may *infer* (hedged) but cannot *assert*
    # that churn caused the decline. That is G4 working as intended.


# --------------------------------------------------------------------- main
def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Why did revenue fall in Q2 2026?"
    backend = os.environ.get("ECP_BACKEND", "anthropic")
    llm = ollama_llm("llama3.1") if backend == "ollama" else anthropic_llm()

    # Rollout switch: start every deployment in observe mode, watch the
    # would_reject report for a week, then flip to enforce.
    mode = os.environ.get("ECP_MODE", "enforce")

    con = make_db()
    store, = (EvidenceStore(),)
    calcs = CalcRegistry(store)
    gather_evidence(question, con, store, calcs)

    pipeline = VerifiedPipeline(llm=llm, config=PipelineConfig(
        mode=mode, max_repairs=2, render_mode="deterministic",
        audit_path="audit/analyst.jsonl"))

    result = pipeline.run(question, store=store, calcs=calcs)

    print("ANSWER\n------")
    print(result.text)
    print("\nPROVENANCE\n----------")
    for s in result.proof["sentences"]:
        print(f"[{s['type']:<12}] {s['text']}")
        for ref in s["cites"]:
            ev, calc = store.get(ref), calcs.get(ref)
            src = ev.source.ref or ev.source.tool if ev else calc.expression
            print(f"              └─ {ref}: {src}")
    if result.rejected:
        print("\nBLOCKED\n-------")
        for vr in result.rejected:
            print(f"{vr.claim_id}: {vr.repair_hint}")
    if result.observe_report:
        print("\nOBSERVE REPORT (nothing blocked)\n----------------")
        print(json.dumps(result.observe_report, indent=2))


if __name__ == "__main__":
    main()
