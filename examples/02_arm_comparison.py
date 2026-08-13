"""Reproducible two-arm comparison (the source of docs/figures/two-arm-illustration.svg).

    python examples/02_arm_comparison.py          # writes arm_comparison_record.json
    python examples/02_arm_comparison.py --figure # also regenerates the SVG figures

Arm A is a CONSTRUCTED unverified answer: hand-written to be representative of
fluent unverified synthesis, with each error drawn from a benchmark attack
class. It is a product illustration of the verifier's discrimination — NOT
end-to-end benchmark evidence, and NOT a measured hallucination rate for any
model. The scoring of both arms is fully mechanical (this verifier, this file,
this evidence), and the complete record is written to disk.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecp import EvidenceStore, CalcRegistry, VerifiedPipeline, Verifier
from ecp.claims import Claim, AssertedValue
from ecp.llm import MockLLM

QUESTION = "How did sales perform in Q2 and why?"

def world():
    s = EvidenceStore()
    a = s.add_value("Q1 unit sales", 114, "units", source_tool="sales_db")
    b = s.add_value("Q2 unit sales", 100, "units", source_tool="sales_db")
    t = s.add_text("Customer survey (n unknown): several respondents mentioned pricing concerns.",
                   source_tool="survey_tool")
    c = CalcRegistry(s)
    pc = c.register("pct_change", [a.evidence_id, b.evidence_id])
    return s, c, a, b, t, pc

ARM_A_PROSE = ("Q2 sales came in at 100 units, down about 14% from Q1 — a steeper decline than "
               "the roughly 10% the team had forecast. The drop was driven primarily by pricing "
               "concerns, which cost the company around 20 accounts. Surveys confirm pricing was "
               "cited by a majority of respondents. Dips this size recover within 2 quarters; "
               "Q4 should return to the 114-unit level.")

def arm_a_claims(a, b, t, pc):
    AV = AssertedValue
    return [
      ("Q2 sales came in at 100 units,", Claim("A1","finding","Q2 sales were 100 units.",[b.evidence_id],[AV(100,"units",b.evidence_id)]), False),
      ("down about 14% from Q1 —", Claim("A2","finding","Sales fell about 14% from Q1.",[pc.calc_id]), False),
      ("a steeper decline than the roughly 10% the team had forecast.", Claim("A3","finding","The team forecast a roughly 10% decline.",[]), False),
      ("The drop was driven primarily by pricing concerns,", Claim("A4","observation","The drop was driven primarily by pricing concerns.",[t.evidence_id]), False),
      ("which cost the company around 20 accounts.", Claim("A5","finding","Pricing cost around 20 accounts.",[t.evidence_id]), False),
      ("Surveys confirm pricing was cited by a majority of respondents.", Claim("A6","observation","A majority of respondents cited pricing.",[t.evidence_id]), True),
      ("Dips this size recover within 2 quarters; Q4 should return to 114 units.", Claim("A7","finding","Recovery to 114 units within 2 quarters.",[a.evidence_id],[AV(114,"units",a.evidence_id)]), False),
    ]

def main(render_figure=False):
    s, c, a, b, t, pc = world()
    v = Verifier(s, c)
    armA = []
    for frag, cl, tier2_territory in arm_a_claims(a, b, t, pc):
        r = v.verify(cl)
        status = "tier2" if (r.ok and tier2_territory) else ("ok" if r.ok else "bad")
        armA.append({"fragment": frag, "claim": cl.to_dict(), "status": status,
                     "verifier": r.to_dict()})

    s2, c2, a2, b2, t2, pc2 = world()
    ids = dict(a=a2.evidence_id, b=b2.evidence_id, t=t2.evidence_id, pc=pc2.calc_id)
    synth = json.dumps({"claims":[
     {"claim_type":"finding","text":"Q2 sales were 100 units, down from 114 units in Q1.","cites":[ids["a"],ids["b"]],
      "asserted_values":[{"value":100,"unit":"units","from":ids["b"]},{"value":114,"unit":"units","from":ids["a"]}]},
     {"claim_type":"comparison","text":"Sales declined 12.28% quarter over quarter.","cites":[ids["pc"]],
      "asserted_values":[{"value":-12.28,"unit":"%","from":ids["pc"]}]},
     {"claim_type":"observation","text":"Several survey respondents mentioned pricing concerns.","cites":[ids["t"]]},
     {"claim_type":"inference","text":"pricing concerns may be related to the decline","cites":[ids["t"],ids["pc"]],"hedge_level":"soft"},
     {"claim_type":"gap","text":"The evidence does not establish the cause of the decline, the number of affected accounts, or any forecast.","cites":[]},
    ]})
    res = VerifiedPipeline(llm=MockLLM([synth])).run(QUESTION, store=s2, calcs=c2)

    record = {
      "question": QUESTION,
      "evidence": s.snapshot(),
      "arm_a": {"prose": ARM_A_PROSE, "scored": armA,
                "summary": "1 grounded · 5 deterministically rejected · 1 semantic residual (6 of 7 unsupported overall)"},
      "arm_b": {"text": res.text, "proof": res.proof, "metrics": res.metrics},
      "note": ("Arm A is constructed to be representative; scoring is mechanical. "
               "This is an illustration of verifier discrimination, not an "
               "end-to-end benchmark: that requires live models, repeated runs, "
               "independent labels, and completeness/latency/cost measurement."),
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "arm_comparison_record.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"record -> {out}")
    if render_figure:
        from benchmark.figures import main as render
        render()
    print("ARM A:", record["arm_a"]["summary"])
    print("ARM B:", res.text)
    for x in res.proof["sentences"]:
        print(f"  [{x['verification_level']}] {x['text']}")
    return record

if __name__ == "__main__":
    main(render_figure="--figure" in sys.argv)
