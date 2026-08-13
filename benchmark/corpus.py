"""Labelled adversarial corpus for verifier evaluation.

Every case constructs its own evidence world and claim, and carries a label
assigned BY CONSTRUCTION — the label encodes what we know to be true about
the claim's relationship to its evidence, independent of what the verifier
decides. That independence is what makes false negatives measurable.

Labels:
  accept  — the claim is faithful to its evidence; rejecting it is a false reject.
  reject  — the claim fabricates, inverts, miscites, or overreaches; accepting
            it is a false accept.
  reject_tier2 — the claim is deterministically clean but semantically
            unsupported (misleading composition). Catching it requires an
            entailment backend; the deterministic tiers are EXPECTED to miss
            it. Reported separately so the deterministic guarantee is scored
            only on what it claims to cover.
"""
from ecp import EvidenceStore, CalcRegistry
from ecp.claims import Claim, AssertedValue


def _world():
    s = EvidenceStore()
    a = s.add_value("Q1 unit sales", 114, "units", source_tool="sales_db", ref="SELECT ... q1")
    b = s.add_value("Q2 unit sales", 100, "units", source_tool="sales_db", ref="SELECT ... q2")
    t = s.add_text("Customer survey: several respondents mentioned pricing concerns.",
                   source_tool="survey", evidence_type="document_chunk")
    cz = s.add_text("Controlled A/B price test: the increase reduced conversions.",
                    source_tool="experiment", supports_causality=True)
    c = CalcRegistry(s)
    pc = c.register("pct_change", [a.evidence_id, b.evidence_id], unit="%")
    rt = c.register("ratio", [b.evidence_id, a.evidence_id])          # no unit
    return s, c, a, b, t, cz, pc, rt


def build_corpus():
    """-> list of (case_id, category, label, store, calcs, claim)"""
    cases = []

    def add(cid, cat, label, claim_fn):
        s, c, a, b, t, cz, pc, rt = _world()
        cases.append((cid, cat, label, s, c, claim_fn(a, b, t, cz, pc, rt)))

    AV = AssertedValue
    # ---- valid claims (label: accept) --------------------------------------
    add("V01", "valid_finding", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Q2 unit sales were 100 units.",[b.evidence_id],[AV(100,"units",b.evidence_id)]))
    add("V02", "valid_comparison_calc", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","comparison","Sales fell 12.28% from Q1 to Q2.",[pc.calc_id],[AV(12.28,"%",pc.calc_id)]))
    add("V03", "valid_negative_asserted", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","comparison","Sales decreased by 12.28%.",[pc.calc_id],[AV(-12.28,"%",pc.calc_id)]))
    add("V04", "valid_causal_with_evidence", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","causal","The price increase caused a drop in conversions.",[cz.evidence_id],causal=True))
    add("V05", "valid_gap", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","gap","The evidence does not establish why sales changed.",[]))
    add("V06", "valid_inference_grounded", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","inference","pricing concerns may relate to the 12.28% decline",[pc.calc_id, t.evidence_id],
              [AV(12.28,"%",pc.calc_id)]))
    add("V07", "valid_year_context", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Q2 sales were 100 units in 2026.",[b.evidence_id],[AV(100,"units",b.evidence_id)]))
    add("V08", "valid_observation_text", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","Survey respondents mentioned pricing concerns.",[t.evidence_id]))
    add("V09", "valid_both_values", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","comparison","Sales were 114 units in Q1 and 100 units in Q2.",
              [a.evidence_id,b.evidence_id],[AV(114,"units",a.evidence_id),AV(100,"units",b.evidence_id)]))
    add("V10", "valid_unitless_ratio", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","The Q2/Q1 sales ratio was 0.877193.",[rt.calc_id],[AV(0.877193,None,rt.calc_id)]))
    add("V11", "valid_half_year_context", "accept", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","Respondents raised pricing concerns through H1 2026.",[t.evidence_id]))

    # ---- deterministic rejects (label: reject) -----------------------------
    add("R01", "fabricated_number", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales fell 14% in Q2.",[b.evidence_id]))
    add("R02", "direction_inversion", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales rose 12.28% in Q2.",[pc.calc_id],[AV(12.28,"%",pc.calc_id)]))
    add("R03", "nonexistent_citation", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales were 100 units.",["E-999"],[AV(100,"units","E-999")]))
    add("R04", "uncited_asserted_ref", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales were 114 units.",[b.evidence_id],[AV(114,"units",a.evidence_id)]))
    add("R05", "causal_no_evidence", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","causal","Pricing caused the sales decline.",[t.evidence_id],causal=True))
    add("R06", "causal_marker_leak", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","Sales fell due to pricing concerns.",[t.evidence_id]))
    add("R07", "causal_synonym_leak", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","Pricing concerns triggered the decline.",[t.evidence_id]))
    add("R08", "unit_mismatch", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Conversion was 100%.",[b.evidence_id],[AV(100.0,"%",b.evidence_id)]))
    add("R09", "unit_omitted_bypass", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Conversion was 100%.",[b.evidence_id],[AV(100.0,None,b.evidence_id)]))
    add("R10", "spelled_out_number", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales fell by twelve point three percent.",[pc.calc_id]))
    add("R11", "small_int_fabrication", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","The company lost 2 customers.",[b.evidence_id]))
    add("R12", "small_pct_fabrication", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","The conversion rate was 3%.",[b.evidence_id]))
    add("R13", "bare_year_quantity", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","The company shipped 1950 units.",[a.evidence_id]))
    add("R14", "inference_fabricated_number", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","inference","revenue increased 999% because pricing worked",[b.evidence_id]))
    add("R15", "recommendation_fabricated_number", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","recommendation","The company lost 999 customers, so close the division.",[b.evidence_id]))
    add("R16", "pct_token_nonpct_evidence", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales hit 100% of target.",[b.evidence_id],[AV(100,"units",b.evidence_id)]))
    add("R17", "no_citation_factual", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales were strong.",[]))
    add("R18", "wrong_magnitude", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Sales were 1000 units in Q2.",[b.evidence_id],[AV(1000,"units",b.evidence_id)]))

    add("R19", "misleading_composition_numeric_cite", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","The company is insolvent.",[b.evidence_id]))
    add("R20", "text_unit_relabeling", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Revenue was 100 dollars.",[b.evidence_id],[AV(100,"units",b.evidence_id)]))
    add("R21", "scientific_notation", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","Revenue was 1e9 dollars.",[b.evidence_id]))
    add("R22", "invented_unit_on_unitless", "reject", lambda a,b,t,cz,pc,rt:
        Claim("S","finding","The Q2/Q1 ratio was 87.7193%.",[rt.calc_id],[AV(87.7193,"%",rt.calc_id)]))

    # ---- beyond the deterministic guarantee (label: reject_tier2) ----------
    add("T01", "misleading_composition_text_cite", "reject_tier2", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","The company is insolvent.",[t.evidence_id]))
    add("T02", "wrong_entity_attribution", "reject_tier2", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","Respondents praised the product quality.",[t.evidence_id]))
    add("T03", "novel_causal_phrasing", "reject_tier2", lambda a,b,t,cz,pc,rt:
        Claim("S","observation","The decline traces back to pricing concerns.",[t.evidence_id]))

    return cases
