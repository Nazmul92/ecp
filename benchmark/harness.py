"""Verifier benchmark: runs the labelled adversarial corpus and reports a
confusion matrix with precision / recall / false-accept / false-reject rates.

    python benchmark/harness.py

What this measures: the DETERMINISTIC verifier's discrimination on claims whose
ground truth is known by construction (see corpus.py). Because labels are
independent of the verifier, false negatives ARE measurable here — this is not
the verifier grading itself.

What this does NOT measure: end-to-end answer quality with a live LLM
(completeness loss, repair convergence, latency, cost). That requires a real
model and a domain corpus; wire VerifiedPipeline into run_end_to_end() below
and report only what you measure.

'reject_tier2' cases are semantically unsupported but deterministically clean.
The deterministic tiers are EXPECTED to miss them; they are scored separately
so the guarantee is evaluated only on its stated scope — and the residual is
published, not hidden.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from ecp import Verifier
from benchmark.corpus import build_corpus


def run(tier2=None):
    rows, by_cat = [], defaultdict(lambda: [0, 0])
    tp = fp = tn = fn = 0                      # positive class = "should reject"
    t2_caught = t2_total = 0
    for cid, cat, label, store, calcs, claim in build_corpus():
        v = Verifier(store, calcs, tier2=tier2, tier2_policy="reject")
        rejected = v.verify(claim).status == "rejected"
        if label == "reject_tier2":
            t2_total += 1
            t2_caught += rejected
            rows.append((cid, cat, label, "rejected" if rejected else "accepted", "n/a (tier-2 scope)"))
            continue
        want_reject = label == "reject"
        ok = rejected == want_reject
        by_cat[cat][0] += ok; by_cat[cat][1] += 1
        tp += want_reject and rejected;  fn += want_reject and not rejected
        tn += (not want_reject) and (not rejected); fp += (not want_reject) and rejected
        rows.append((cid, cat, label, "rejected" if rejected else "accepted",
                     "OK" if ok else "** MISS **"))
    return rows, (tp, fp, tn, fn), (t2_caught, t2_total), by_cat


if __name__ == "__main__":
    rows, (tp, fp, tn, fn), (t2c, t2n), by_cat = run()
    w = max(len(c) for _, c, *_ in rows) + 2
    for cid, cat, label, got, verdict in rows:
        print(f"{cid}  {cat:<{w}} label={label:<13} verifier={got:<9} {verdict}")
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    print(f"\nDeterministic scope: {n} cases")
    print(f"  bad claims rejected (recall):        {rec:.1%}  ({tp}/{tp+fn})")
    print(f"  rejections that were right (precision): {prec:.1%}  ({tp}/{tp+fp})")
    print(f"  false-accept rate (missed bad claims):  {fn/max(1,tp+fn):.1%}")
    print(f"  false-reject rate (blocked good claims):{fp/max(1,tn+fp):.1%}")
    all_bad = tp + fn + t2n
    print(f"\nBeyond deterministic scope (needs Tier-2): {t2c}/{t2n} caught "
          f"without an entailment backend (expected ~0; this is the published residual).")
    print(f"Full-corpus aggregate (deterministic + Tier-2 territory together): "
          f"{tp + t2c}/{all_bad} bad claims rejected ({(tp + t2c)/all_bad:.1%}), "
          f"{tp + t2c + tn}/{n + t2n} cases correct ({(tp + t2c + tn)/(n + t2n):.1%}).")
    miss = [r for r in rows if r[4].startswith("**")]
    if miss:
        print(f"\n{len(miss)} in-scope misses — the guarantee is NOT met:")
        for r in miss:
            print("  ", r[0], r[1])
        sys.exit(1)
    print("\nAll in-scope cases correct. Residual is confined to Tier-2 territory, as documented.")
