"""Figure generation for the benchmark and the two-arm illustration.

    python benchmark/figures.py            # writes docs/figures/*.svg (light + dark)

Stdlib only, like the rest of the package: figures are SVG emitted directly, so
producing them needs no plotting dependency and CI can regenerate them on any
runner. SVG also renders inline in GitHub markdown.

STYLE RULE: the graph carries the shape, the README carries the nuance. These
figures hold a short title, large numbers, at most four colors and direct labels
-- no paragraphs, no methodology, no caveat prose inside the image. Text baked
into an SVG cannot be edited, translated, selected or read by a screen reader,
and it makes a figure look like documentation rather than a chart. The one
exception is the "Constructed illustration" badge on fig 3, because that caveat
must survive being screenshotted away from its caption.

WHAT EACH FIGURE IS ALLOWED TO CLAIM -- enforced in the README captions:

  fig 1  verifier confusion matrix   MEASURED. The verifier against a labelled
                                     corpus whose labels are assigned by
                                     construction, independent of the verifier.
                                     Says nothing about live agents.
  fig 2  rejection coverage by tier  MEASURED, same corpus. Which tier does the
                                     rejecting, and where the deterministic
                                     floor is (the Tier-2 bar is empty).
  fig 3  two-arm illustration        ILLUSTRATIVE. The baseline arm is
                                     hand-written, not model output.
  fig 4  live-run cost               MEASURED, n=1. Latency and tokens only.
                                     Never an accuracy rate.

Fig 2 is bars by tier rather than one mark per attack class: the corpus has
exactly one case per class, so a per-class chart would be 32 identical full-height
bars encoding nothing. The class list belongs in README prose.
"""
from __future__ import annotations

import json
import os
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.corpus import build_corpus                       # noqa: E402
from ecp import Verifier                                        # noqa: E402

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

# Palette per references/palette.md. Both modes are selected, not flipped:
# the dark column is the same hues re-stepped for the dark surface. Validated
# with scripts/validate_palette.js -- categorical 5 slots PASS in both modes
# (light carries a contrast WARN, discharged by the visible direct labels and
# the table view that ships beside every figure).
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "seq": ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#184f95"],
        "good": "#0ca30c", "warning": "#fab219",
        "on_dark_fill": "#ffffff", "on_light_fill": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
        "good": "#0ca30c", "warning": "#fab219",
        "on_dark_fill": "#ffffff", "on_light_fill": "#0b0b0b",
    },
}

GAP = 2          # surface gap between adjacent/stacked fills (never a border)
RADIUS = 4       # rounded data-ends


def ink_on(fill: str) -> str:
    """Pick black or white for text sitting on `fill`, by WCAG luminance.

    Computed rather than indexed off the ramp position: the dark theme's
    sequential ramp runs dark->light where the light theme's runs light->dark,
    so any index-based rule is inverted in one of the two modes (it shipped
    black-on-#0d366b in the first draft of these figures).
    """
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#0b0b0b" if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "#ffffff"


# --------------------------------------------------------------- svg helpers
def _esc(s: str) -> str:
    return escape(str(s))


def text(x, y, s, *, fill, size=12, weight=400, anchor="start", tabular=False):
    extra = ' style="font-variant-numeric:tabular-nums"' if tabular else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{FONT}\' '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{_esc(s)}</text>')


def rect(x, y, w, h, fill, *, r=0, corners="all"):
    """Rect with selective corner rounding.

    corners: 'all' | 'left' | 'right' | 'none'. Stacked segments round only the
    outer data-end so the stack reads as one bar with a rounded cap, per the
    mark spec -- interior joins stay square and are separated by a surface gap.
    """
    if w <= 0 or h <= 0:
        return ""
    if r <= 0 or corners == "none":
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>'
    r = min(r, h / 2, w / 2)
    if corners == "all":
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="{r:.1f}" fill="{fill}"/>')
    if corners == "left":
        d = (f"M{x+r:.1f},{y:.1f} H{x+w:.1f} V{y+h:.1f} H{x+r:.1f} "
             f"A{r:.1f},{r:.1f} 0 0 1 {x:.1f},{y+h-r:.1f} V{y+r:.1f} "
             f"A{r:.1f},{r:.1f} 0 0 1 {x+r:.1f},{y:.1f} Z")
    else:  # right
        d = (f"M{x:.1f},{y:.1f} H{x+w-r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x+w:.1f},{y+r:.1f} "
             f"V{y+h-r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x+w-r:.1f},{y+h:.1f} H{x:.1f} Z")
    return f'<path d="{d}" fill="{fill}"/>'


def svg_doc(w, h, body, theme) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">'
            f'<rect width="{w}" height="{h}" fill="{theme["surface"]}"/>'
            f"{body}</svg>")


def legend(x, y, items, theme, *, size=11):
    """Legend is always present for >=2 series; swatch carries identity, the
    label wears ink tokens rather than the series color."""
    out, cx = [], x
    for label, color in items:
        out.append(rect(cx, y - 7, 9, 9, color, r=2, corners="all"))
        out.append(text(cx + 14, y, label, fill=theme["ink2"], size=size))
        cx += 14 + len(label) * (size * 0.55) + 20
    return "".join(out)


def wrap_lines(s, width_chars):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width_chars:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    lines.append(cur)
    return lines


def wrapped(x, y, s, width_chars, theme, *, size=10.5, lh=14, fill=None):
    """Draw wrapped text. Returns (svg, height_used) so callers advance a
    cursor instead of guessing -- guessing is how the first draft put the
    footnote below the bottom edge of the canvas."""
    lines = wrap_lines(s, width_chars)
    svg = "".join(text(x, y + i * lh, ln, fill=fill or theme["muted"], size=size)
                  for i, ln in enumerate(lines))
    return svg, (len(lines) - 1) * lh + size


def check_bounds(doc: str, w: int, h: int, name: str) -> None:
    """Fail loudly if any element sits outside the canvas.

    The palette validator checks color, not geometry. This is the geometry
    equivalent: a cheap guard against the exact bug this file shipped in its
    first draft (a caption drawn at y=332 inside a 330px viewBox, silently
    clipped by every renderer).
    """
    import re
    bad = []
    pat = (r'<text x="(-?[\d.]+)" y="(-?[\d.]+)"[^>]*?font-size="([\d.]+)"'
           r'[^>]*?text-anchor="(\w+)"[^>]*>([^<]*)</text>')
    for m in re.finditer(pat, doc):
        x, y, size, anchor, content = (float(m.group(1)), float(m.group(2)),
                                       float(m.group(3)), m.group(4), m.group(5))
        # ~0.55em average advance for this sans at these sizes; deliberately
        # generous, since the point is to catch overflow, not to typeset.
        tw = len(content) * size * 0.55
        left = x if anchor == "start" else (x - tw / 2 if anchor == "middle" else x - tw)
        if y > h - 4 or y < 0 or left < 0 or left + tw > w:
            bad.append(f"text {content[:24]!r} at ({x},{y}) span {left:.0f}-{left + tw:.0f}")
    for m in re.finditer(r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" '
                         r'height="([\d.]+)"', doc):
        x, y, rw, rh = (float(g) for g in m.groups())
        if x + rw > w or y + rh > h:
            bad.append(f"rect to ({x + rw},{y + rh})")
    if bad:
        raise AssertionError(f"{name}: {len(bad)} element(s) outside {w}x{h}: "
                             + "; ".join(bad[:4]))


# ------------------------------------------------------------------- data
def corpus_results():
    """Re-verify the corpus and record which tier rejected each case.

    Derived from actual verifier behaviour rather than a hand-maintained map,
    so the coverage figure cannot drift away from what the code does.
    """
    out = []
    for cid, cat, label, store, calcs, claim in build_corpus():
        vr = Verifier(store, calcs).verify(claim)
        rejected = vr.status == "rejected"
        tier = next((t.tier for t in vr.tier_results if not t.passed), None)
        out.append({"id": cid, "category": cat, "label": label,
                    "rejected": rejected, "tier": tier})
    return out


# --------------------------------------------------- fig 1: confusion matrix
def fig_confusion(rows, theme):
    tp = sum(1 for r in rows if r["label"] == "reject" and r["rejected"])
    fn = sum(1 for r in rows if r["label"] == "reject" and not r["rejected"])
    tn = sum(1 for r in rows if r["label"] == "accept" and not r["rejected"])
    fp = sum(1 for r in rows if r["label"] == "accept" and r["rejected"])
    t2 = [r for r in rows if r["label"] == "reject_tier2"]
    t2_caught = sum(1 for r in t2 if r["rejected"])

    del t2, t2_caught          # out-of-scope residual belongs in fig 2, not here
    PAD, cell, x0, y0 = 28, 118, 168, 96
    W = x0 + 2 * cell + GAP + PAD

    b = [text(PAD, 34, "Verifier discrimination", fill=theme["ink"], size=15, weight=600)]
    b.append(text(x0, y0 - 30, "VERIFIER", fill=theme["muted"], size=9.5, weight=600))
    for j, lab in enumerate(("reject", "accept")):
        b.append(text(x0 + j * (cell + GAP) + cell / 2, y0 - 12, lab,
                      fill=theme["ink2"], size=12, anchor="middle"))
    b.append(text(x0 - 14, y0 - 30, "ACTUAL", fill=theme["muted"], size=9.5,
                  weight=600, anchor="end"))
    for i, lab in enumerate(("bad", "good")):
        b.append(text(x0 - 14, y0 + i * (cell + GAP) + cell / 2 + 5, lab,
                      fill=theme["ink2"], size=12, anchor="end"))

    grid = [[tp, fn], [fp, tn]]
    hi = max(max(r) for r in grid) or 1
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            x, y = x0 + j * (cell + GAP), y0 + i * (cell + GAP)
            # sequential ramp: zero recedes toward the surface, magnitude darkens
            idx = 0 if v == 0 else (2 if v / hi < 0.6 else 5)
            fill = theme["seq"][idx]
            b.append(rect(x, y, cell, cell, fill, r=RADIUS, corners="all"))
            b.append(text(x + cell / 2, y + cell / 2 + 14, str(v), fill=ink_on(fill),
                          size=40, weight=600, anchor="middle"))

    H = y0 + 2 * cell + GAP + PAD
    doc = svg_doc(W, H, "".join(b), theme)
    check_bounds(doc, W, H, "confusion")
    return doc


# ------------------------------------------------- fig 2: attack-class grid
TIER_LABELS = [("structural", "Structural"), ("value", "Value"),
               ("causal_gate", "Causal gate")]


def fig_coverage(rows, theme):
    """Rejection coverage by verifier tier.

    Bars, not a grid of class names: the reader needs the shape (which tier does
    the work, and where the floor is), and the class list reads better as text in
    the README. The empty Tier-2 bar is the point of the figure.
    """
    caught = [r for r in rows if r["label"] == "reject"]
    residual = [r for r in rows if r["label"] == "reject_tier2"]

    bars = []
    for tier, label in TIER_LABELS:
        items = [r for r in caught if r["tier"] == tier]
        bars.append((label, len(items), len(items)))
    bars.append(("Tier-2 needed", sum(1 for r in residual if r["rejected"]), len(residual)))

    PAD, x0, bw, bh, gap = 28, 132, 300, 26, 16
    W = x0 + bw + 74
    b = [text(PAD, 34, "Rejection coverage by tier", fill=theme["ink"], size=15,
              weight=600)]

    hi = max(tot for _, _, tot in bars) or 1
    cur = 74
    for label, got, tot in bars:
        b.append(text(x0 - 14, cur + bh / 2 + 4, label, fill=theme["ink2"], size=12,
                      anchor="end"))
        track_w = bw * (tot / hi)
        b.append(rect(x0, cur, track_w, bh, theme["grid"], r=RADIUS, corners="all"))
        if got:
            b.append(rect(x0, cur, bw * (got / hi), bh, theme["seq"][4], r=RADIUS,
                          corners="all"))
        b.append(text(x0 + track_w + 12, cur + bh / 2 + 5, f"{got}/{tot}",
                      fill=theme["ink"] if got else theme["muted"], size=13,
                      weight=600, tabular=True))
        cur += bh + gap

    H = cur - gap + PAD
    doc = svg_doc(W, H, "".join(b), theme)
    check_bounds(doc, W, H, "coverage")
    return doc


# -------------------------------------------------- fig 3: two-arm illustration
# Four slots, shared across both arms so a color means the same thing in each.
# The baseline's deterministic-reject and semantic-residual fragments collapse
# into one "unsupported" block: the split between them is a property of the
# verifier, not of the answer, and it is already the subject of figure 2.
SEGMENTS = [
    ("grounded", 0),
    ("citation resolved", 2),
    ("interpretive / gap", 3),
    ("unsupported", 1),
]


def fig_two_arm(record, theme):
    a = record["arm_a"]["scored"]
    a_counts = {"grounded": sum(1 for s in a if s["status"] == "ok"),
                "citation resolved": 0,
                "interpretive / gap": 0,
                "unsupported": sum(1 for s in a if s["status"] in ("bad", "tier2"))}
    lv = [s["verification_level"] for s in record["arm_b"]["proof"]["sentences"]]
    b_counts = {"grounded": lv.count("numerically_grounded"),
                "citation resolved": lv.count("citation_resolved"),
                "interpretive / gap": lv.count("interpretive"),
                "unsupported": 0}

    PAD, x0, bar_w, bar_h = 28, 150, 520, 44
    W = x0 + bar_w + 60
    total = max(sum(a_counts.values()), sum(b_counts.values()))
    unit = bar_w / total

    b = [text(PAD, 34, "Statement fate per answer", fill=theme["ink"], size=15,
              weight=600)]
    # badge, not a paragraph: the caveat that must not be croppable
    badge = "Constructed illustration"
    bw_px = len(badge) * 5.6 + 18
    b.append(rect(W - PAD - bw_px, 21, bw_px, 18, theme["grid"], r=9, corners="all"))
    b.append(text(W - PAD - bw_px / 2, 34, badge, fill=theme["ink2"], size=10,
                  anchor="middle"))

    cur = 74
    for name, counts in [("Baseline", a_counts), ("ECP", b_counts)]:
        b.append(text(x0 - 14, cur + bar_h / 2 + 5, name, fill=theme["ink2"], size=12.5,
                      anchor="end"))
        cx = x0
        drawn = [(slot, counts[lab]) for lab, slot in SEGMENTS if counts[lab]]
        for i, (slot, n) in enumerate(drawn):
            w = n * unit - (GAP if i < len(drawn) - 1 else 0)
            corners = ("all" if len(drawn) == 1 else
                       "left" if i == 0 else "right" if i == len(drawn) - 1 else "none")
            fill = theme["series"][slot]
            b.append(rect(cx, cur, w, bar_h, fill, r=RADIUS, corners=corners))
            if w >= 24:
                b.append(text(cx + w / 2, cur + bar_h / 2 + 6, str(n), fill=ink_on(fill),
                              size=16, weight=600, anchor="middle"))
            cx += n * unit
        b.append(text(cx + 12, cur + bar_h / 2 + 5, str(sum(counts.values())),
                      fill=theme["muted"], size=12, tabular=True))
        cur += bar_h + 18

    cur += 14
    b.append(legend(x0 - 14, cur,
                    [(lab, theme["series"][slot]) for lab, slot in SEGMENTS], theme,
                    size=10.5))
    H = int(cur + PAD)
    doc = svg_doc(W, H, "".join(b), theme)
    check_bounds(doc, W, H, "two-arm")
    return doc


# --------------------------------------------------- fig 4: live-run cost
def fig_live_cost(record, theme):
    a = record.get("arm_a_naive", {}).get("cost") or {}
    e = record.get("arm_b_ecp", {}).get("cost") or {}
    if not a.get("llm_calls") or not e.get("llm_calls"):
        return None

    PAD, x0, bw = 28, 150, 460
    W = x0 + bw + 90
    b = [text(PAD, 34, "Cost per answer", fill=theme["ink"], size=15, weight=600)]

    metrics = [("LLM calls", a["llm_calls"], e["llm_calls"], ""),
               ("Wall seconds", a["wall_seconds"], e["wall_seconds"], "s"),
               ("Prompt tokens", a["prompt_tokens"], e["prompt_tokens"], ""),
               ("Output tokens", a["output_tokens"], e["output_tokens"], "")]

    cur = 68
    for label, av, ev, unit in metrics:
        hi = max(av, ev) or 1
        b.append(text(x0 - 14, cur + 15, label, fill=theme["ink2"], size=11.5,
                      anchor="end"))
        for k, (val, color) in enumerate([(av, theme["series"][0]),
                                          (ev, theme["series"][1])]):
            yy = cur + k * (13 + GAP)
            w = max(2.0, bw * (val / hi))
            b.append(rect(x0, yy, w, 13, color, r=RADIUS, corners="right"))
            b.append(text(x0 + w + 10, yy + 11, f"{val:g}{unit}", fill=theme["ink2"],
                          size=11, tabular=True))
        cur += 28 + 22

    cur += 6
    b.append(legend(x0 - 14, cur, [("baseline", theme["series"][0]),
                                   ("ECP", theme["series"][1])], theme, size=10.5))
    H = int(cur + PAD)
    doc = svg_doc(W, H, "".join(b), theme)
    check_bounds(doc, W, H, "live-cost")
    return doc


# ------------------------------------------------------------------ preview
PREVIEW_CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--line:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:rgba(255,255,255,.10)}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;
--ink2:#c3c2b7;--line:rgba(255,255,255,.10)}
body{background:var(--bg);color:var(--ink);margin:0;padding:40px 24px;line-height:1.55;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:24px;margin:0 0 6px}
.sub{color:var(--ink2);font-size:14px;margin:0 0 32px}
figure{margin:0 0 34px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:18px;overflow-x:auto}
figcaption{font-size:12.5px;color:var(--ink2);margin-top:12px;padding-top:12px;
border-top:1px solid var(--line)}
.tag{font-size:11px;font-weight:700;letter-spacing:.05em;color:var(--muted)}
svg{max-width:100%;height:auto;display:block}
.dark-only{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .light-only{display:none}
:root:not([data-theme=light]) .dark-only{display:block}}
:root[data-theme=dark] .light-only{display:none}
:root[data-theme=dark] .dark-only{display:block}
"""

PREVIEW_FIGS = [
    ("verifier-confusion-matrix", "MEASURED",
     "The verifier against a labelled corpus whose labels are assigned by construction, "
     "independent of the verifier. Says nothing about live-agent hallucination rates."),
    ("rejection-coverage-by-tier", "MEASURED",
     "Which tier does the rejecting, and where the deterministic floor is: the "
     "Tier-2 bar is empty without an entailment backend."),
    ("two-arm-illustration", "ILLUSTRATIVE",
     "The unverified arm is hand-written, not model output. The caveat is drawn into "
     "the image so it travels with the figure."),
    ("live-run-cost", "MEASURED, n=1",
     "Latency and token cost from a single local-model run. Never an accuracy rate."),
]


def write_preview(out_dir: str) -> str:
    """One self-contained page showing every generated figure in both themes."""
    parts = ["<title>ECP Benchmark Figures</title>", f"<style>{PREVIEW_CSS}</style>",
             '<div class="wrap"><h1>ECP benchmark figures</h1>',
             '<p class="sub">Generated by <code>python benchmark/figures.py</code> - '
             "stdlib only, no plotting dependency. Each figure states what it is "
             "allowed to claim.</p>"]
    for slug, tag, claim in PREVIEW_FIGS:
        light = os.path.join(out_dir, f"{slug}.svg")
        dark = os.path.join(out_dir, f"{slug}-dark.svg")
        if not os.path.exists(light):
            continue
        with open(light, encoding="utf-8") as f:
            lsvg = f.read()
        with open(dark, encoding="utf-8") as f:
            dsvg = f.read()
        parts.append(f'<figure><div class="light-only">{lsvg}</div>'
                     f'<div class="dark-only">{dsvg}</div>'
                     f'<figcaption><span class="tag">{tag}</span><br>{_esc(claim)}'
                     "</figcaption></figure>")
    parts.append("</div>")
    path = os.path.join(out_dir, "preview.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ------------------------------------------------------------------- driver
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = corpus_results()
    written = []

    def write(name, builder, *args):
        for mode, theme in THEMES.items():
            doc = builder(*args, theme)
            if doc is None:
                return
            suffix = "" if mode == "light" else "-dark"
            path = os.path.join(OUT_DIR, f"{name}{suffix}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            written.append(path)

    write("verifier-confusion-matrix", fig_confusion, rows)
    write("rejection-coverage-by-tier", fig_coverage, rows)

    root = os.path.join(os.path.dirname(__file__), "..")
    arm = os.path.join(root, "arm_comparison_record.json")
    if os.path.exists(arm):
        with open(arm, encoding="utf-8") as f:
            write("two-arm-illustration", fig_two_arm, json.load(f))
    else:
        print("skip two-arm figure: run examples/02_arm_comparison.py first")

    live = os.path.join(root, "live_run_record.json")
    if os.path.exists(live):
        with open(live, encoding="utf-8") as f:
            write("live-run-cost", fig_live_cost, json.load(f))
    else:
        print("skip live-run figure: run examples/04_ecp_real_agent.py first")

    written.append(write_preview(OUT_DIR))
    for p in written:
        print(f"wrote {os.path.relpath(p, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
