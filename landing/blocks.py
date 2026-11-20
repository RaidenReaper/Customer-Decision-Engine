"""Shared landing-page sections, settled on version 1 (11/13).

Plain business language first, with the precise rule as a secondary line;
sections in the order a user works; every chart says where to find it.
Each variant arranges these blocks in its own layout.
"""
from __future__ import annotations

from html import escape
from typing import Any

from landing.common import (
    BAND_RULES,
    BANDS,
    CONFIDENCE_LEVELS,
    FAQ,
    FEATURES,
    WIN_VS_CONFIDENCE,
    band_pill,
    confidence_pill,
    cutoff_sentence,
    product_cutoffs,
)

# The user's workflow: (title, what to do, where in the tool).
JOURNEY: list[tuple[str, str, str]] = [
    ("Choose your account and products", "Pick a health system and the products you want to review.", "Top of the page"),
    ("Explore a different scenario", "Adjust the levers, then select Run Simulation. Change one thing at a time.", "Simulation"),
    ("Read what changed — and why", "Read the outcome and confidence together, then the drivers and Narrative Summary.", "Simulation"),
    ("Explore account characteristics", "Review the account's size, pricing, buying behaviour, relationship signals and history.", "Customer Deep Dive"),
    ("Prepare your next discussion", "Test another scenario, reset your inputs, or download the report.", "Simulation"),
]

# Exact click paths, using the tool's own headings.
FIND_DRIVERS = ("Simulation tab", "Contribution to predicted Win probability",
                "open “SHAP waterfall — driver-level detail”")
FIND_CURVE = ("Simulation tab", "open “SHAP waterfall — driver-level detail”", "below the driver chart")

# Covered by the Reading-the-simulation and driving-factors sections instead.
_COVERED_ELSEWHERE = {"How do I read the result?", "How do I see why?"}
QUICK_GUIDE: list[tuple[str, str]] = [(q, a) for q, a in FAQ if q not in _COVERED_ELSEWHERE]

SECTION_SIMULATION = ("Reading the simulation",
                      "Every product gets a band and a confidence level. Read both before you act.")
SECTION_DRIVERS = ("Understand the driving factors",
                   "Two charts explain each prediction. Here is how to read them.")

BLOCKS_CSS = """
<style>
.lp-guide-title {display:flex; align-items:baseline; justify-content:space-between; gap:1rem; margin:1.6rem 0 .1rem;}
.lp-sim {display:grid; grid-template-columns:minmax(0,1fr) 1px minmax(0,1fr); gap:1.6rem;}
.lp-sim-divider {background:var(--border);}
.lp-sim-lead {font-size:.86rem; color:var(--text-secondary); line-height:1.45; margin:.1rem 0 .45rem;}
.lp-lrow {display:flex; align-items:flex-start; gap:.75rem; padding:.5rem 0; border-bottom:1px dashed var(--border);}
.lp-lrow:last-child {border-bottom:none;}
.lp-ltag {flex:0 0 165px;}
.lp-ltext {font-size:.86rem; color:var(--text-primary); line-height:1.45;}
.lp-lrule {font-size:.76rem; color:var(--text-secondary); margin-top:.15rem;}
.lp-callout {margin-top:.75rem; padding:.65rem .8rem; border-radius:10px; background:var(--light-purple);
    font-size:.84rem; color:var(--text-primary); line-height:1.45;}
.lp-chart-title {font-size:.98rem; font-weight:800; color:var(--text-primary); margin:.2rem 0 .2rem;}
.lp-find {display:inline-flex; flex-wrap:wrap; align-items:center; gap:.3rem; margin:.1rem 0 .55rem;
    padding:.22rem .7rem; border-radius:999px; background:var(--light-purple); color:var(--purple);
    font-size:.78rem; font-weight:600;}
.lp-find b {font-weight:800;}
.lp-find-arrow {opacity:.7;}
.lp-chart-lead {font-size:.88rem; color:var(--text-secondary); line-height:1.5; margin-bottom:.8rem;}
.lp-chart-sep {height:1px; background:var(--border); margin:1.2rem 0 1rem;}
.lp-drv {display:grid; grid-template-columns:190px minmax(0,1fr) 320px; gap:.55rem 1rem; align-items:center;}
.lp-drv-label {font-size:.88rem; color:var(--text-primary);}
.lp-drv-track {position:relative; height:18px;}
.lp-drv-track::before {content:""; position:absolute; left:50%; top:-5px; bottom:-5px; width:1px; background:#918884;}
.lp-drv-bar {position:absolute; top:0; height:18px; border-radius:3px;}
.lp-drv-pos {left:50%; background:#1E88E5;}
.lp-drv-neg {right:50%; background:#FF0D57;}
.lp-drv-note {font-size:.84rem; font-weight:700;}
.lp-drv-note-pos {color:#1565C0;}
.lp-drv-note-neg {color:#C2185B;}
.lp-drv-axis {display:flex; justify-content:space-between; font-size:.72rem; font-weight:700;
    color:var(--text-secondary); border-top:1px solid var(--border); padding-top:.3rem;}
.lp-curve-wrap {max-width:900px;}
.lp-curve {width:100%; height:auto; display:block;}
.lp-curve-tips {margin:.6rem 0 0; padding-left:1.1rem; font-size:.86rem; color:var(--text-secondary); line-height:1.6;}
.lp-curve-tips b {color:var(--text-primary);}
.lp-ftable {width:100%; border-collapse:collapse; font-size:.85rem;}
.lp-ftable th {text-align:left; font-size:.7rem; text-transform:uppercase; letter-spacing:.04rem;
    color:var(--text-secondary); padding:.35rem .5rem; border-bottom:1px solid var(--border);}
.lp-ftable td {padding:.45rem .5rem; border-bottom:1px solid var(--border); vertical-align:top; color:var(--text-primary);}
.lp-ftable td.lp-dim {color:var(--text-secondary);}
/* Narrow containers (side columns, slides) can opt into the stacked layout. */
.lp-narrow .lp-sim {grid-template-columns:minmax(0,1fr); gap:1rem;}
.lp-narrow .lp-sim-divider {height:1px;}
@media (max-width: 1100px) {
    .lp-lrow {flex-direction:column; gap:.3rem;}
    .lp-ltag {flex-basis:auto;}
    .lp-sim {grid-template-columns:minmax(0,1fr); gap:1rem;}
    .lp-sim-divider {height:1px;}
    .lp-drv {grid-template-columns:150px minmax(0,1fr);}
    .lp-drv-note {grid-column:2; margin-top:-.3rem;}
    .lp-drv > div:nth-last-child(1) {display:none;}
}
</style>
"""


def section_title(title: str, note: str = "") -> str:
    """Section heading with an optional short note on the right."""
    side = f"<div class='lp-note'>{escape(note)}</div>" if note else ""
    return f"<div class='lp-guide-title'><div class='lp-h2' style='font-size:1.3rem;'>{escape(title)}</div>{side}</div>"


_PIN = (
    "<svg width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' "
    "stroke-linecap='round' stroke-linejoin='round'><path d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'/>"
    "<circle cx='12' cy='10' r='3'/></svg>"
)


def find_it(*path: str) -> str:
    """'Where to find it' pill: the click path inside the tool."""
    steps = " <span class='lp-find-arrow'>→</span> ".join(escape(step) for step in path)
    return f"<div class='lp-find'>{_PIN}<b>Where to find it:</b> {steps}</div>"


# --- Reading the simulation ---------------------------------------------------

def _label_rows(items: list[tuple[str, str, str]]) -> str:
    return "".join(
        f"<div class='lp-lrow'><div class='lp-ltag'>{tag}</div>"
        f"<div class='lp-ltext'>{escape(meaning)}"
        + (f"<div class='lp-lrule'>{escape(rule)}</div>" if rule else "")
        + "</div></div>"
        for tag, meaning, rule in items
    )


def win_probability_html(metadata: dict[str, Any] | None) -> str:
    bands = _label_rows([(band_pill(label, kind), meaning, BAND_RULES[label]) for label, kind, meaning in BANDS])
    return (
        "<div><div class='lp-h2'>1. Win probability</div>"
        "<div class='lp-sim-lead'>How likely we are to win this account under the scenario you set. "
        "Each product is compared with its own winning cutoff.</div>"
        f"{bands}"
        f"<div class='lp-note' style='margin-top:.35rem;'>{escape(cutoff_sentence(metadata))}</div></div>"
    )


def confidence_html() -> str:
    levels = _label_rows([(confidence_pill(level), meaning, "") for level, meaning in CONFIDENCE_LEVELS])
    return (
        "<div><div class='lp-h2'>2. Confidence</div>"
        "<div class='lp-sim-lead'>How much we can rely on the prediction. Even High confidence "
        "does not guarantee the outcome.</div>"
        f"{levels}</div>"
    )


def win_vs_confidence_html() -> str:
    head, body = WIN_VS_CONFIDENCE
    return f"<div class='lp-callout'><strong>{escape(head)}</strong> {escape(body)}</div>"


def simulation_card_html(metadata: dict[str, Any] | None, narrow: bool = False) -> str:
    """Win probability and confidence side by side in one card (stacked when narrow)."""
    return (
        f"<div class='lp-card{' lp-narrow' if narrow else ''}'><div class='lp-sim'>"
        f"{win_probability_html(metadata)}<div class='lp-sim-divider'></div>{confidence_html()}"
        f"</div>{win_vs_confidence_html()}</div>"
    )


# --- Understand the driving factors -------------------------------------------

# Illustrative drivers (not real data): (factor, signed contribution).
_EXAMPLE_DRIVERS = [
    ("Price vs benchmark", 38),
    ("Prior-year wallet share", 26),
    ("Negotiation stance", -22),
    ("Competitor price gap", -14),
]
# Illustrative price-response points at -20..+20% around a $20.00 current price.
_EXAMPLE_STEPS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
_EXAMPLE_PROBS = [0.74, 0.71, 0.67, 0.63, 0.58, 0.53, 0.47, 0.42, 0.37]
_EXAMPLE_PRICE = 20.0
_WATCH_WIDTH = 0.20


def _band_name(p: float, cutoff: float) -> str:
    if p >= cutoff:
        return "Comfortable Win"
    return "Watch" if p >= cutoff - _WATCH_WIDTH else "At Risk"


def drivers_chart_html() -> str:
    """Example driver bars, each annotated with what it does in words."""
    largest_pos = max(v for _, v in _EXAMPLE_DRIVERS if v > 0)
    largest_neg = min(v for _, v in _EXAMPLE_DRIVERS if v < 0)
    scale = max(abs(v) for _, v in _EXAMPLE_DRIVERS)
    rows = []
    for name, value in _EXAMPLE_DRIVERS:
        kind = "pos" if value > 0 else "neg"
        strength = "significantly" if value in (largest_pos, largest_neg) else "less significantly"
        verb = "Improves" if value > 0 else "Reduces"
        rows.append(
            f"<div class='lp-drv-label'>{escape(name)}</div>"
            f"<div class='lp-drv-track'><div class='lp-drv-bar lp-drv-{kind}' "
            f"style='width:{abs(value) / scale * 48:.1f}%;'></div></div>"
            f"<div class='lp-drv-note lp-drv-note-{kind}'>{verb} chances of winning {strength}</div>"
        )
    axis = (
        "<div></div><div class='lp-drv-axis'><span>◀ Reduces chance of winning</span>"
        "<span>Supports chance of winning ▶</span></div><div></div>"
    )
    return f"<div class='lp-drv'>{''.join(rows)}{axis}</div>"


def price_curve_svg(cutoff: float) -> str:
    """Example price-response curve with labelled axes, band zones and the current price."""
    width, height = 760, 330
    left, right, top, bottom = 84, 736, 18, 246
    y_min, y_max = 0.30, 0.80

    def x(i: int) -> float:
        return left + i * (right - left) / (len(_EXAMPLE_STEPS) - 1)

    def y(p: float) -> float:
        p = min(max(p, y_min), y_max)
        return bottom - (p - y_min) / (y_max - y_min) * (bottom - top)

    watch_floor = cutoff - _WATCH_WIDTH
    zones = [
        (y_max, cutoff, "#E6F6EC", "#15803D", "Comfortable Win"),
        (cutoff, watch_floor, "#FFF1E5", "#EA580C", "Watch"),
        (watch_floor, y_min, "#FDE8E8", "#DC2626", "At Risk"),
    ]
    parts = []
    for hi, lo, fill, ink, label in zones:
        if hi <= y_min or lo >= y_max:
            continue
        y_hi, y_lo = y(hi), y(lo)
        parts.append(f"<rect x='{left}' y='{y_hi:.1f}' width='{right - left}' height='{y_lo - y_hi:.1f}' fill='{fill}'/>")
        parts.append(f"<text x='{right - 8}' y='{y_hi + 15:.1f}' text-anchor='end' font-size='11' "
                     f"font-weight='700' fill='{ink}'>{label}</text>")
    for tick in range(30, 81, 10):
        ty = y(tick / 100)
        parts.append(f"<line x1='{left}' x2='{right}' y1='{ty:.1f}' y2='{ty:.1f}' stroke='#DED5D0' stroke-width='.8'/>")
        parts.append(f"<text x='{left - 8}' y='{ty + 4:.1f}' text-anchor='end' font-size='11' fill='#716964'>{tick}%</text>")
    for i, pct in enumerate(_EXAMPLE_STEPS):
        tx = x(i)
        sign = "+" if pct > 0 else ("−" if pct < 0 else "")
        price = _EXAMPLE_PRICE * (1 + pct / 100)
        weight = "800" if pct == 0 else "400"
        parts.append(f"<text x='{tx:.1f}' y='{bottom + 18}' text-anchor='middle' font-size='11' "
                     f"font-weight='{weight}' fill='#433C38'>{sign}{abs(pct)}%</text>")
        parts.append(f"<text x='{tx:.1f}' y='{bottom + 32}' text-anchor='middle' font-size='10' "
                     f"fill='#716964'>&#36;{price:.2f}</text>")  # &#36; = "$" (st.markdown reads $..$ as maths)
    cy = y(cutoff)
    parts.append(f"<line x1='{left}' x2='{right}' y1='{cy:.1f}' y2='{cy:.1f}' stroke='#DC2626' "
                 "stroke-dasharray='6 4' stroke-width='1.6'/>")
    parts.append(f"<text x='{left + 8}' y='{cy - 6:.1f}' font-size='11' font-weight='700' fill='#DC2626'>"
                 f"Win threshold {cutoff:.0%}</text>")
    points = " ".join(f"{x(i):.1f},{y(p):.1f}" for i, p in enumerate(_EXAMPLE_PROBS))
    parts.append(f"<polyline points='{points}' fill='none' stroke='#EC7200' stroke-width='2.6'/>")
    current = _EXAMPLE_STEPS.index(0)
    for i, p in enumerate(_EXAMPLE_PROBS):
        if i != current:
            parts.append(f"<circle cx='{x(i):.1f}' cy='{y(p):.1f}' r='3.5' fill='#CF7D4A'/>")
    px, py = x(current), y(_EXAMPLE_PROBS[current])
    parts.append(f"<line x1='{px:.1f}' x2='{px:.1f}' y1='{py:.1f}' y2='{bottom}' stroke='#EC7200' "
                 "stroke-dasharray='2 3' opacity='.6'/>")
    parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='6' fill='#EC7200' stroke='#fff' stroke-width='2'/>")
    parts.append(f"<text x='{px + 10:.1f}' y='{py - 10:.1f}' font-size='11' font-weight='800' fill='#1F1F29'>"
                 f"Current price: {_EXAMPLE_PROBS[current]:.0%}</text>")
    parts.append(f"<text x='{(left + right) / 2:.1f}' y='{height - 6}' text-anchor='middle' font-size='12' "
                 "font-weight='700' fill='#433C38'>Price change from the current price (resulting price per dose)</text>")
    mid_y = (top + bottom) / 2
    parts.append(f"<text x='16' y='{mid_y:.1f}' text-anchor='middle' font-size='12' font-weight='700' "
                 f"fill='#433C38' transform='rotate(-90 16 {mid_y:.1f})'>Predicted win probability</text>")
    parts.append(f"<line x1='{left}' x2='{right}' y1='{bottom}' y2='{bottom}' stroke='#918884'/>")
    parts.append(f"<line x1='{left}' x2='{left}' y1='{top}' y2='{bottom}' stroke='#918884'/>")
    return (
        f"<svg class='lp-curve' viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Example price-response curve: predicted win probability at each price change'>"
        + "".join(parts) + "</svg>"
    )


def curve_takeaway(cutoff: float) -> str:
    """One sentence on where the example line crosses the threshold."""
    current = _EXAMPLE_STEPS.index(0)
    now = _band_name(_EXAMPLE_PROBS[current], cutoff)
    if _EXAMPLE_PROBS[current] >= cutoff:
        for pct, p in zip(_EXAMPLE_STEPS[current + 1:], _EXAMPLE_PROBS[current + 1:]):
            if p < cutoff:
                return (f"In this example the current price sits in {now}. A price rise of about {pct}% takes the "
                        f"prediction below the {cutoff:.0%} threshold, into {_band_name(p, cutoff)}.")
    else:
        for pct, p in zip(reversed(_EXAMPLE_STEPS[:current]), reversed(_EXAMPLE_PROBS[:current])):
            if p >= cutoff:
                return (f"In this example the current price sits in {now}. A price cut of about {abs(pct)}% lifts "
                        f"the prediction to the {cutoff:.0%} threshold, into Comfortable Win.")
    return f"In this example the prediction stays in {now} across the whole price range shown."


def drivers_part_html() -> str:
    """Chart 1 on its own: title, where to find it, how to read it, the annotated bars."""
    return (
        "<div class='lp-chart-title'>1. Driver contribution chart</div>"
        f"{find_it(*FIND_DRIVERS)}"
        "<div class='lp-chart-lead'>Each bar is one factor. Blue bars (to the right) support the predicted chance "
        "of winning; red bars (to the left) reduce it. The longer the bar, the bigger its effect.</div>"
        f"{drivers_chart_html()}"
    )


def curve_part_html(metadata: dict[str, Any] | None) -> str:
    """Chart 2 on its own: title, where to find it, the labelled curve, tips and takeaway."""
    cutoff = product_cutoffs(metadata).get("Viresta", 0.56)
    tips = "".join(f"<li>{t}</li>" for t in (
        "<b>The coloured zones</b> match the outcome bands, and the dashed line is this product's win threshold.",
        "<b>A steeper line</b> means price matters more for this account.",
    ))
    return (
        "<div class='lp-chart-title'>2. Price-response curve</div>"
        f"{find_it(*FIND_CURVE)}"
        "<div class='lp-chart-lead'>The predicted win probability if the price moved by ±5, 10, 15 or 20%, with "
        "every other factor held as it is.</div>"
        f"<div class='lp-curve-wrap'>{price_curve_svg(cutoff)}</div>"
        f"<ul class='lp-curve-tips'>{tips}</ul>"
        f"<div class='lp-callout'>{escape(curve_takeaway(cutoff))}</div>"
    )


def drivers_card_html(metadata: dict[str, Any] | None) -> str:
    """Both charts in one card, marked as an illustrative example."""
    return (
        "<div class='lp-card'>"
        "<span class='lp-ribbon'>Illustrative example — not real customer data</span>"
        f"{drivers_part_html()}<div class='lp-chart-sep'></div>{curve_part_html(metadata)}"
        "</div>"
    )


# --- Quick guide ----------------------------------------------------------------

def features_table_html() -> str:
    rows = "".join(
        f"<tr><td><strong>{escape(name)}</strong></td><td>{escape(text)}</td>"
        f"<td class='lp-dim'>{escape(where)}</td></tr>"
        for name, text, where in FEATURES
    )
    return (
        "<table class='lp-ftable'><thead><tr><th>Feature</th><th>What it gives you</th><th>Where</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
