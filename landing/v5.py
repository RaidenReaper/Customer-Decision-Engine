"""Landing variant 5: guided walkthrough, one idea per slide.

Slides follow the user's workflow (blocks.JOURNEY). Reading the simulation and
the two driver charts are wide slides built from the shared blocks.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from landing.blocks import (
    QUICK_GUIDE,
    curve_part_html,
    drivers_part_html,
    features_table_html,
    find_it,
    simulation_card_html,
)
from landing import go_to
from landing.common import TAGLINE, TOOL_NAME, html, open_tool_button

STEP_LABELS = [
    "Welcome", "Choose account", "Run a scenario", "Read the result",
    "Driving factors", "Price curve", "Account characteristics", "Prepare",
]

_CSS = """
<style>
.v5-top {display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap; margin:.4rem 0 .9rem;}
.v5-progress {min-width:220px; flex:0 1 320px;}
.v5-progress-label {font-size:.78rem; font-weight:800; color:var(--text-secondary); text-align:right; margin-bottom:.35rem;}
.v5-progress-track {height:6px; border-radius:999px; background:#F3E3DB; overflow:hidden;}
.v5-progress-fill {height:100%; border-radius:999px; background:linear-gradient(90deg,#B25502,#EC7200);}
.st-key-v5-slide {background:var(--card); border:1px solid var(--border); border-radius:18px;
    padding:1.6rem 1.8rem 1.3rem; box-shadow:0 8px 24px rgba(46,17,0,.06);}
.v5-art {background:linear-gradient(160deg,#FDF0E9 0%,#FFFFFF 70%); border:1px solid #F8E7DD;
    border-radius:14px; padding:1.2rem; min-height:260px; display:flex; flex-direction:column; justify-content:center; gap:.7rem;}
.v5-kicker {font-size:.72rem; font-weight:900; letter-spacing:.06rem; text-transform:uppercase; color:var(--purple); margin-bottom:.3rem;}
.v5-title {font-size:1.55rem; font-weight:800; color:var(--text-primary); line-height:1.25; margin:0 0 .7rem;}
.v5-lines {margin:0 0 .6rem; padding:0; list-style:none;}
.v5-lines li {display:flex; gap:.6rem; font-size:1rem; line-height:1.5; color:var(--text-primary); margin-bottom:.55rem;}
.v5-lines li::before {content:""; flex:none; width:7px; height:7px; border-radius:50%; background:var(--purple); margin-top:.6rem;}
.v5-wide-head {display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom:.8rem;}
.v5-wide-head .v5-title {margin:0;}
.v5-wide-lead {font-size:.95rem; color:var(--text-secondary); line-height:1.5; margin:.35rem 0 0; max-width:820px;}
.v5-wide-body {border:1px solid #F8E7DD; border-radius:14px; padding:1rem 1.1rem; background:#FFFFFF;}
.v5-wide-body .lp-card {border:none; padding:0; background:transparent;}
/* mock UI pieces used in the illustrations */
.v5-mock-label {font-size:.7rem; font-weight:800; color:var(--text-primary);}
.v5-mock-select {background:#FFF; border:1px solid #DDD4CF; border-radius:9px; padding:.5rem .7rem; font-size:.85rem;
    display:flex; justify-content:space-between; align-items:center; color:var(--text-primary);}
.v5-mock-select::after {content:"⌄"; color:var(--purple); font-weight:800;}
.v5-chip {display:inline-block; background:#F7D7C5; color:#4A2002; border-radius:999px; padding:.15rem .55rem; font-size:.75rem; font-weight:600; margin-right:.3rem;}
.v5-mock-addr {font-size:.8rem; color:var(--text-secondary); display:flex; gap:.35rem; align-items:center; margin-top:-.2rem;}
.v5-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem;}
.v5-kpi {background:#FFF; border:1px solid var(--border); border-radius:10px; padding:.55rem .65rem;}
.v5-kpi b {display:block; font-size:.95rem; color:var(--text-primary);}
.v5-kpi span {font-size:.68rem; color:var(--text-secondary); text-transform:uppercase; font-weight:700;}
.v5-signal {display:flex; align-items:center; gap:.6rem; font-size:.78rem; color:var(--text-primary);}
.v5-signal-bar {flex:1; height:6px; border-radius:999px; background:#F3E8E1; position:relative;}
.v5-signal-bar i {position:absolute; top:-3px; width:12px; height:12px; border-radius:50%; background:var(--purple);}
.v5-lever {display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:.4rem; align-items:center; font-size:.78rem;}
.v5-lever div {background:#FFF; border:1px solid var(--border); border-radius:7px; padding:.3rem .45rem;}
.v5-lever .v5-changed {border-color:var(--purple); background:var(--light-purple); font-weight:700;}
.v5-lever .v5-head {background:none; border:none; font-weight:800; color:var(--text-secondary); font-size:.7rem; text-transform:uppercase;}
.v5-btns {display:flex; gap:.5rem; margin-top:.3rem;}
.v5-btn {flex:1; text-align:center; border-radius:9px; padding:.42rem; font-size:.8rem; font-weight:700;}
.v5-btn-primary {background:var(--purple); color:#FFF;}
.v5-btn-secondary {background:#FFF; border:1px solid #DDD4CF; color:var(--text-primary);}
.v5-flow {display:flex; align-items:center; justify-content:space-between; gap:.3rem;}
.v5-flow-step {display:flex; flex-direction:column; align-items:center; gap:.4rem; text-align:center; flex:1; font-size:.76rem; font-weight:700; color:var(--text-primary);}
.v5-flow-icon {width:52px; height:52px; border-radius:16px; background:#FFF; border:1px solid #E3DAD5; display:flex; align-items:center; justify-content:center; color:var(--purple); box-shadow:0 4px 12px rgba(236,114,0,.12);}
.v5-flow-arrow {color:#E4B397; font-weight:900; margin-bottom:1.3rem;}
.v5-doc {display:flex; align-items:center; gap:.7rem; background:#FFF; border:1px solid var(--border); border-radius:10px; padding:.6rem .75rem; font-size:.84rem; color:var(--text-primary);}
.v5-doc svg {flex:none; color:var(--purple);}
.v5-doc span {display:block; font-size:.72rem; color:var(--text-secondary);}
.v5-guide-title {font-size:1.05rem; font-weight:800; color:var(--text-primary); margin:1.4rem 0 .5rem;}
.st-key-v5-nav [data-testid="stHorizontalBlock"] {align-items:center;}
.st-key-v5-jump {margin-bottom:.6rem;}
@media (max-width: 1100px) {
  .st-key-v5-slide {padding:1.2rem 1.1rem 1rem;}
  .v5-title {font-size:1.3rem;}
  .v5-grid {grid-template-columns:repeat(2,1fr);}
}
</style>
"""

_ICON = {
    "account": "<path d='M3 21h18M5 21V8l7-5 7 5v13M9 21v-5h6v5'/>",
    "levers": "<path d='M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0'/><circle cx='16' cy='6' r='2'/><circle cx='10' cy='12' r='2'/><circle cx='18' cy='18' r='2'/>",
    "result": "<circle cx='12' cy='12' r='9'/><path d='M8 12l3 3 5-6'/>",
    "deep": "<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>",
    "prepare": "<path d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z'/><path d='M14 3v5h5M9 13h6M9 17h4'/>",
}


def _icon(name: str, size: int = 24) -> str:
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
        f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{_ICON[name]}</svg>"
    )


_PIN = (
    "<svg width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='#EC7200' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round'><path d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'/>"
    "<circle cx='12' cy='10' r='3'/></svg>"
)


def _art_welcome() -> str:
    steps = [("account", "Choose account"), ("levers", "Run a scenario"), ("result", "Read the result"),
             ("deep", "Account characteristics"), ("prepare", "Prepare")]
    parts = []
    for i, (icon, label) in enumerate(steps):
        if i:
            parts.append("<div class='v5-flow-arrow'>›</div>")
        parts.append(f"<div class='v5-flow-step'><div class='v5-flow-icon'>{_icon(icon)}</div>{label}</div>")
    return (
        "<div class='v5-art'><div class='v5-flow'>" + "".join(parts) + "</div>"
        "<div class='lp-note' style='text-align:center;margin-top:.6rem;'>Five steps, in the order you work.</div></div>"
    )


def _art_account() -> str:
    return (
        "<div class='v5-art'>"
        "<div class='v5-mock-label'>Health System</div>"
        "<div class='v5-mock-select'>Example Health System</div>"
        f"<div class='v5-mock-addr'>{_PIN} 100 Main St, Columbus, Ohio</div>"
        "<div class='v5-mock-label' style='margin-top:.4rem;'>Products</div>"
        "<div class='v5-mock-select'><span><span class='v5-chip'>Viresta</span><span class='v5-chip'>Immunexa</span>"
        "<span class='v5-chip'>Noventra</span></span></div>"
        "<div class='lp-note'>Illustrative example — not real customer data.</div>"
        "</div>"
    )


def _art_levers() -> str:
    rows = [("Dead net price", "&#36;20.00", "&#36;18.00 ✎", True), ("Share of wallet", "—", "—", False),
            ("Engagement", "—", "—", False), ("Negotiation stance", "—", "—", False)]
    body = "".join(
        f"<div>{name}</div><div>{a}</div><div class='{'v5-changed' if changed else ''}'>{b}</div>"
        for name, a, b, changed in rows
    )
    return (
        "<div class='v5-art'><div class='v5-lever'>"
        "<div class='v5-head'>Lever</div><div class='v5-head'>Current</div><div class='v5-head'>Your scenario</div>"
        f"{body}</div>"
        "<div class='v5-btns'><div class='v5-btn v5-btn-primary'>Run Simulation</div>"
        "<div class='v5-btn v5-btn-secondary'>Reset Inputs</div></div>"
        "<div class='lp-note'>Changed values are highlighted. Illustrative example — not real customer data.</div></div>"
    )


def _art_deep() -> str:
    kpis = [("Price / dose", "—"), ("Market opp.", "—"), ("Shipments", "—"),
            ("Market share", "—"), ("Price sens.", "—"), ("Quadrant", "—")]
    cards = "".join(f"<div class='v5-kpi'><span>{label}</span><b>{value}</b></div>" for label, value in kpis)
    signals = "".join(
        f"<div class='v5-signal'><span style='width:7.5rem'>{name}</span>"
        f"<div class='v5-signal-bar'><i style='left:{pos}%'></i></div></div>"
        for name, pos in [("Pricing", 30), ("Loyalty", 68), ("Relationship", 52), ("Clinical", 80)]
    )
    return f"<div class='v5-art'><div class='v5-grid'>{cards}</div>{signals}</div>"


def _art_prepare() -> str:
    doc = _icon("prepare", 22)
    return (
        "<div class='v5-art'>"
        f"<div class='v5-doc'>{doc}<div>Scenario report (PDF)<span>Download beside Contracting Levers</span></div></div>"
        f"<div class='v5-doc'>{doc}<div>Account-signals report (PDF)<span>Download in Customer Deep Dive</span></div></div>"
        "<div class='v5-btns'><div class='v5-btn v5-btn-primary'>Run Simulation</div>"
        "<div class='v5-btn v5-btn-secondary'>Reset Inputs</div></div>"
        "<div class='lp-note'>Try a few scenarios, then take the one you want into the meeting.</div></div>"
    )


def _slides(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Split slides (art + text) or wide slides (text on top, full-width content below)."""
    return [
        {"kicker": "Welcome", "title": "Prepare your next contracting conversation",
         "lines": [escape(TAGLINE), "This short walkthrough shows where everything is and how to read the results."],
         "art": _art_welcome()},
        {"kicker": "Step 1 of 5", "title": "Choose your account and products",
         "lines": ["Pick a <b>Health System</b> at the top of the page. Its address shows underneath.",
                   "Pick the <b>Products</b> you want to review. Everything below follows this choice."],
         "find": ("Top of the page", "Health System", "Products"), "art": _art_account()},
        {"kicker": "Step 2 of 5", "title": "Explore a different scenario",
         "lines": ["In <b>Contracting Levers</b>, change a value for a product, such as the price. "
                   "Change one thing at a time so you can see its effect.",
                   "Select <b>Run Simulation</b> to update the result. <b>Reset Inputs</b> takes you back to the "
                   "account's recorded values."],
         "find": ("Simulation tab", "Contracting Levers", "Run Simulation"), "art": _art_levers()},
        {"kicker": "Step 3 of 5 · Reading the simulation", "title": "Reading the simulation",
         "lead": "Every product gets a band and a confidence level. Read both before you act.",
         "find": ("Simulation tab", "Predicted Outcome"), "wide": simulation_card_html(metadata)},
        {"kicker": "Step 3 of 5 · Driving factors (1 of 2)", "title": "Understand the driving factors",
         "lead": "Which factors pushed this prediction up or down.",
         "wide": "<span class='lp-ribbon'>Illustrative example — not real customer data</span>" + drivers_part_html()},
        {"kicker": "Step 3 of 5 · Driving factors (2 of 2)", "title": "How price moves the chances of winning",
         "lead": "What the prediction would be at a lower or higher price.",
         "wide": "<span class='lp-ribbon'>Illustrative example — not real customer data</span>"
                 + curve_part_html(metadata)},
        {"kicker": "Step 4 of 5", "title": "Explore account characteristics",
         "lines": ["See the account's size, price per dose, market share, shipments and strategic quadrant.",
                   "Account signals show pricing, loyalty, the relationship with customer and clinical inclination, "
                   "each against other accounts.",
                   "The charts show how price and shipments have moved by season and brand."],
         "find": ("Customer Deep Dive tab",), "art": _art_deep()},
        {"kicker": "Step 5 of 5", "title": "Prepare your next discussion",
         "lines": ["Test another scenario, or select <b>Reset Inputs</b> to start again from the recorded values.",
                   "Read the <b>Narrative Summary</b> for the main points in everyday language.",
                   "Download the scenario report, and the account-signals report from Customer Deep Dive."],
         "find": ("Simulation tab", "Contracting Levers", "download button"), "art": _art_prepare()},
    ]


def _go(step: int) -> None:
    st.session_state["v5_step"] = step
    st.rerun()


def _sync_jump() -> None:
    picked = st.session_state.get("v5_jump")
    if picked in STEP_LABELS:
        st.session_state["v5_step"] = STEP_LABELS.index(picked)


def _full_guide() -> None:
    html("<div class='v5-guide-title'>Full guide</div>")
    for question, answer in QUICK_GUIDE:
        with st.expander(question):
            st.markdown(answer)
    with st.expander("What the tool provides"):
        html(features_table_html())


def _slide_text(slide: dict[str, Any]) -> str:
    lines = "".join(f"<li><span>{line}</span></li>" for line in slide["lines"])
    find = find_it(*slide["find"]) if slide.get("find") else ""
    return (
        f"<div class='v5-kicker'>{escape(slide['kicker'])}</div>"
        f"<div class='v5-title'>{escape(slide['title'])}</div>"
        f"<ul class='v5-lines'>{lines}</ul>{find}"
    )


def _wide_slide(slide: dict[str, Any]) -> str:
    find = find_it(*slide["find"]) if slide.get("find") else ""
    return (
        "<div class='v5-wide-head'><div>"
        f"<div class='v5-kicker'>{escape(slide['kicker'])}</div>"
        f"<div class='v5-title'>{escape(slide['title'])}</div>"
        f"<div class='v5-wide-lead'>{escape(slide['lead'])}</div></div>{find}</div>"
        f"<div class='v5-wide-body'>{slide['wide']}</div>"
    )


def render(metadata: dict[str, Any] | None) -> None:
    html(_CSS)
    slides = _slides(metadata)
    last = len(slides) - 1
    step = int(st.session_state.get("v5_step", 0))
    step = max(0, min(step, last))
    slide = slides[step]

    html(
        "<div class='v5-top'><div>"
        f"<div class='lp-eyebrow'>{escape(TOOL_NAME)}</div>"
        "<div class='lp-h1'>Choose an account. Test a scenario. Understand the result.</div></div>"
        "<div class='v5-progress'>"
        f"<div class='v5-progress-label'>{escape(slide['kicker'])}</div>"
        f"<div class='v5-progress-track'><div class='v5-progress-fill' style='width:{(step + 1) / len(slides):.0%};'></div></div>"
        "</div></div>"
    )

    st.session_state["v5_jump"] = STEP_LABELS[step]
    with st.container(key="v5-jump"):
        st.segmented_control(
            "Jump to step", STEP_LABELS, key="v5_jump", on_change=_sync_jump, label_visibility="collapsed"
        )

    with st.container(key="v5-slide"):
        if "wide" in slide:
            html(_wide_slide(slide))
        else:
            art, text = st.columns([1, 1.15], gap="large", vertical_alignment="center")
            with art:
                html(slide["art"])
            with text:
                html(_slide_text(slide))
                if step == last:
                    st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)
                    open_tool_button("Start exploring", key="v5-open-final")

    with st.container(key="v5-nav"):
        back, nxt, _, skip = st.columns([1, 1, 3.2, 1.3], gap="small")
        if back.button("Back", key="v5-back", icon=":material/arrow_back:", disabled=step == 0, width="stretch"):
            _go(step - 1)
        if step < last:
            if nxt.button("Next", key="v5-next", type="primary", icon=":material/arrow_forward:",
                          icon_position="right", width="stretch"):
                _go(step + 1)
        skip.button("Skip to the tool", key="v5-skip", type="tertiary", width="stretch",
                    on_click=go_to, args=(False,))

    if step == last:
        _full_guide()
