"""Variant 4 — "Your contracting briefing": a business-style cover page."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from landing.blocks import (
    JOURNEY,
    QUICK_GUIDE,
    SECTION_DRIVERS,
    SECTION_SIMULATION,
    drivers_card_html,
    features_table_html,
    section_title,
    simulation_card_html,
)
from landing.common import TOOL_NAME, html, open_tool_button

_ICON = (
    "<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{}</svg>"
)
# One icon per JOURNEY step, in the same order.
_STEP_ICONS = [
    # Choose your account and products
    _ICON.format("<path d='M3 21h18'/><path d='M5 21V7l7-4 7 4v14'/><path d='M9 21v-6h6v6'/>"),
    # Explore a different scenario
    _ICON.format("<path d='M4 21v-7'/><path d='M4 10V3'/><path d='M12 21v-9'/><path d='M12 8V3'/>"
                 "<path d='M20 21v-5'/><path d='M20 12V3'/><path d='M1 14h6'/><path d='M9 8h6'/>"
                 "<path d='M17 16h6'/>"),
    # Read what changed — and why
    _ICON.format("<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>"),
    # Explore account characteristics
    _ICON.format("<circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/>"),
    # Prepare your next discussion
    _ICON.format("<path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/>"
                 "<path d='M8 9h8M8 13h5'/>"),
]

_CSS = """
<style>
.st-key-v4-hero-wrap {background: linear-gradient(120deg, #FFFFFF 0%, #FCECE3 55%, #F9DAC8 100%);
    border:1px solid var(--border); border-radius:18px; padding:2rem 2.2rem 1.6rem; position:relative; overflow:hidden;}
.st-key-v4-hero-wrap::after {content:''; position:absolute; right:-90px; top:-90px; width:320px; height:320px;
    border-radius:50%; background:radial-gradient(circle, rgba(236,114,0,.18) 0%, rgba(236,114,0,0) 70%);
    pointer-events:none;}
.v4-hero .lp-h1 {font-size:2.05rem; max-width:760px;}
.v4-hero .lp-lead {margin-bottom:.2rem;}
.v4-steps {display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:.9rem; margin-top:.6rem;}
.v4-section {box-sizing:border-box; display:flex; flex-direction:column; gap:.55rem; padding:1.05rem 1.05rem;
    border-top:4px solid var(--purple);}
.v4-section-head {display:flex; align-items:center; gap:.6rem;}
.v4-icon {flex:none; width:40px; height:40px; border-radius:12px; background:var(--light-purple);
    color:var(--purple); display:flex; align-items:center; justify-content:center;}
.v4-num {font-size:.7rem; font-weight:900; letter-spacing:.06rem; color:var(--purple); text-transform:uppercase;}
.v4-where {display:inline-block; font-size:.7rem; font-weight:800; color:var(--text-secondary);
    background:var(--bg); border:1px solid var(--border); border-radius:999px; padding:.12rem .55rem;}
.v4-section p {margin:0; color:var(--text-secondary); font-size:.88rem; line-height:1.5;}
.v4-cta {display:flex; flex-direction:column; align-items:center; text-align:center; gap:.2rem; padding:1.3rem 1rem;}
.st-key-v4-cta-wrap {align-items:center;}
@media (max-width: 1300px) {
    .v4-steps {grid-template-columns:repeat(3, minmax(0, 1fr));}
}
@media (max-width: 900px) {
    .v4-steps {grid-template-columns:minmax(0, 1fr);}
}
</style>
"""


def _step_card(num: int, icon: str, title: str, text: str, where: str) -> str:
    return (
        "<div class='lp-card v4-section'>"
        "<div class='v4-section-head'>"
        f"<div class='v4-icon'>{icon}</div>"
        f"<div><div class='v4-num'>Step {num}</div><div class='lp-h3' style='margin:0;'>{escape(title)}</div></div>"
        "</div>"
        f"<div><span class='v4-where'>{escape(where)}</span></div>"
        f"<p>{escape(text)}</p>"
        "</div>"
    )


def _steps_html() -> str:
    cards = "".join(
        _step_card(i, icon, title, text, where)
        for i, (icon, (title, text, where)) in enumerate(zip(_STEP_ICONS, JOURNEY), start=1)
    )
    return f"<div class='v4-steps'>{cards}</div>"


def render(metadata: dict[str, Any] | None) -> None:
    html(_CSS)

    with st.container(key="v4-hero-wrap"):
        html(
            "<div class='v4-hero'>"
            f"<div class='lp-eyebrow'>{escape(TOOL_NAME)}</div>"
            "<div class='lp-h1'>A clearer view before your next contracting conversation.</div>"
            "<div class='lp-lead'>See where the account stands, test different offers, and understand "
            "what is driving the predicted outcome.</div>"
            "</div>"
        )
        open_tool_button("Review a health system", key="v4-open-top")

    html(_steps_html())

    html(section_title(*SECTION_SIMULATION))
    html(simulation_card_html(metadata))

    html(section_title(*SECTION_DRIVERS))
    html(drivers_card_html(metadata))

    html(section_title("Feature guide"))
    with st.expander("Everything the tool provides"):
        html(features_table_html())
    with st.expander("How to use it — common questions"):
        for question, answer in QUICK_GUIDE:
            st.markdown(f"**{question}**  \n{answer}")

    with st.container(key="v4-cta-wrap"):
        html(
            "<div class='v4-cta'><div class='lp-h2'>Ready to prepare your next conversation?</div>"
            "<div class='lp-note'>Pick a health system and products, then run your first scenario.</div></div>"
        )
        open_tool_button("Review a health system", key="v4-open-bottom")
