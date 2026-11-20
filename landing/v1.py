"""Landing variant 1: "Your contracting journey" — five steps, then how to read a result."""
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
from landing.common import TAGLINE, TOOL_NAME, html, open_tool_button

_CSS = """
<style>
.v1-hero {padding: 1.6rem 1.8rem 1.4rem; border-radius: 16px; margin-bottom: .9rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #FDEFE7 100%); border: 1px solid var(--border);}
.v1-hero .lp-h1 {max-width: 900px;}
.v1-section {font-size: .72rem; font-weight: 900; letter-spacing: .06rem; text-transform: uppercase;
    color: var(--purple); margin: 1.6rem 0 .7rem;}
.v1-flow {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; position: relative;}
.v1-flow::before {content: ""; position: absolute; top: 22px; left: 10%; right: 10%; height: 3px;
    background: linear-gradient(90deg, #EAA57C, var(--purple)); border-radius: 3px; z-index: 0;}
.v1-step {position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; text-align: center;}
.v1-num {width: 46px; height: 46px; border-radius: 50%; background: var(--purple); color: #fff;
    font-size: 1.15rem; font-weight: 800; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 0 5px var(--bg), 0 6px 14px rgba(236, 114, 0, .28); margin-bottom: .7rem;}
.v1-step-card {background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: .9rem .9rem .8rem; width: 100%; flex: 1; display: flex; flex-direction: column;}
.v1-step-card .lp-h3 {margin-bottom: .35rem;}
.v1-step-text {font-size: .86rem; color: var(--text-secondary); line-height: 1.45; flex: 1;}
.v1-where {display: inline-block; margin-top: .65rem; align-self: center; font-size: .7rem; font-weight: 800;
    color: var(--purple); background: var(--light-purple); border-radius: 999px; padding: .18rem .6rem;}
@media (max-width: 1100px) {
    .v1-flow {grid-template-columns: 1fr; gap: .7rem;}
    .v1-flow::before {top: 44px; bottom: 44px; left: 22px; right: auto; width: 3px; height: auto;
        background: linear-gradient(180deg, #EAA57C, var(--purple));}
    .v1-step {flex-direction: row; align-items: stretch; text-align: left; gap: .9rem;}
    .v1-num {margin-bottom: 0; flex: none; align-self: center;}
    .v1-where {align-self: flex-start;}
}
</style>
"""


def _steps_html() -> str:
    cards = "".join(
        f"<div class='v1-step'><div class='v1-num'>{i}</div>"
        f"<div class='v1-step-card'><div class='lp-h3'>{escape(title)}</div>"
        f"<div class='v1-step-text'>{escape(text)}</div>"
        f"<span class='v1-where'>Where: {escape(where)}</span></div></div>"
        for i, (title, text, where) in enumerate(JOURNEY, start=1)
    )
    return f"<div class='v1-flow'>{cards}</div>"


def render(metadata: dict[str, Any] | None) -> None:
    html(_CSS)

    html(
        "<div class='v1-hero'>"
        f"<div class='lp-eyebrow'>{escape(TOOL_NAME)}</div>"
        "<div class='lp-h1'>Understand the account. Explore your options. Prepare your next move.</div>"
        f"<div class='lp-lead'>{escape(TAGLINE)}</div>"
        "</div>"
    )
    open_tool_button("Explore an account", key="v1-open-top")

    html("<div class='v1-section'>Your contracting journey</div>")
    html(_steps_html())

    html(section_title(*SECTION_SIMULATION))
    html(simulation_card_html(metadata))

    html(section_title(*SECTION_DRIVERS))
    html(drivers_card_html(metadata))

    html("<div class='v1-section'>Quick guide</div>")
    for question, answer in QUICK_GUIDE:
        with st.expander(question):
            st.markdown(answer)
    with st.expander("What the tool provides"):
        html(features_table_html())

    with st.container(key="v1-bottom-cta"):
        html("<style>.st-key-v1-bottom-cta {align-items: center; margin-top: 1.4rem;}</style>")
        open_tool_button("Explore an account", key="v1-open-bottom")
