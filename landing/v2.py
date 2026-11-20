"""Variant 2 — "Start with your question": question cards in workflow order."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from landing.blocks import (
    QUICK_GUIDE,
    curve_part_html,
    drivers_part_html,
    find_it,
    simulation_card_html,
)
from landing.common import (
    BANDS,
    CONFIDENCE_LEVELS,
    TOOL_NAME,
    WIN_VS_CONFIDENCE,
    band_pill,
    confidence_pill,
    cutoff_sentence,
    html,
    open_tool_button,
)

_SVG = (
    "<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{}</svg>"
)
_ICONS = {
    "customer": _SVG.format("<path d='M3 21h18'/><path d='M5 21V7l7-4 7 4v14'/><path d='M9 21v-6h6v6'/><path d='M12 8v3M10.5 9.5h3'/>"),
    "offer": _SVG.format("<path d='M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3'/><path d='M1 14h6M9 8h6M17 16h6'/>"),
    "result": _SVG.format("<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>"),
    "why": _SVG.format("<circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/><path d='M8 13v-2M11 13V8M14 13v-3'/>"),
    "discussion": _SVG.format("<path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/><path d='M8 9h8M8 13h5'/>"),
}

_BAND_SHORT = {
    "Comfortable Win": "At or above the cutoff",
    "Watch": "Up to 20 pts below",
    "At Risk": "Over 20 pts below",
}

_GUIDE = dict(QUICK_GUIDE)

# (key, title, what you'll find, how to use it, what it tells you, click path, guide questions)
_OFFER = (
    "offer", "Test a different offer",
    "Six contracting levers, one column per product.",
    "Change the price or an assumption, then select Run Simulation.",
    "How the prediction responds to the change you made.",
    ("Simulation tab", "Contracting Levers", "Run Simulation"),
    ["What can I change?", "How do I try another scenario?"],
)
_RESULT = (
    "result", "Read my simulation result",
    "An outcome band and a confidence level for each product, plus the Narrative Summary.",
    "Read the band and the confidence together before you act.",
    "How likely we are to win this account, and how much to rely on that.",
    ("Simulation tab", "Predicted Outcome"),
    [],
)
_DRIVERS = (
    "why", "Understand the driving factors",
    "A driver contribution chart and a price-response curve.",
    "After Run Simulation, open the SHAP waterfall section under the contribution chart.",
    "Which factors improve or reduce the chances of winning, and how the chances change with price.",
)
_ACCOUNT = (
    "customer", "Explore account characteristics",
    "Overview cards, account signals and price and shipment history.",
    "Pick a health system, then open the Customer Deep Dive tab.",
    "Where the account stands today: size, pricing, loyalty and relationship.",
    ("Customer Deep Dive tab", "Account signals"),
    ["What can I learn about my customer?"],
)
_DISCUSSION = (
    "discussion", "Prepare for a discussion",
    "Alternative scenarios and downloadable PDF reports.",
    "Compare a few scenarios, then download the scenario report.",
    "A record of the scenario to take into the meeting.",
    ("Simulation tab", "Contracting Levers", "download button"),
    ["How do I take the findings away?"],
)

_CSS = """
<style>
.v2-hero {padding: 1.6rem 1.8rem 4.6rem; border-radius: 16px; border: 1px solid var(--border);
    background: linear-gradient(120deg, #FFFFFF 0%, var(--light-purple) 100%);}
.v2-hero .lp-h1 {font-size: 2rem; margin-top: .35rem;}
/* Button drawn inside the hero card: pulled up into its bottom padding. */
.st-key-v2-hero {gap: 0 !important;}
.st-key-v2-hero [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
    margin-top: -3.45rem; padding-left: 1.8rem; position: relative; z-index: 2; margin-bottom: 1.3rem;}
.st-key-v2-hero [data-testid="stButton"] button {padding: .55rem 1.2rem; border-radius: 10px;}
.v2-strip {display: flex; flex-wrap: wrap; align-items: center; gap: .55rem 1.4rem;
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: .7rem 1rem; margin: .9rem 0 1.1rem;}
.v2-strip-title {font-size: .72rem; font-weight: 900; letter-spacing: .05rem; text-transform: uppercase;
    color: var(--text-primary); margin-right: .2rem;}
.v2-strip-group {display: flex; flex-wrap: wrap; align-items: center; gap: .45rem .9rem;}
.v2-strip-item {display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem; color: var(--text-secondary);}
.v2-strip-sep {width: 1px; align-self: stretch; background: var(--border);}
.v2-strip-cutoff {flex-basis: 100%; font-size: .76rem; color: var(--text-secondary);
    padding-top: .45rem; border-top: 1px dashed var(--border);}
.v2-card {background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.15rem 1.25rem 1rem; height: 100%; box-shadow: 0 4px 14px rgba(46, 17, 0, .05);}
.v2-card-head {display: flex; align-items: center; gap: .75rem; margin-bottom: .85rem;}
.v2-icon {flex: none; width: 42px; height: 42px; border-radius: 12px; display: flex;
    align-items: center; justify-content: center; background: var(--light-purple); color: var(--purple);}
.v2-card-title {font-size: 1.12rem; font-weight: 800; color: var(--text-primary);}
.v2-row {display: grid; grid-template-columns: 128px 1fr; gap: .6rem; padding: .45rem 0;
    border-top: 1px dashed var(--border); font-size: .88rem; color: var(--text-primary); line-height: 1.4;}
.v2-row-label {font-size: .7rem; font-weight: 900; text-transform: uppercase; letter-spacing: .03rem;
    color: var(--purple); padding-top: .12rem;}
.v2-card .lp-find {margin: .7rem 0 0;}
.v2-charts {margin-top: 1rem; padding-top: .9rem; border-top: 1px solid var(--border);}
div[class*="st-key-v2-card-"] [data-testid="stExpander"] {margin-top: .55rem;}
div[class*="st-key-v2-card-"] [data-testid="stExpander"] p {font-size: .9rem;}
div[class*="st-key-v2-card-"] [data-testid="stExpander"] details {border-radius: 12px; background: #FCFBFE;}
.v2-myth {border-left: 4px solid var(--purple); background: var(--card); border-radius: 10px;
    border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    padding: .7rem .9rem; margin: 1.1rem 0 .4rem;}
.v2-myth strong {display: block; font-size: .9rem; color: var(--text-primary); margin-bottom: .15rem;}
.v2-myth span {font-size: .84rem; color: var(--text-secondary);}
.st-key-v2-cta {align-items: center; margin-top: 1rem;}
@media (max-width: 1100px) {
    .v2-row {grid-template-columns: 110px 1fr;}
}
</style>
"""


def _strip(metadata: dict[str, Any] | None) -> str:
    bands = "".join(
        f"<span class='v2-strip-item'>{band_pill(label, kind)}{escape(_BAND_SHORT[label])}</span>"
        for label, kind, _ in BANDS
    )
    confidence = "".join(
        f"<span class='v2-strip-item'>{confidence_pill(level)}</span>" for level, _ in CONFIDENCE_LEVELS
    )
    return (
        "<div class='v2-strip'>"
        "<span class='v2-strip-title'>How to read results</span>"
        f"<div class='v2-strip-group'>{bands}</div>"
        "<span class='v2-strip-sep'></span>"
        "<div class='v2-strip-group'><span class='v2-strip-item'><b>Confidence</b></span>"
        f"{confidence}</div>"
        f"<div class='v2-strip-cutoff'>Each product has its own winning cutoff. "
        f"{escape(cutoff_sentence(metadata))}</div>"
        "</div>"
    )


def _rows(find: str, use: str, tells: str) -> str:
    return "".join(
        f"<div class='v2-row'><span class='v2-row-label'>{label}</span><span>{escape(text)}</span></div>"
        for label, text in (("What you'll find", find), ("How to use it", use), ("What it tells you", tells))
    )


def _head(key: str, title: str) -> str:
    return (
        f"<div class='v2-card-head'><span class='v2-icon'>{_ICONS[key]}</span>"
        f"<span class='v2-card-title'>{escape(title)}</span></div>"
    )


def _card(key: str, title: str, find: str, use: str, tells: str, path: tuple[str, ...]) -> str:
    return f"<div class='v2-card'>{_head(key, title)}{_rows(find, use, tells)}{find_it(*path)}</div>"


def _question_card(spec: tuple, metadata: dict[str, Any] | None) -> None:
    key, title, find, use, tells, path, questions = spec
    with st.container(key=f"v2-card-{key}"):
        html(_card(key, title, find, use, tells, path))
        with st.expander("More detail"):
            for question in questions:
                st.markdown(f"**{question}**  \n{_GUIDE[question]}")
            if key == "result":
                html(simulation_card_html(metadata, narrow=True))


def _drivers_card(metadata: dict[str, Any] | None) -> str:
    key, title, find, use, tells = _DRIVERS
    # The two chart parts carry their own "Where to find it" pills.
    return (
        f"<div class='v2-card'>{_head(key, title)}{_rows(find, use, tells)}"
        "<div class='v2-charts'>"
        "<span class='lp-ribbon'>Illustrative example — not real customer data</span>"
        f"{drivers_part_html()}<div class='lp-chart-sep'></div>{curve_part_html(metadata)}"
        "</div></div>"
    )


def render(metadata: dict[str, Any] | None) -> None:
    html(_CSS)

    with st.container(key="v2-hero"):
        html(
            "<div class='v2-hero'>"
            f"<div class='lp-eyebrow'>{escape(TOOL_NAME)}</div>"
            "<div class='lp-h1'>Start with your contracting question.</div>"
            "<div class='lp-lead'>Test an offer, read the result, see what is driving it, "
            "and understand the account behind it.</div>"
            "</div>"
        )
        open_tool_button("Open the tool", key="v2-open")

    html(_strip(metadata))

    columns = st.columns(2, gap="medium")
    for column, spec in zip(columns, (_OFFER, _RESULT)):
        with column:
            _question_card(spec, metadata)
    html("<div style='height:.4rem;'></div>")

    with st.container(key="v2-card-why"):
        html(_drivers_card(metadata))
    html("<div style='height:.4rem;'></div>")

    columns = st.columns(2, gap="medium")
    for column, spec in zip(columns, (_ACCOUNT, _DISCUSSION)):
        with column:
            _question_card(spec, metadata)

    head, body = WIN_VS_CONFIDENCE
    html(f"<div class='v2-myth'><strong>{escape(head)}</strong><span>{escape(body)}</span></div>")

    with st.container(key="v2-cta"):
        open_tool_button("Open the tool", key="v2-open-bottom")
