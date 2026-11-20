"""Landing variant 3 — "Learn through one example".

Walks a clearly fictional account through the user's workflow, so win
probability, confidence and drivers are seen in context before the real
tool opens. Shared wording and charts come from landing.blocks.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from landing.blocks import (
    QUICK_GUIDE,
    drivers_part_html,
    curve_part_html,
    find_it,
    simulation_card_html,
    win_vs_confidence_html,
)
from landing.common import (
    BAND_RULES,
    BANDS,
    CONFIDENCE_LEVELS,
    GLOSSARY,
    TOOL_NAME,
    band_pill,
    confidence_pill,
    html,
    open_tool_button,
)

_RIBBON = "<span class='lp-ribbon'>Illustrative example — not real customer data</span>"

_CSS = """
<style>
.v3-hero {padding: .6rem 0 .4rem;}
.v3-step {display: grid; grid-template-columns: 54px minmax(0, 1fr) minmax(0, 1.15fr);
  gap: 1.1rem; align-items: start; position: relative; padding-bottom: 1.6rem;}
.v3-step::before {content: ""; position: absolute; left: 26px; top: 46px; bottom: 0;
  width: 2px; background: var(--border);}
.v3-step-last::before {display: none;}
/* Wide step: text on top, the visual spans the text and visual columns. */
.v3-step-wide .v3-visual {grid-column: 2 / span 2;}
.v3-num {width: 44px; height: 44px; border-radius: 50%; background: var(--purple); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem;
  box-shadow: 0 4px 12px rgba(236, 114, 0, .25); position: relative; z-index: 1;}
.v3-text {padding-top: .35rem;}
.v3-text p {font-size: .92rem; color: var(--text-secondary); line-height: 1.5; margin: .2rem 0 .55rem;}
.v3-visual {background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  padding: .85rem 1rem 1rem; position: relative;}
.v3-pick {display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: .7rem;}
.v3-field-label {font-size: .72rem; font-weight: 800; color: var(--text-primary); margin-bottom: .25rem;}
.v3-field {border: 1px solid var(--border); border-radius: 8px; padding: .45rem .65rem; background: #fff;
  font-size: .84rem; color: var(--text-primary); display: flex; flex-wrap: wrap; gap: .3rem; align-items: center;}
.v3-chip {background: var(--light-purple); color: var(--purple); font-size: .74rem; font-weight: 700;
  border-radius: 6px; padding: .1rem .45rem;}
.v3-kpis {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .55rem;}
.v3-kpi {border: 1px solid var(--border); border-radius: 10px; padding: .55rem .65rem; background: #FCFBFE;}
.v3-kpi-label {font-size: .66rem; font-weight: 800; text-transform: uppercase; color: var(--text-secondary);}
.v3-kpi-value {font-size: 1.05rem; font-weight: 800; color: var(--text-primary); margin-top: .15rem;}
.v3-outcome {display: grid; grid-template-columns: minmax(0, 200px) minmax(0, 1fr); gap: 1rem; align-items: start;}
.v3-card {border: 1px solid var(--border); border-radius: 12px; padding: .8rem .9rem; background: #fff;
  box-shadow: 0 2px 8px rgba(31, 31, 41, .04);}
.v3-card-name {font-size: .78rem; font-weight: 800; color: var(--text-primary); margin-bottom: .55rem;}
.v3-card-conf {display: flex; justify-content: space-between; align-items: center; margin-top: .75rem;
  padding-top: .55rem; border-top: 1px solid var(--border); font-size: .76rem; font-weight: 700;
  color: var(--text-secondary);}
.v3-callouts {display: flex; flex-direction: column; gap: .6rem;}
.v3-callout {display: flex; gap: .55rem; align-items: flex-start; font-size: .84rem; color: var(--text-primary); line-height: 1.4;}
.v3-callout small {display: block; font-size: .76rem; color: var(--text-secondary); margin-top: .15rem;}
.v3-badge {flex: none; width: 22px; height: 22px; border-radius: 50%; background: var(--light-purple);
  color: var(--purple); font-size: .75rem; font-weight: 800; display: flex; align-items: center; justify-content: center;}
.v3-mark {position: relative;}
.v3-mark .v3-badge {position: absolute; top: -9px; right: -11px; width: 18px; height: 18px; font-size: .66rem;
  background: var(--purple); color: #fff;}
.v3-lever {display: flex; align-items: center; justify-content: space-between; gap: .8rem;
  border: 1px solid var(--border); border-radius: 10px; padding: .6rem .8rem; background: #FCFBFE;}
.v3-lever-name {font-size: .84rem; font-weight: 700;}
.v3-lever-vals {font-size: .92rem; font-weight: 800;}
.v3-lever-vals s {color: var(--text-secondary); font-weight: 600; margin-right: .35rem;}
.v3-lever-vals b {color: var(--purple); background: var(--light-purple); border-radius: 6px; padding: .1rem .4rem;}
.v3-fake-btn {margin-top: .7rem; display: inline-flex; align-items: center; gap: .4rem; background: var(--purple);
  color: #fff; font-size: .82rem; font-weight: 700; padding: .45rem 1rem; border-radius: 8px; cursor: default;
  opacity: .92;}
.v3-teach {margin-top: .7rem; font-size: .78rem; color: #8A5A00; background: #FFFBEB; border-radius: 8px;
  padding: .5rem .65rem; line-height: 1.45;}
.v3-explore {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .55rem;}
.v3-explore div {border: 1px solid var(--border); border-radius: 10px; padding: .6rem .7rem; font-size: .8rem;
  color: var(--text-secondary); line-height: 1.4; background: #FCFBFE;}
.v3-explore strong {display: block; color: var(--text-primary); font-size: .84rem; margin-bottom: .15rem;}
.v3-terms {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem; margin-top: .6rem;}
.v3-term {background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: .75rem .85rem;}
.v3-term-name {font-size: .86rem; font-weight: 800; color: var(--text-primary); margin-bottom: .2rem;}
.v3-term-text {font-size: .82rem; color: var(--text-secondary); line-height: 1.45;}
.v3-section-gap {height: 1.4rem;}
@media (max-width: 1100px) {
  .v3-step {grid-template-columns: 54px minmax(0, 1fr);}
  .v3-visual, .v3-step-wide .v3-visual {grid-column: 2;}
  .v3-terms {grid-template-columns: repeat(2, minmax(0, 1fr));}
}
@media (max-width: 760px) {
  .v3-kpis, .v3-explore, .v3-pick {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .v3-outcome, .v3-terms {grid-template-columns: minmax(0, 1fr);}
}
</style>
"""


def _step(number: int, title: str, body: str, visual: str, *, find: str = "",
          last: bool = False, wide: bool = False) -> str:
    classes = ["v3-step"] + (["v3-step-last"] if last else []) + (["v3-step-wide"] if wide else [])
    return (
        f"<div class='{' '.join(classes)}'>"
        f"<div class='v3-num'>{number}</div>"
        f"<div class='v3-text'><div class='lp-h2'>{escape(title)}</div><p>{body}</p>{find}</div>"
        f"<div class='v3-visual'>{_RIBBON}{visual}</div>"
        "</div>"
    )


def _band(label: str) -> tuple[str, str, str]:
    """(kind, plain meaning, rule) for a band label."""
    kind, meaning = next((k, m) for name, k, m in BANDS if name == label)
    return kind, meaning, BAND_RULES[label]


def _outcome_card(band_label: str, confidence: str) -> str:
    kind = _band(band_label)[0]
    band = f"<span class='v3-mark'>{band_pill(band_label, kind)}<span class='v3-badge'>1</span></span>"
    conf = f"<span class='v3-mark'>{confidence_pill(confidence)}<span class='v3-badge'>2</span></span>"
    return (
        "<div class='v3-card'>"
        "<div class='v3-card-name'>Viresta</div>"
        f"{band}"
        f"<div class='v3-card-conf'><span>Confidence</span>{conf}</div>"
        "</div>"
    )


def render(metadata: dict[str, Any] | None) -> None:
    html(_CSS)

    html(
        "<div class='v3-hero'>"
        f"<div class='lp-eyebrow'>{escape(TOOL_NAME)}</div>"
        "<div class='lp-h1'>See how an account review becomes a contracting decision.</div>"
        "<div class='lp-lead'>Follow one simple example, then explore your own customer.</div>"
        "</div>"
    )
    open_tool_button("Try your own account", key="v3-open-top")
    html("<div class='v3-section-gap'></div>")

    pick_visual = (
        "<div class='v3-pick'>"
        "<div><div class='v3-field-label'>Health System</div>"
        "<div class='v3-field'>Example Health System</div></div>"
        "<div><div class='v3-field-label'>Products</div>"
        "<div class='v3-field'><span class='v3-chip'>Viresta</span><span class='v3-chip'>Immunexa</span></div></div>"
        "</div>"
    )
    change_visual = (
        "<div class='v3-field-label'>Contracting Levers</div>"
        "<div class='v3-lever'><span class='v3-lever-name'>Dead Net Price · Viresta</span>"
        "<span class='v3-lever-vals'><s>&#36;20.00</s><b>&#36;18.00</b></span></div>"
        "<span class='v3-fake-btn'>▶ Run Simulation</span>"
    )
    band_label = "Comfortable Win"
    band_meaning = _band(band_label)[1]
    confidence = "Moderate"
    confidence_meaning = dict(CONFIDENCE_LEVELS)[confidence]
    result_visual = (
        "<div class='v3-outcome'>"
        f"{_outcome_card(band_label, confidence)}"
        "<div class='v3-callouts'>"
        f"<div class='v3-callout'><span class='v3-badge'>1</span><span><b>Win probability: {band_label}.</b> "
        f"{escape(band_meaning)}</span></div>"
        f"<div class='v3-callout'><span class='v3-badge'>2</span><span><b>Confidence: {confidence}.</b> "
        f"{escape(confidence_meaning)}</span></div>"
        "</div></div>"
        f"{win_vs_confidence_html()}"
        "<div class='v3-teach'>Teaching example only. It does not claim that this price change "
        "produces this result for any real account.</div>"
    )
    drivers_visual = f"{drivers_part_html()}<div class='lp-chart-sep'></div>{curve_part_html(metadata)}"
    kpis = "".join(
        f"<div class='v3-kpi'><div class='v3-kpi-label'>{label}</div><div class='v3-kpi-value'>{value}</div></div>"
        for label, value in [
            ("Price per dose", "&#36;20.00"),
            ("Market opportunity", "100K doses"),
            ("Price sensitivity", "High"),
            ("Relationship", "Strong"),
        ]
    )
    prepare_visual = (
        "<div class='v3-explore'>"
        "<div><strong>Download the report</strong>Save the scenario report as a PDF, or the account-signals "
        "report from Customer Deep Dive.</div>"
        "<div><strong>Try another scenario</strong>Change a lever and select Run Simulation again to compare.</div>"
        "<div><strong>Start again</strong>Reset Inputs returns every lever to the account's recorded values.</div>"
        "</div>"
    )

    html(
        "<div class='lp-h2' style='margin-bottom:.9rem;'>Follow Example Health System</div>"
        + _step(1, "Choose your account and products",
                "Pick the health system and the products you want to review. Everything that follows is "
                "for this account and these products.",
                pick_visual, find=find_it("Top of the page", "Health System", "Products"))
        + _step(2, "Explore a different scenario",
                "Adjust a lever, here the proposed price, then select <b>Run Simulation</b>. "
                "Change one thing at a time so you can see its effect.",
                change_visual, find=find_it("Simulation tab", "Contracting Levers", "Run Simulation"))
        + _step(3, "Read the simulation",
                "Each product gets two labels. <b>Win probability</b> shows how likely we are to win this "
                "account, as a band against the product's own cutoff. <b>Confidence</b> shows how much we can "
                "rely on that prediction.",
                result_visual, find=find_it("Simulation tab", "Predicted Outcome"))
        + _step(4, "Understand the driving factors",
                "Two charts explain the prediction: which factors push the chances of winning up or down, and "
                "how the chances move if the price changes.",
                drivers_visual, wide=True)
        + _step(5, "Explore account characteristics",
                "Look at who the customer is: size, price, buying behaviour and how the account relates to "
                "us. This is recorded account data, so the simulation does not change it.",
                f"<div class='v3-kpis'>{kpis}</div>", find=find_it("Customer Deep Dive tab"))
        + _step(6, "Prepare your next discussion",
                "Compare a few scenarios, start again if you need to, and take the findings into the meeting.",
                prepare_visual, find=find_it("Simulation tab", "Contracting Levers"), last=True)
    )

    terms = "".join(
        f"<div class='v3-term'><div class='v3-term-name'>{escape(name)}</div>"
        f"<div class='v3-term-text'>{escape(text)}</div></div>"
        for name, text in GLOSSARY
    )
    html(
        "<div class='v3-section-gap'></div>"
        "<div class='lp-h2'>Terms at a glance</div>"
        f"<div class='v3-terms'>{terms}</div>"
        "<div class='v3-section-gap'></div>"
    )

    html("<div class='lp-h3'>Common questions</div>")
    for question, answer in QUICK_GUIDE:
        with st.expander(question):
            st.markdown(answer)
        # Workflow order: reading the result comes right after changing the levers.
        if question == "What can I change?":
            with st.expander("How do I read the simulation result?"):
                html(
                    "<div class='lp-sim-lead'>After you select Run Simulation, each product shows two labels: "
                    "an outcome band for its win probability, and a confidence level. Read them together.</div>"
                    + find_it("Simulation tab", "Predicted Outcome")
                    + simulation_card_html(metadata)
                )

    html("<div class='v3-section-gap'></div>")
    open_tool_button("Try your own account", key="v3-open-bottom")
