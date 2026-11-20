"""Shared copy and helpers for the landing-page variants (v1-v5).

Every variant explains the same tool, so terms, cutoffs and cautions live
here once; each variant only decides layout and tone.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

TOOL_NAME = "Customer Response Prediction Engine"
TAGLINE = (
    "Understand your health system, explore different offers, and see what may "
    "influence the predicted outcome for Viresta, Immunexa and Noventra."
)

_PRODUCT_LABELS = {"HD": "Viresta", "STD": "Immunexa", "FLK": "Noventra"}
_FALLBACK_CUTOFFS = {"Viresta": 0.56, "Immunexa": 0.49, "Noventra": 0.48}


def product_cutoffs(metadata: dict[str, Any] | None) -> dict[str, float]:
    """Winning cutoff per product, read from the model metadata."""
    products = (metadata or {}).get("products", {})
    cutoffs = {
        _PRODUCT_LABELS[key]: float(info["classification_threshold"])
        for key, info in products.items()
        if key in _PRODUCT_LABELS and "classification_threshold" in info
    }
    return cutoffs or dict(_FALLBACK_CUTOFFS)


def cutoff_sentence(metadata: dict[str, Any] | None) -> str:
    parts = [f"{name} {value:.0%}" for name, value in product_cutoffs(metadata).items()]
    return "Current cutoffs: " + ", ".join(parts) + "."


# What the tool provides: (feature, plain-language explanation, where it lives)
FEATURES: list[tuple[str, str, str]] = [
    ("Health system and products", "Choose the customer and the brands you want to explore.", "Top of every page"),
    ("Account overview cards", "Price per dose, market opportunity, shipments, prebooks, market share, price sensitivity and strategic quadrant.", "Customer Deep Dive"),
    ("Account signals", "Pricing, customer loyalty, the relationship with customer and clinical inclination, each shown against other accounts.", "Customer Deep Dive"),
    ("Historical charts", "How price and shipments have moved across seasons, by brand.", "Customer Deep Dive"),
    ("Contracting levers", "Test price, share of wallet, partnership advocacy, engagement, negotiation stance and clinical focus.", "Simulation"),
    ("Predicted outcome and confidence", "The direction of the prediction for each product and how strongly it is supported.", "Simulation"),
    ("Driver explanation", "Which factors push the predicted chance of winning up or down.", "Simulation"),
    ("Price-response curve", "How the predicted chance of winning moves at ±5, 10, 15 and 20% around the current price.", "Simulation → SHAP waterfall — driver-level detail"),
    ("Narrative Summary", "The main findings in everyday language.", "Simulation"),
    ("Reset and download", "Restore the starting inputs, or download the scenario report and the account-signals report.", "Simulation and Customer Deep Dive"),
]

# Plain-language wording (settled on version 1, 11/13): what each label means
# for the account. BAND_RULES keeps the precise rule as a secondary line.
BANDS: list[tuple[str, str, str]] = [
    ("Comfortable Win", "comfortable", "The model predicts a win for this account under the current scenario."),
    ("Watch", "watch", "There is a risk of losing this account. Changes to price or other factors may improve our chances."),
    ("At Risk", "risk", "The model predicts a low chance of winning this account under the current scenario."),
]
BAND_RULES: dict[str, str] = {
    "Comfortable Win": "At or above the product's winning cutoff.",
    "Watch": "Below the cutoff, but by no more than 20 points.",
    "At Risk": "More than 20 points below the cutoff.",
}

CONFIDENCE_LEVELS: list[tuple[str, str]] = [
    ("High", "The model has strong support for this prediction."),
    ("Moderate", "The prediction is less certain. Consider other account factors before deciding."),
    ("Low", "The prediction is uncertain. Review the account details before making decisions."),
]

WIN_VS_CONFIDENCE: tuple[str, str] = (
    "Win probability is not confidence.",
    "Win probability tells you how likely a win is; confidence tells you how much to rely on that. "
    "A 70% win probability does not mean 70% confidence.",
)

GLOSSARY: list[tuple[str, str]] = [
    ("Win probability", "How likely we are to win this account under the scenario you set. It is not a guarantee."),
    ("Winning cutoff", "The line each product must reach for a predicted win. Each product has its own, and it is not always 50%."),
    ("Outcome band", "Comfortable Win, Watch or At Risk: a quick read of where the win probability sits against the cutoff."),
    ("Confidence", "How much we can rely on the prediction: High, Moderate or Low."),
    ("Driver contribution", "How much a factor moves this prediction. Blue bars improve the chances of winning; red bars reduce them. Longer bars matter more."),
    ("Price sensitivity", "How much the chances of winning change when price moves. High sensitivity means price matters more for this account."),
]

CONFIDENCE_MADE_OF: list[tuple[str, str, str]] = [
    ("Margin", "50%", "How far the result is from the cutoff. Close calls score lower."),
    ("Stability", "25%", "Whether five versions of the model agree on this account."),
    ("Reliability", "25%", "How accurate this product's model has been historically."),
]

# Kept to the one distinction the user wants stated (11/13).
MYTHS: list[tuple[str, str]] = [WIN_VS_CONFIDENCE]

CAUTIONS: list[str] = [
    "Select Run Simulation after changing inputs. The result does not update on its own.",
    "Customer Deep Dive shows recorded account data. Simulation changes do not alter it.",
    "\"No Data\" means the information is missing, not that the value is low or unfavourable.",
    "Drivers explain the model's prediction. They do not promise that changing a factor will cause the same business outcome.",
    "The price-response curve shows predicted win chance at different prices. It does not identify the most profitable price.",
]

# FAQ used by the expandable guides: (question, answer markdown)
FAQ: list[tuple[str, str]] = [
    ("What can I learn about my customer?",
     "Open **Customer Deep Dive**. The overview cards give size, price, share and the strategic quadrant. "
     "Account signals show pricing, loyalty, relationship and clinical inclination, each placed against other accounts. "
     "The charts show price and shipments by season and brand."),
    ("What can I change?",
     "In **Simulation**, the Contracting Levers table has one column per product. Dead net price is the main offer lever; "
     "the others (share of wallet, partnership advocacy, engagement, negotiation stance, clinical focus) are account assumptions. "
     "Changed values are highlighted. Change one thing at a time to see its effect."),
    ("How do I read the result?",
     "Each product gets an outcome band (Comfortable Win, Watch or At Risk) against its own cutoff, and a confidence level "
     "(High, Moderate or Low). Read the two together."),
    ("How do I see why?",
     "The **contribution chart** shows which factors pushed the result up (blue) or down (red). "
     "Open **SHAP waterfall — driver-level detail** for the full waterfall and the price-response curve. "
     "The **Narrative Summary** puts the main points into words."),
    ("How do I try another scenario?",
     "Change the levers and select **Run Simulation** again. **Reset Inputs** returns every lever to the account's recorded values."),
    ("How do I take the findings away?",
     "The download button beside Contracting Levers saves the scenario report as a PDF. "
     "Customer Deep Dive has its own download for the account-signals report."),
]


def open_tool_button(label: str, key: str, icon: str = ":material/arrow_forward:") -> None:
    """Primary call to action that leaves the landing page."""
    from landing import go_to  # local: landing/__init__ imports this module

    st.button(label, key=key, type="primary", icon=icon, on_click=go_to, args=(False,))


def band_pill(label: str, kind: str) -> str:
    return (
        f"<span class='lp-band lp-band-{kind}'><span class='lp-band-dot'></span>"
        f"{escape(label).upper()}</span>"
    )


def confidence_pill(level: str) -> str:
    return f"<span class='lp-conf lp-conf-{level.lower()}'>{escape(level)}</span>"


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _example_bars() -> str:
    # Mock driver contributions for the illustrative example (not real data).
    rows = [
        ("Price vs benchmark", "pos", 38),
        ("Prior-year wallet share", "pos", 26),
        ("Negotiation stance", "neg", 22),
        ("Competitor price gap", "neg", 14),
    ]
    out = "".join(
        "<div class='lp-bar-row'>"
        f"<span>{escape(label)}</span>"
        f"<div class='lp-bar-track'><div class='lp-bar lp-bar-{kind}' style='width:{width}%;'></div></div>"
        "</div>"
        for label, kind, width in rows
    )
    return f"<div class='lp-bars'>{out}</div>"


def _example_price_curve(cutoff: float) -> str:
    # Mock curve: win chance falls as price rises; dashed line is the cutoff.
    xs = [20, 60, 100, 140, 180, 220, 260, 300, 340]
    ys = [0.74, 0.71, 0.67, 0.63, 0.58, 0.53, 0.47, 0.42, 0.37]

    def y(p: float) -> float:
        return 104 - p * 100

    points = " ".join(f"{x},{y(p):.1f}" for x, p in zip(xs, ys))
    dots = "".join(
        f"<circle cx='{x}' cy='{y(p):.1f}' r='{4.5 if i == 4 else 3}' fill='{'#EC7200' if i == 4 else '#CF7D4A'}'/>"
        for i, (x, p) in enumerate(zip(xs, ys))
    )
    cy = y(cutoff)
    return (
        "<svg viewBox='0 0 360 120' width='100%' height='120' role='img' aria-label='Example price-response curve'>"
        f"<line x1='10' x2='350' y1='{cy:.1f}' y2='{cy:.1f}' stroke='#DC2626' stroke-dasharray='5 4' stroke-width='1.4'/>"
        f"<text x='350' y='{cy - 5:.1f}' text-anchor='end' font-size='10' fill='#DC2626'>cutoff {cutoff:.0%}</text>"
        f"<polyline points='{points}' fill='none' stroke='#EC7200' stroke-width='2.4'/>"
        f"{dots}"
        "<text x='20' y='118' font-size='10' fill='#716964'>−20%</text>"
        "<text x='180' y='118' font-size='10' fill='#716964' text-anchor='middle'>current price</text>"
        "<text x='340' y='118' font-size='10' fill='#716964' text-anchor='end'>+20%</text>"
        "</svg>"
    )


def why_example_html(metadata: dict[str, Any] | None) -> str:
    """Ribbon + mock driver bars + mock price curve, with their one-line legends."""
    cutoff = product_cutoffs(metadata).get("Viresta", _FALLBACK_CUTOFFS["Viresta"])
    return (
        "<span class='lp-ribbon'>Illustrative example — not real customer data</span>"
        "<div class='lp-why-split'>"
        f"<div>{_example_bars()}<div class='lp-why-legend'>Blue supports the predicted chance of winning, "
        "red reduces it, longer bars matter more.</div></div>"
        f"<div>{_example_price_curve(cutoff)}<div class='lp-why-legend'>The price-response curve shows the "
        "predicted win chance at ±5, 10, 15 and 20% around the current price.</div></div>"
        "</div>"
    )


def why_section_html(metadata: dict[str, Any] | None, title: str = "Understand why") -> str:
    """Drivers section: text on the left, illustrative drivers + price curve on the right."""
    return (
        "<div class='lp-why'>"
        f"<div class='lp-why-text'><div class='lp-h2' style='font-size:1.3rem;'>{escape(title)}</div>"
        "<p>The contribution chart shows which factors pushed the prediction up or down, and the "
        "price curve shows how the win chance moves with price.</p></div>"
        f"<div class='lp-card'>{why_example_html(metadata)}</div></div>"
    )


def inject_base_css() -> None:
    html(
        """
        <style>
        .lp-band {display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem;
            border-radius:999px; font-size:.74rem; font-weight:800; letter-spacing:.03rem; white-space:nowrap;}
        .lp-band-dot {width:.5rem; height:.5rem; border-radius:50%; background:currentColor;}
        .lp-band-comfortable {background:#E6F6EC; color:#15803D;}
        .lp-band-watch {background:#FFF1E5; color:#EA580C;}
        .lp-band-risk {background:#FDE8E8; color:#DC2626;}
        .lp-conf {display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.74rem; font-weight:800;}
        .lp-conf-high {background:var(--light-teal); color:var(--teal);}
        .lp-conf-moderate {background:var(--light-orange); color:var(--orange);}
        .lp-conf-low {background:#FDECEC; color:var(--red);}
        .lp-muted {color:var(--text-secondary);}
        .lp-card {background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.1rem;}
        .lp-h1 {font-size:1.9rem; font-weight:800; color:var(--text-primary); line-height:1.2; margin:0 0 .4rem;}
        .lp-h2 {font-size:1.15rem; font-weight:800; color:var(--text-primary); margin:0 0 .35rem;}
        .lp-h3 {font-size:.95rem; font-weight:800; color:var(--text-primary); margin:0 0 .25rem;}
        .lp-lead {font-size:1.02rem; color:var(--text-secondary); max-width:780px; line-height:1.5;}
        .lp-eyebrow {font-size:.72rem; font-weight:900; letter-spacing:.06rem; text-transform:uppercase; color:var(--purple);}
        .lp-note {font-size:.78rem; color:var(--text-secondary);}
        .lp-why {display:grid; grid-template-columns:minmax(0,.85fr) minmax(0,1.15fr); gap:1rem;
            align-items:start; margin-top:1.6rem;}
        .lp-why-text p {font-size:.95rem; color:var(--text-secondary); line-height:1.55; margin:.3rem 0 0;}
        .lp-ribbon {display:inline-block; font-size:.66rem; font-weight:800; letter-spacing:.03rem;
            text-transform:uppercase; color:#8A5A00; background:#FFF6DB; border:1px dashed #E2B53E;
            border-radius:6px; padding:.12rem .5rem; margin-bottom:.65rem;}
        .lp-why-split {display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr); gap:1rem; align-items:center;}
        .lp-bars {display:flex; flex-direction:column; gap:.4rem;}
        .lp-bar-row {display:grid; grid-template-columns:150px minmax(0,1fr); gap:.6rem; align-items:center; font-size:.78rem;}
        .lp-bar-track {position:relative; height:14px;}
        .lp-bar-track::before {content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background:#CEC5C1;}
        .lp-bar {position:absolute; top:0; height:14px; border-radius:3px;}
        .lp-bar-pos {left:50%; background:#1E88E5;}
        .lp-bar-neg {right:50%; background:#FF0D57;}
        .lp-why-legend {font-size:.78rem; color:var(--text-secondary); margin-top:.6rem; line-height:1.45;}
        @media (max-width: 1100px) {.lp-why {grid-template-columns:minmax(0,1fr);}}
        @media (max-width: 760px) {.lp-why-split {grid-template-columns:minmax(0,1fr);}}
        </style>
        """
    )
