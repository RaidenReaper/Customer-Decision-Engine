from __future__ import annotations

import hashlib
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from src.scoring import score_customer_response as external_score_customer_response
except ImportError:
    external_score_customer_response = None

from src.utils import (
    DATA_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    calculate_derived_fields,
    load_contracting_data,
    read_json,
)


BRAND_FAMILY = {
    "Flublok": "Influenza",
    "Fluzone HD": "Influenza",
    "Fluzone SD": "Influenza",
}

DRIVER_CATEGORIES = {
    "commercial_offer": "Offer Dynamics",
    "customer_relationship": "Customer Context",
    "competitive_pressure": "Competitive Pressure",
    "deal_context": "Negotiation Context",
}

OUTCOME_LABELS = ["Accepted", "Negotiated", "Rejected"]


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file is missing. Please run python src/generate_dummy_data.py and python src/train_model.py first."
        )
    return joblib.load(model_path)


def load_feature_metadata(path: Path = FEATURE_COLUMNS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Feature metadata is missing. Please run python src/train_model.py first.")
    return read_json(path)


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def local_driver(
    name: str,
    direction: str,
    impact_pp: int,
    explanation: str,
    category: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction,
        "impact_pp": impact_pp,
        "explanation": explanation,
        "category": category,
    }


def local_calculate_driver_influence(user_inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    inputs = {**user_inputs, **calculate_derived_fields(user_inputs)}
    drivers: dict[str, list[dict[str, Any]]] = {
        "commercial_offer": [],
        "customer_relationship": [],
        "competitive_pressure": [],
        "deal_context": [],
        "watchouts": [],
    }

    if inputs["rebate_pct"] >= inputs["prior_rebate_pct"] + 3:
        drivers["commercial_offer"].append(
            local_driver("Rebate support", "Positive", 5, "Proposed rebate is above the prior contract level.", "commercial_offer")
        )
    elif inputs["rebate_pct"] < inputs["prior_rebate_pct"]:
        drivers["commercial_offer"].append(
            local_driver("Rebate support", "Negative", -4, "Proposed rebate is below the prior contract level.", "commercial_offer")
        )

    if inputs["price_gap_vs_prior"] > 25:
        drivers["commercial_offer"].append(
            local_driver("Net price movement", "Negative", -6, "Net price is materially above prior year.", "commercial_offer")
        )
        drivers["watchouts"].append(
            local_driver("Price increase", "Watchout", 0, "Net price is more than 25% above prior net price.", "commercial_offer")
        )
    elif inputs["price_gap_vs_prior"] > 4:
        drivers["commercial_offer"].append(
            local_driver("Net price movement", "Negative", -3, "Net price is above prior year.", "commercial_offer")
        )
    elif inputs["price_gap_vs_prior"] < -3:
        drivers["commercial_offer"].append(
            local_driver("Net price movement", "Positive", 4, "Net price is more favorable than prior year.", "commercial_offer")
        )

    if inputs["reimbursement_rate_pct"] >= 86:
        drivers["commercial_offer"].append(
            local_driver("Reimbursement support", "Positive", 5, "Reimbursement rate is favorable for access and pull-through.", "commercial_offer")
        )
    elif inputs["reimbursement_rate_pct"] < 60:
        drivers["commercial_offer"].append(
            local_driver("Reimbursement support", "Negative", -7, "Reimbursement rate falls materially below the level typically required for pull-through.", "commercial_offer")
        )
        drivers["watchouts"].append(
            local_driver("Low reimbursement", "Watchout", 0, "Reimbursement below 60% may create customer pushback.", "commercial_offer")
        )

    if inputs["volume_gap_vs_prior"] > 50:
        drivers["watchouts"].append(
            local_driver("Volume validation", "Watchout", 0, "Volume commitment exceeds 150% of prior volume.", "deal_context")
        )
    elif inputs["volume_gap_vs_prior"] >= 5:
        drivers["commercial_offer"].append(
            local_driver("Volume commitment", "Positive", 3, "Volume commitment is higher than prior volume.", "commercial_offer")
        )

    if inputs["market_share_commitment_pct"] > 80:
        drivers["deal_context"].append(
            local_driver("Aggressive share ask", "Negative", -4, "Market share commitment is above 80%.", "deal_context")
        )
        drivers["watchouts"].append(
            local_driver("Aggressive share target", "Watchout", 0, "Market share commitment is above 80%.", "deal_context")
        )
    elif inputs["market_share_gap_vs_prior"] >= 4:
        drivers["commercial_offer"].append(
            local_driver("Market share commitment", "Positive", 3, "Market share commitment improves versus prior year.", "commercial_offer")
        )

    if inputs["historical_win_rate"] >= 0.70:
        drivers["customer_relationship"].append(
            local_driver("Historical relationship", "Positive", 5, "Customer has a strong historical win rate.", "customer_relationship")
        )
    elif inputs["historical_win_rate"] < 0.45:
        drivers["customer_relationship"].append(
            local_driver("Historical relationship", "Negative", -5, "Customer has a weaker historical acceptance pattern.", "customer_relationship")
        )

    if inputs["prior_year_attainment_pct"] >= 95:
        drivers["customer_relationship"].append(
            local_driver("Prior attainment", "Positive", 4, "Customer has strong prior attainment against contracted goals.", "customer_relationship")
        )
    elif inputs["prior_year_attainment_pct"] < 75:
        drivers["customer_relationship"].append(
            local_driver("Prior attainment", "Negative", -4, "Weak prior attainment lowers confidence in proposed commitments.", "customer_relationship")
        )

    if inputs["competitive_pressure_score"] >= 70:
        drivers["competitive_pressure"].append(
            local_driver("Competitive pressure", "Negative", -6, "Competitive pressure is high in this account.", "competitive_pressure")
        )
    elif inputs["competitive_pressure_score"] <= 35:
        drivers["competitive_pressure"].append(
            local_driver("Competitive position", "Positive", 3, "Competitive pressure is relatively low.", "competitive_pressure")
        )
    else:
        drivers["competitive_pressure"].append(
            local_driver("Competitive position", "Neutral", 0, "Competitive pressure is moderate.", "competitive_pressure")
        )

    if inputs["returnability_pct"] < inputs["prior_returnability_pct"] - 2:
        drivers["deal_context"].append(
            local_driver("Returnability", "Negative", -3, "Returnability is below customer expectation based on prior terms.", "deal_context")
        )
    if inputs["cancellation_pct"] < inputs["prior_cancellation_pct"] - 2:
        drivers["deal_context"].append(
            local_driver("Cancellation allowance", "Negative", -3, "Cancellation allowance is lower than prior contract.", "deal_context")
        )
    if inputs["deal_attractiveness_score"] >= 68:
        drivers["deal_context"].append(
            local_driver("Deal attractiveness", "Positive", 5, "Overall economics are attractive relative to historical behavior.", "deal_context")
        )
    elif inputs["deal_attractiveness_score"] < 45:
        drivers["deal_context"].append(
            local_driver("Deal attractiveness", "Negative", -5, "Overall economics fall below the range this customer has historically accepted.", "deal_context")
        )

    defaults = {
        "commercial_offer": local_driver("Offer balance", "Neutral", 0, "Commercial terms are broadly in line with the baseline.", "commercial_offer"),
        "customer_relationship": local_driver("Relationship baseline", "Neutral", 0, "No dominant relationship signal was detected.", "customer_relationship"),
        "competitive_pressure": local_driver("Competitive position", "Neutral", 0, "Competitive pressure is within the expected range.", "competitive_pressure"),
        "deal_context": local_driver("Deal context", "Neutral", 0, "Deal context does not show a major directional signal.", "deal_context"),
    }
    for category, default in defaults.items():
        if not drivers[category]:
            drivers[category].append(default)
    if not drivers["watchouts"]:
        drivers["watchouts"].append(
            local_driver("Monitor assumptions", "Watchout", 0, "Validate assumptions before using this scenario outside the demo.", "deal_context")
        )
    return drivers


def local_score_customer_response(
    model,
    user_inputs: dict[str, Any],
    feature_metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {**user_inputs, **calculate_derived_fields(user_inputs)}
    feature_columns = feature_metadata["feature_columns"]
    input_df = pd.DataFrame([{column: row[column] for column in feature_columns}])

    probability_values = model.predict_proba(input_df)[0]
    probabilities = {label: float(probability_values[index]) for index, label in enumerate(OUTCOME_LABELS)}
    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    recommended_outcome = OUTCOME_LABELS[int(probability_values.argmax())]
    top_two_gap = (sorted_probabilities[0] - sorted_probabilities[1]) * 100
    confidence_score = probabilities[recommended_outcome] * 100
    if top_two_gap < 10:
        confidence_score = max(confidence_score - 10, 0)
    confidence_score = round(min(confidence_score, 100), 1)
    confidence_band = "High" if confidence_score >= 75 else "Moderate" if confidence_score >= 55 else "Low"

    drivers = local_calculate_driver_influence(user_inputs)
    positive_driver = next(
        (driver for category in DRIVER_CATEGORIES for driver in drivers[category] if driver["direction"] == "Positive"),
        drivers["commercial_offer"][0],
    )
    negative_driver = next(
        (driver for category in DRIVER_CATEGORIES for driver in drivers[category] if driver["direction"] == "Negative"),
        drivers["competitive_pressure"][0],
    )
    watchout = drivers["watchouts"][0]
    narrative = (
        f"The scenario is currently recommended as {recommended_outcome} with a "
        f"{confidence_band.lower()} confidence score of {confidence_score:.1f}%. "
        f"The strongest supportive signal is {positive_driver['name'].lower()}, while "
        f"{negative_driver['name'].lower()} is the main pressure point to manage. "
        f"The team should watch {watchout['name'].lower()} before advancing the offer."
    )

    return {
        "probabilities": probabilities,
        "predicted_outcome": recommended_outcome,
        "recommended_outcome": recommended_outcome,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "top_two_gap_pp": round(top_two_gap, 1),
        "drivers": drivers,
        "watchouts": drivers["watchouts"],
        "narrative_summary": narrative,
    }


def score_customer_response(model, user_inputs: dict[str, Any], feature_metadata: dict[str, Any]) -> dict[str, Any]:
    if external_score_customer_response is not None:
        return external_score_customer_response(model, user_inputs, feature_metadata)
    return local_score_customer_response(model, user_inputs, feature_metadata)

LEVER_CONFIGS = [
    {
        "label": "Net Price",
        "key": "net_price",
        "current_column": "prior_net_price",
        "value_type": "currency",
        "helper": "$/dose",
        "min_value": 0.0,
        "max_value": 300.0,
        "step": 0.1,
        "format": "%.2f",
    },
    {
        "label": "Rebate %",
        "key": "rebate_pct",
        "current_column": "prior_rebate_pct",
        "value_type": "percent",
        "helper": "Contract rebate",
        "min_value": 0.0,
        "max_value": 50.0,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "label": "Multi-year Discount %",
        "key": "multi_year_discount_pct",
        "current_column": None,
        "value_type": "percent",
        "helper": "Incremental term discount",
        "min_value": 0.0,
        "max_value": 25.0,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "label": "Cancellation %",
        "key": "cancellation_pct",
        "current_column": "prior_cancellation_pct",
        "value_type": "percent",
        "helper": "Allowed cancellation rate",
        "min_value": 0.0,
        "max_value": 25.0,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "label": "Volume Commitment",
        "key": "volume_commitment",
        "current_column": "prior_volume",
        "value_type": "volume",
        "helper": "Total committed doses",
        "min_value": 0,
        "max_value": 2_000_000,
        "step": 5_000,
        "format": None,
    },
    {
        "label": "Market Share Commitment %",
        "key": "market_share_commitment_pct",
        "current_column": "prior_year_market_share",
        "value_type": "percent",
        "helper": "% of flu portfolio",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 0.5,
        "format": "%.1f",
    },
    {
        "label": "Returnability %",
        "key": "returnability_pct",
        "current_column": "prior_returnability_pct",
        "value_type": "percent",
        "helper": "Allowed return rate",
        "min_value": 0.0,
        "max_value": 30.0,
        "step": 0.1,
        "format": "%.1f",
    },
]

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F8F7FB;
            --primary: #7432A8;
            --purple: #7432A8;
            --orange: #F37000;
            --teal: #36A39B;
            --blue: #4285F4;
            --light-purple: #F1E8FA;
            --light-orange: #FFF3E8;
            --light-teal: #EAF8F6;
            --light-blue: #EAF2FF;
            --border: #E5E2EF;
            --text-primary: #1F1F29;
            --text-secondary: #6B6878;
            --green: #36A39B;
            --amber: #F37000;
            --red: #DC2626;
            --card: #FFFFFF;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background: var(--bg) !important; color: var(--text-primary) !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1540px;}
        .app-header {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: 1.1rem 1.25rem; border: 1px solid #DED7EF; border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.98) 0%, rgba(241,232,250,.92) 100%),
                linear-gradient(90deg, #F37000 0%, #36A39B 34%, #4285F4 68%, #7432A8 100%);
            box-shadow: 0 12px 30px rgba(31, 31, 41, .07); margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .app-header:before {
            content: "";
            position: absolute;
            left: 0; top: 0; right: 0; height: 5px;
            background: linear-gradient(90deg, #F37000 0%, #36A39B 34%, #4285F4 68%, #7432A8 100%);
        }
        .app-title {font-size: 1.75rem; line-height: 1.1; font-weight: 820; margin: 0; color: var(--text-primary);}
        .app-subtitle {font-size: .94rem; color: var(--text-secondary); margin-top: .38rem;}
        .header-badges {display: flex; gap: .45rem; flex-wrap: wrap; justify-content: flex-end;}
        .badge {
            display: inline-flex; align-items: center; border-radius: 999px; padding: .34rem .64rem;
            background: var(--light-purple); color: var(--primary); border: 1px solid #DCCEFF;
            font-size: .76rem; font-weight: 760; white-space: nowrap;
        }
        .badge:nth-child(1) {
            background: var(--light-teal);
            color: #19766F;
            border-color: #BDE4E0;
        }
        .badge:nth-child(2) {
            background: var(--light-purple);
            color: var(--purple);
            border-color: #DCCEFF;
        }
        .section-card {
            background: var(--card); border: 1px solid var(--border); border-radius: 14px;
            padding: .95rem; box-shadow: 0 1px 2px rgba(31, 31, 41, .04); margin-bottom: .85rem;
        }
        .section-title {font-size: 1.02rem; font-weight: 800; color: var(--text-primary); margin: 0 0 .16rem 0;}
        .section-subtitle {font-size: .82rem; color: var(--text-secondary); margin: 0 0 .75rem 0;}
        .small-label {
            font-size: .72rem; font-weight: 760; color: var(--text-secondary);
            text-transform: uppercase; letter-spacing: .03rem; margin-bottom: .3rem;
        }
        .metric-card {
            border: 1px solid var(--border); border-radius: 14px; padding: .86rem .92rem;
            background: #FFFFFF; height: 140px; box-shadow: 0 1px 2px rgba(31,31,41,.04);
            display: flex; flex-direction: column;
        }
        .metric-card-green {border-left: 5px solid var(--teal); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-teal) 100%);}
        .metric-card-amber {border-left: 5px solid var(--orange); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-orange) 100%);}
        .metric-card-red {border-left: 5px solid var(--red); background: linear-gradient(180deg, #FFFFFF 0%, #FFF1F2 100%);}
        .metric-card-blue {border-left: 5px solid var(--blue); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-blue) 100%);}
        .metric-label {font-size: .76rem; color: var(--text-secondary); font-weight: 760; text-transform: uppercase;}
        .metric-value {font-size: 1.65rem; color: var(--text-primary); font-weight: 820; margin-top: .24rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
        .metric-value-compact {font-size: 1.15rem; line-height: 1.25;}
        .metric-help {font-size: .78rem; color: var(--text-secondary); margin-top: .22rem;}
        .current-card {
            background: #FFFFFF; border: 1px solid #D8D3E6; border-radius: 11px;
            padding: 0 .75rem; min-height: 42px;
            display: flex; align-items: center;
            font-size: 1rem; font-weight: 780; color: var(--text-primary);
            box-shadow: none;
        }
        .current-value {font-size: 1rem; font-weight: 780; color: var(--text-primary);}
        .stacked-bar {
            display: flex; width: 100%; height: 28px; border-radius: 999px; overflow: hidden;
            background: #EDEAF5; border: 1px solid var(--border);
        }
        .legend {display: flex; gap: .9rem; flex-wrap: wrap; margin-top: .55rem; color: var(--text-secondary); font-size: .8rem;}
        .legend-dot {display:inline-block; width:.62rem; height:.62rem; border-radius:999px; margin-right:.28rem;}
        .driver-card {
            background: #FFFFFF; border: 1px solid var(--border); border-radius: 14px;
            padding: .9rem; min-height: 230px; margin-bottom: .8rem;
        }
        .driver-header {display:flex; align-items:center; justify-content:space-between; gap:.5rem; margin-bottom:.5rem;}
        .driver-title {font-size: .95rem; font-weight: 790; color: var(--text-primary);}
        .driver-meta {font-size: .78rem; color: var(--text-secondary);}
        .driver-list {margin: .4rem 0 0 0; padding-left: 0; list-style: none;}
        .driver-list li {margin-bottom: .5rem; color: var(--text-primary); font-size: .84rem; line-height: 1.35;}
        .positive-chip, .negative-chip, .watchout-chip, .neutral-chip {
            display: inline-flex; border-radius: 999px; padding: .12rem .45rem; font-size: .7rem;
            font-weight: 760; margin-right: .35rem;
        }
        .positive-chip {background: var(--light-teal); color: #19766F;}
        .negative-chip {background: #FEE2E2; color: var(--red);}
        .watchout-chip {background: var(--light-orange); color: #B44E00;}
        .neutral-chip {background: var(--light-blue); color: #2564C9;}
        .watchout-card {
            border: 1px solid #FFC58F; background: var(--light-orange); border-radius: 12px;
            padding: .72rem .82rem; color: #8A3D00; font-size: .84rem; margin-top: .65rem;
        }
        .narrative-card {
            border: 1px solid var(--border); border-radius: 14px; padding: .95rem 1rem;
            background: #FFFFFF; color: var(--text-primary); line-height: 1.55;
        }
        .lever-row {
            border: 1px solid var(--border); border-radius: 14px; padding: .48rem;
            background: #FFFFFF; margin-bottom: .52rem;
        }
        .lever-name {
            font-size: .98rem;
            font-weight: 820;
            color: var(--text-primary);
            margin: .48rem 0 .42rem 0;
        }
        .lever-value-label {
            font-size: .72rem;
            font-weight: 800;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: .035rem;
            margin-bottom: .28rem;
        }
        .lever-input-wrap {padding-top: .05rem;}

        .lever-table-header {
            display: flex; padding: .3rem .1rem; margin-bottom: .2rem;
            border-bottom: 2px solid var(--border);
        }
        .lever-table-header-cell {
            font-size: .78rem; font-weight: 800; color: var(--text-primary);
            text-transform: none;
        }
        div[class*="st-key-lever-row-even-"] {
            background: #FAF9FC; border-radius: 12px; padding: .45rem .5rem .3rem .5rem; margin-bottom: .25rem;
        }
        div[class*="st-key-lever-row-odd-"] {
            background: #FFFFFF; border-radius: 12px; padding: .45rem .5rem .3rem .5rem; margin-bottom: .25rem;
        }
        .lever-table-metric {
            font-size: .9rem; font-weight: 800; color: var(--text-primary);
            padding-top: .5rem;
        }
        .lever-table-season-label {
            font-size: .74rem; font-weight: 700; color: var(--text-secondary);
            padding-top: .55rem;
        }
        .lever-table-season-label--prev { font-style: italic; font-weight: 600; color: #9C97A8; }
        .lever-table-prev-value {
            font-size: .88rem; font-style: italic; color: #8D879C;
            padding-top: .5rem;
        }

        /* Keep the Pricing Levers panel pinned in view while the Outcome column scrolls.
           Streamlit wraps keyed containers in an internal stLayoutWrapper div that only
           sizes to its child's natural height, leaving no room for sticky to travel within
           the taller (stretched) column — force that wrapper to fill the column's height. */
        div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-lever-sticky-panel"]) {
            height: 100% !important;
        }
        /* The panel itself inherits Streamlit's default flex: 1 1 0% (stretch-to-fill),
           which overrides height/auto. Pin it back to its natural content size so there's
           slack in the (now taller) wrapper above for the sticky offset to travel within. */
        div[class*="st-key-lever-sticky-panel"] {
            position: sticky;
            top: 1rem;
            flex: 0 0 auto !important;
        }

        /* Force Streamlit/BaseWeb controls onto the light product theme. */
        label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important; font-weight: 700 !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stMultiSelect"] div[role="combobox"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="input"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D8D3E6 !important;
            border-radius: 11px !important;
            box-shadow: none !important;
            min-height: 42px !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] *,
        [data-testid="stMultiSelect"] [data-baseweb="select"] *,
        [data-testid="stSelectbox"] div[role="combobox"] *,
        [data-testid="stMultiSelect"] div[role="combobox"] *,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
            caret-color: var(--primary) !important;
        }
        input, textarea {
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }
        input::placeholder, textarea::placeholder {
            color: #8D879C !important;
            -webkit-text-fill-color: #8D879C !important;
        }
        [data-testid="stNumberInput"] input {
            background-color: #FFFFFF !important;
            font-weight: 760 !important;
            min-height: 42px !important;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button[aria-label],
        [data-testid="stNumberInput"] [role="button"] {
            background-color: var(--light-purple) !important;
            color: var(--purple) !important;
            border-color: #D8D3E6 !important;
        }
        [data-testid="stNumberInput"] button svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {
            fill: var(--primary) !important;
            color: var(--primary) !important;
        }
        [data-baseweb="tag"] {
            background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"] span, [data-baseweb="tag"] svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
            background: #FFFFFF !important;
            color: var(--text-primary) !important;
        }
        [role="option"] {
            background: #FFFFFF !important;
            color: var(--text-primary) !important;
        }
        [role="option"]:hover {
            background: #F4F1FA !important;
        }
        div[data-testid="stButton"] > button {
            border-radius: 11px !important;
            font-weight: 780 !important;
            min-height: 42px !important;
            border: 1px solid #D8D3E6 !important;
            background: #FFFFFF !important;
            color: var(--purple) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #4285F4, #7432A8) !important;
            color: #FFFFFF !important;
            border-color: #7432A8 !important;
        }
        div[data-testid="stButton"] > button:disabled {
            opacity: .55 !important;
            background: #F3F1F8 !important;
            color: #8D879C !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border-color: var(--border) !important;
            box-shadow: 0 1px 2px rgba(31,31,41,.04);
        }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stable_match_score(values: dict[str, Any]) -> int:
    encoded = "|".join(str(values.get(key, "")) for key in ["customer_name", "contract_name", "brand_scope", "net_price"])
    digest = hashlib.md5(encoded.encode("utf-8")).hexdigest()
    return 70 + int(digest[:4], 16) % 21


def format_currency(value: float) -> str:
    return f"${float(value):,.2f}"


def format_percent(value: float) -> str:
    return f"{float(value):,.1f}%"


def format_volume(value: float) -> str:
    return f"{int(value):,} doses"


def format_value(value: float | int, value_type: str) -> str:
    if value_type == "currency":
        return format_currency(float(value))
    if value_type == "volume":
        return format_volume(float(value))
    return format_percent(float(value))


def slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def lever_state_key(brand: str, lever_key: str) -> str:
    return f"proposed_{slugify(brand)}_{lever_key}"


def safe_selectbox(container, label: str, options: list[Any], key: str) -> Any:
    if not options:
        st.error(f"No options are available for {label}.")
        st.stop()
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return container.selectbox(label, options, key=key)


def safe_multiselect(container, label: str, options: list[Any], default: list[Any], key: str) -> list[Any]:
    if not options:
        st.error(f"No options are available for {label}.")
        st.stop()
    valid_default = [value for value in default if value in options] or options[:1]
    current = st.session_state.get(key, valid_default)
    if not isinstance(current, list):
        current = valid_default
    selected = [value for value in current if value in options]
    st.session_state[key] = selected or valid_default
    return container.multiselect(label, options, key=key)


def default_for_lever(row: pd.Series, config: dict[str, Any]) -> float | int:
    if config["key"] == "multi_year_discount_pct":
        return float(row.get("multi_year_discount_pct", 0.0))
    value = row[config["current_column"]] if config["current_column"] else row[config["key"]]
    if config["key"] == "volume_commitment":
        return int(value)
    return float(value)


def show_setup_block(missing_paths: list[Path]) -> None:
    st.error("Please run python src/generate_dummy_data.py and python src/train_model.py first.")
    st.write("Missing artifacts:")
    for path in missing_paths:
        st.write(f"- {path}")
    st.code("python src/generate_dummy_data.py\npython src/train_model.py\nstreamlit run app.py", language="powershell")
    st.stop()


def initialize_product_values(product_rows: dict[str, pd.Series]) -> None:
    for brand, row in product_rows.items():
        for config in LEVER_CONFIGS:
            key = lever_state_key(brand, config["key"])
            if key not in st.session_state:
                st.session_state[key] = default_for_lever(row, config)


def reset_product_values(product_rows: dict[str, pd.Series]) -> None:
    for brand, row in product_rows.items():
        for config in LEVER_CONFIGS:
            st.session_state[lever_state_key(brand, config["key"])] = default_for_lever(row, config)


def reset_scenario(product_rows: dict[str, pd.Series]) -> None:
    reset_product_values(product_rows)
    st.session_state["latest_result"] = None


def render_header(model_version: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <p class="app-title">Customer Decision Engine</p>
                <div class="app-subtitle">Customer-level pricing simulation, sensitivity analysis, and scenario decisioning</div>
            </div>
            <div class="header-badges">
                <span class="badge">Synthetic PoC Model</span>
                <span class="badge">Model {escape(model_version)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, help_text: str, css_class: str, compact_value: bool = False) -> None:
    value_class = "metric-value metric-value-compact" if compact_value else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{escape(label)}</div>
            <div class="{value_class}">{escape(value)}</div>
            <div class="metric-help">{escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_scenario_inputs(row: pd.Series, selected_scope: str, selected_brands: list[str], brand: str) -> dict[str, Any]:
    inputs = row.to_dict()
    primary_brand = brand
    brand_scope = primary_brand if selected_scope == "Single Product" else ", ".join(selected_brands)
    proposed_values = {
        config["key"]: st.session_state[lever_state_key(primary_brand, config["key"])]
        for config in LEVER_CONFIGS
    }

    inputs.update(
        {
            "product_scope": selected_scope,
            "brand_scope": brand_scope,
            "brand_name": primary_brand,
            "product_name": primary_brand,
            "product_family": BRAND_FAMILY.get(primary_brand, str(row["product_family"])),
            **proposed_values,
        }
    )
    inputs.update(calculate_derived_fields(inputs))
    return inputs


def collect_scenario_watchouts(inputs: dict[str, Any]) -> list[str]:
    watchouts: list[str] = []
    if inputs["reimbursement_rate_pct"] < 60:
        watchouts.append("Reimbursement rate is below 60%.")
    if inputs["volume_commitment"] > inputs["prior_volume"] * 1.5:
        watchouts.append("Volume commitment is more than 150% of prior volume.")
    if inputs["net_price"] > inputs["prior_net_price"] * 1.25:
        watchouts.append("Proposed net price is more than 25% above prior net price.")
    if inputs["rebate_pct"] < inputs["prior_rebate_pct"] and inputs["net_price"] > inputs["prior_net_price"]:
        watchouts.append("Rebate is below prior rebate while net price is higher.")
    if inputs["market_share_commitment_pct"] > 80:
        watchouts.append("Market share commitment is above 80%.")
    return watchouts


def render_watchouts(watchouts: list[str]) -> None:
    if not watchouts:
        return
    items = "".join(f"<li>{escape(item)}</li>" for item in watchouts)
    st.markdown(
        f"""
        <div class="watchout-card">
            <strong>Scenario Watchouts</strong>
            <ul style="margin:.35rem 0 0 1.1rem; padding:0;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stacked_bar(probabilities: dict[str, float]) -> None:
    accepted = probabilities["Accepted"] * 100
    negotiated = probabilities["Negotiated"] * 100
    rejected = probabilities["Rejected"] * 100
    st.markdown(
        f"""
        <div class="stacked-bar">
            <div title="Accepted {accepted:.1f}%" style="width:{accepted:.2f}%; background:#36A39B;"></div>
            <div title="Negotiated {negotiated:.1f}%" style="width:{negotiated:.2f}%; background:#F37000;"></div>
            <div title="Rejected {rejected:.1f}%" style="width:{rejected:.2f}%; background:#DC2626;"></div>
        </div>
        <div class="legend">
            <span><span class="legend-dot" style="background:#36A39B;"></span>Accepted {accepted:.1f}%</span>
            <span><span class="legend-dot" style="background:#F37000;"></span>Negotiated {negotiated:.1f}%</span>
            <span><span class="legend-dot" style="background:#DC2626;"></span>Rejected {rejected:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def driver_direction_summary(drivers: list[dict[str, Any]]) -> tuple[str, str]:
    total = sum(driver.get("impact_pp", 0) for driver in drivers if driver.get("direction") != "Watchout")
    max_impact = max([abs(driver.get("impact_pp", 0)) for driver in drivers] + [0])
    if total > 2:
        direction = "Positive"
    elif total < -2:
        direction = "Negative"
    else:
        direction = "Mixed"
    if max_impact >= 5:
        impact = "High"
    elif max_impact >= 3:
        impact = "Medium"
    else:
        impact = "Low"
    return direction, impact


def impact_chip(driver: dict[str, Any]) -> str:
    direction = driver["direction"]
    if direction == "Positive":
        css = "positive-chip"
        text = f"Positive +{abs(driver['impact_pp'])}pp"
    elif direction == "Negative":
        css = "negative-chip"
        text = f"Negative -{abs(driver['impact_pp'])}pp"
    elif direction == "Watchout":
        css = "watchout-chip"
        text = "Watchout"
    else:
        css = "neutral-chip"
        text = "Neutral"
    return f"<span class='{css}'>{escape(text)}</span>"


def confidence_band(confidence_score: float) -> str:
    if confidence_score >= 75:
        return "High"
    if confidence_score >= 55:
        return "Moderate"
    return "Low"


def aggregate_product_results(product_results: list[dict[str, Any]], portfolio_inputs: dict[str, Any]) -> dict[str, Any]:
    probabilities = {
        outcome: sum(result["probabilities"][outcome] for result in product_results) / len(product_results)
        for outcome in ["Accepted", "Negotiated", "Rejected"]
    }
    recommended_outcome = max(probabilities, key=probabilities.get)
    confidence_score = round(probabilities[recommended_outcome] * 100, 1)
    primary_result = min(product_results, key=lambda result: result["confidence_score"])
    narrative = (
        f"The portfolio scenario is recommended as {recommended_outcome} with "
        f"{confidence_band(confidence_score).lower()} confidence across {len(product_results)} selected products. "
        f"The product requiring the closest review is {primary_result['product']} because it has the lowest product-level confidence. "
        f"{primary_result['narrative_summary']}"
    )
    return {
        "probabilities": probabilities,
        "predicted_outcome": recommended_outcome,
        "recommended_outcome": recommended_outcome,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band(confidence_score),
        "top_two_gap_pp": 0,
        "drivers": primary_result["drivers"],
        "watchouts": [watchout for result in product_results for watchout in result.get("watchouts", [])],
        "narrative_summary": narrative,
        "inputs": portfolio_inputs,
        "product_results": product_results,
    }


def score_selected_products(
    model,
    feature_metadata: dict[str, Any],
    product_scope: str,
    selected_brands: list[str],
    product_rows: dict[str, pd.Series],
) -> dict[str, Any]:
    product_results: list[dict[str, Any]] = []
    scenario_inputs_by_brand: dict[str, dict[str, Any]] = {}
    for brand in selected_brands:
        scenario_inputs = build_scenario_inputs(product_rows[brand], product_scope, selected_brands, brand)
        product_result = score_customer_response(model, scenario_inputs, feature_metadata)
        product_result["inputs"] = scenario_inputs
        product_result["product"] = brand
        product_results.append(product_result)
        scenario_inputs_by_brand[brand] = scenario_inputs

    if len(product_results) == 1:
        result = product_results[0]
        result["product_results"] = product_results
    else:
        first_inputs = next(iter(scenario_inputs_by_brand.values()))
        portfolio_inputs = {
            **first_inputs,
            "brand_name": "Portfolio",
            "product_name": "Portfolio",
            "brand_scope": ", ".join(selected_brands),
            "net_price": round(sum(inputs["net_price"] for inputs in scenario_inputs_by_brand.values()) / len(scenario_inputs_by_brand), 2),
            "rebate_pct": round(sum(inputs["rebate_pct"] for inputs in scenario_inputs_by_brand.values()) / len(scenario_inputs_by_brand), 2),
            "volume_commitment": sum(int(inputs["volume_commitment"]) for inputs in scenario_inputs_by_brand.values()),
        }
        result = aggregate_product_results(product_results, portfolio_inputs)

    result["confidence_factors"] = compute_confidence_factors(
        model,
        result.get("inputs", {}),
        result["probabilities"],
        feature_metadata,
    )
    result["overall_confidence_band"] = compute_overall_confidence_band(result["confidence_factors"])
    result["shap_contributions"] = compute_shap_contributions(model, result.get("inputs", {}), feature_metadata)
    return result


FEATURE_LABEL_MAP: dict[str, str] = {
    "net_price": "Net Price",
    "rebate_pct": "Rebate %",
    "multi_year_discount_pct": "Multi-year Discount %",
    "cancellation_pct": "Cancellation %",
    "reimbursement_rate_pct": "Reimbursement Rate %",
    "volume_commitment": "Volume Commitment",
    "market_share_commitment_pct": "Market Share Commitment %",
    "returnability_pct": "Returnability %",
    "prior_net_price": "Prior Net Price",
    "prior_rebate_pct": "Prior Rebate %",
    "prior_volume": "Prior Volume",
    "prior_year_market_share": "Prior-year Market Share",
    "prior_year_attainment_pct": "Prior-year Attainment %",
    "prior_returnability_pct": "Prior Returnability %",
    "prior_cancellation_pct": "Prior Cancellation %",
    "historical_win_rate": "Historical Win Rate",
    "competitive_pressure_score": "Competitive Pressure",
    "competitor_rebate_estimate_pct": "Competitor Rebate Estimate",
    "deal_attractiveness_score": "Deal Attractiveness Score",
    "revenue": "Revenue",
    "margin_proxy": "Margin Proxy",
    "rebate_dollars": "Rebate $",
    "price_gap_vs_prior": "Price Gap vs Prior",
    "volume_gap_vs_prior": "Volume Gap vs Prior",
    "market_share_gap_vs_prior": "Market Share Gap vs Prior",
    "reimbursement_gap": "Reimbursement Gap",
    "brand_name": "Brand",
    "customer_name": "Customer",
    "customer_segment": "Segment",
    "customer_tier": "Tier",
    "region": "Region",
    "product_family": "Product Family",
    "product_scope": "Product Scope",
    "season_id": "Season",
    "round_number": "Round",
    "brand_scope": "Brand Scope",
    "new_vs_existing_customer": "New vs Existing",
    "prior_contract_status": "Prior Contract Status",
    "prior_acceptance_flag": "Prior Acceptance",
    "strategic_account_flag": "Strategic Account",
}


def _prettify_shap_feature(raw: str) -> str:
    """Convert a sklearn ColumnTransformer feature name into a business label."""
    name = raw
    if name.startswith("numerical__"):
        column = name[len("numerical__"):]
        return FEATURE_LABEL_MAP.get(column, column.replace("_", " ").title())
    if name.startswith("categorical__"):
        rest = name[len("categorical__"):]
        for column in sorted(FEATURE_LABEL_MAP.keys(), key=len, reverse=True):
            prefix = f"{column}_"
            if rest.startswith(prefix):
                value = rest[len(prefix):]
                return f"{FEATURE_LABEL_MAP[column]} = {value}"
        underscore = rest.rfind("_")
        if underscore > 0:
            column = rest[:underscore]
            value = rest[underscore + 1:]
            column_label = FEATURE_LABEL_MAP.get(column, column.replace("_", " ").title())
            return f"{column_label} = {value}"
        return rest
    return name


def compute_shap_contributions(
    model,
    user_inputs: dict[str, Any],
    feature_metadata: dict[str, Any],
    top_n: int = 10,
) -> pd.DataFrame | None:
    """Return top-N SHAP contributions for the Accepted class as a DataFrame."""
    if not SHAP_AVAILABLE:
        return None
    try:
        import shap
    except ImportError:
        return None
    try:
        feature_columns = feature_metadata["feature_columns"]
        row = {**user_inputs, **calculate_derived_fields(user_inputs)}
        input_df = pd.DataFrame([{c: row[c] for c in feature_columns}])
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        input_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(classifier)
        shap_out = explainer(input_transformed)
        accept_shap = shap_out.values[0, :, OUTCOME_LABELS.index("Accepted")]
    except Exception:
        return None

    contributions = pd.DataFrame(
        {
            "feature": [_prettify_shap_feature(name) for name in feature_names],
            "shap": accept_shap.astype(float),
        }
    )
    contributions["abs_shap"] = contributions["shap"].abs()
    # Aggregate duplicate labels (in case categorical grouping collides)
    contributions = (
        contributions.groupby("feature", as_index=False)
        .agg(shap=("shap", "sum"), abs_shap=("abs_shap", "sum"))
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    contributions["direction"] = contributions["shap"].apply(lambda v: "Supports Accept" if v >= 0 else "Pushes toward Reject")
    return contributions


def render_shap_chart(contributions: pd.DataFrame | None) -> None:
    if contributions is None or contributions.empty:
        st.info("SHAP contributions are not available for this scenario.")
        return
    st.markdown(
        "<div class='small-label' style='margin-top:.8rem;'>Model feature contributions (SHAP) — impact on the Accepted probability</div>",
        unsafe_allow_html=True,
    )
    chart = (
        alt.Chart(contributions)
        .mark_bar(cornerRadiusEnd=4, stroke="#FFFFFF", strokeWidth=1)
        .encode(
            x=alt.X("shap:Q", title="SHAP value (log-odds contribution to Accepted)", axis=alt.Axis(format=".2f")),
            y=alt.Y(
                "feature:N",
                title=None,
                sort=alt.EncodingSortField(field="abs_shap", op="max", order="descending"),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Supports Accept", "Pushes toward Reject"],
                    range=["#19766F", "#DC2626"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("shap:Q", format="+.3f", title="SHAP value"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
        .properties(height=max(220, 28 * len(contributions)))
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        "<div style='color:var(--text-secondary); font-size:.78rem; margin-top:.2rem;'>"
        "SHAP values are log-odds contributions from the trained XGBoost model. "
        "Positive values push the prediction toward <strong>Accepted</strong>; negative values push toward <strong>Rejected</strong>. "
        "Bar length is proportional to the strength of the contribution."
        "</div>",
        unsafe_allow_html=True,
    )


def render_driver_section(drivers: dict[str, list[dict[str, Any]]]) -> None:
    st.markdown("<div class='section-title'>Driver & Influence</div>", unsafe_allow_html=True)
    rows = [st.columns(2, gap="medium"), st.columns(2, gap="medium")]
    categories = ["commercial_offer", "customer_relationship", "competitive_pressure", "deal_context"]
    for idx, category in enumerate(categories):
        column = rows[idx // 2][idx % 2]
        category_drivers = drivers.get(category, [])[:5]
        direction, impact = driver_direction_summary(category_drivers)
        bullets = "".join(
            f"<li>{impact_chip(driver)}{escape(driver['name'])}: <span style='color:#6B6878;'>{escape(driver['explanation'])}</span></li>"
            for driver in category_drivers
        )
        with column:
            st.markdown(
                f"""
                <div class="driver-card">
                    <div class="driver-header">
                        <div class="driver-title">{escape(DRIVER_CATEGORIES[category])}</div>
                        <div class="driver-meta">{escape(direction)} | {escape(impact)} impact</div>
                    </div>
                    <ul class="driver-list">{bullets}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_rationale(metadata: dict[str, Any], result: dict[str, Any] | None) -> None:
    with st.expander("View model rationale and PoC details", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Model Type", metadata.get("model_type", "XGBoost Classifier"))
        cols[1].metric("Training Sample", f"{metadata.get('training_sample_size', 500):,}")
        cols[2].metric("Back-test Accuracy", f"{metadata.get('accuracy', 0) * 100:.1f}%")
        cols[3].metric("Model Version", metadata.get("model_version", "v0.1"))
        cols[4].metric("Last Refreshed", metadata.get("last_refreshed", "Not available"))
        st.warning(metadata.get("disclaimer", "Synthetic PoC model for demonstration only."))
        top_features = metadata.get("top_features", [])
        if top_features:
            feature_df = pd.DataFrame(top_features[:10])
            st.dataframe(feature_df, width="stretch", hide_index=True)
        if result:
            st.caption(f"Similar deal match: {stable_match_score(result['inputs'])}%")


def render_saved_scenarios() -> None:
    with st.expander("Saved Scenarios", expanded=False):
        saved = st.session_state.get("saved_scenarios", [])
        if not saved:
            st.caption("No saved scenarios yet.")
            return
        st.dataframe(pd.DataFrame(saved), width="stretch", hide_index=True)


def _deep_dive_dummy_data() -> dict[str, pd.DataFrame]:
    """Static dummy data for the Customer Deep Dive screen.

    Scoped to the default customer; the same dataset displays regardless of which
    customer is picked in the Scenario Context filters (per spec).
    """
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    products = ["Flublok", "Fluzone HD", "Fluzone SD"]
    base_doses = {"Flublok": 95, "Fluzone HD": 42, "Fluzone SD": 28}
    attainment = {
        "Flublok":    [82, 88, 94, 91],
        "Fluzone HD": [76, 84, 92, 96],
        "Fluzone SD": [71, 78, 83, 80],
    }
    sales_rows = []
    for product in products:
        for i, season in enumerate(seasons):
            sales_rows.append(
                {
                    "Season": season,
                    "Product": product,
                    "Doses (000)": round(base_doses[product] * (1 + 0.04 * i), 1),
                    "Attainment %": attainment[product][i],
                }
            )

    # Multi-manufacturer market share trend (within this customer)
    manufacturer_trend_rows = []
    manufacturer_curves = {
        "Sanofi":   [48, 52, 58, 62],
        "Sequirus": [32, 30, 26, 22],
        "Pfizer":   [12, 10,  9,  8],
        "GSK":      [ 8,  8,  7,  8],
    }
    for mfg, values in manufacturer_curves.items():
        for season, value in zip(seasons, values):
            manufacturer_trend_rows.append({"Season": season, "Manufacturer": mfg, "Share %": value})
    market_share = pd.DataFrame(manufacturer_trend_rows)

    # Latest-season snapshot for "Current Purchasing" stacked bar
    current_purchasing = pd.DataFrame(
        [
            {"Group": "Mix", "Manufacturer": "Sanofi",   "Share": 62},
            {"Group": "Mix", "Manufacturer": "Sequirus", "Share": 22},
            {"Group": "Mix", "Manufacturer": "Pfizer",   "Share": 8},
            {"Group": "Mix", "Manufacturer": "GSK",      "Share": 8},
        ]
    )

    brand_share = pd.DataFrame(
        [
            {"Brand": "Flublok",    "Share": 46},
            {"Brand": "Fluzone SD", "Share": 34},
            {"Brand": "Fluzone HD", "Share": 20},
        ]
    )

    segment_share = pd.DataFrame(
        [
            {"Account": "This customer", "Share": 22},
            {"Account": "CVS Health", "Share": 28},
            {"Account": "Walmart Pharmacy", "Share": 18},
            {"Account": "HEB", "Share": 12},
            {"Account": "Other", "Share": 20},
        ]
    )

    return {
        "sales": pd.DataFrame(sales_rows),
        "market_share": market_share,
        "current_purchasing": current_purchasing,
        "brand_share": brand_share,
        "segment_share": segment_share,
    }


def _profile_card(label: str, value: str) -> str:
    return (
        "<div class='current-card' style='flex-direction:column; align-items:flex-start;"
        " min-height:62px; padding:.5rem .75rem;'>"
        f"<div class='lever-value-label'>{escape(label)}</div>"
        f"<div class='current-value'>{escape(value)}</div></div>"
    )


def render_customer_deep_dive_screen(selected_row: pd.Series) -> None:
    """Full-screen Customer Deep Dive (Screen 2)."""
    customer_name = str(selected_row.get("customer_name", "—"))
    segment = str(selected_row.get("customer_segment", "—"))
    tier = str(selected_row.get("customer_tier", selected_row.get("customer_class", "—")))
    region = str(selected_row.get("region", "—"))
    dummy = _deep_dive_dummy_data()

    palette = ["#7432A8", "#36A39B", "#F37000", "#4285F4", "#B44E00"]

    # =================== Section 1: Customer Profile ===================
    with st.container(border=True):
        st.markdown("<div class='section-title'>Customer Profile</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Account context, sales attainment, and share trends.</div>",
            unsafe_allow_html=True,
        )
        cards_html = (
            "<div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:.6rem; margin-top:.4rem;'>"
            + _profile_card("Customer", customer_name)
            + _profile_card("Segment", segment)
            + _profile_card("Tier", tier)
            + _profile_card("Region", region)
            + "</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown("<div style='height:.8rem;'></div>", unsafe_allow_html=True)
        chart_cols = st.columns(2, gap="medium")
        with chart_cols[0]:
            st.markdown(
                "<div class='small-label'>Sales trend & contract attainment — by product</div>",
                unsafe_allow_html=True,
            )
            bars = (
                alt.Chart(dummy["sales"])
                .mark_bar(opacity=0.88)
                .encode(
                    x=alt.X("Season:O", title=None),
                    y=alt.Y("Doses (000):Q", title="Doses (000)"),
                    color=alt.Color(
                        "Product:N",
                        scale=alt.Scale(range=palette[:3]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    xOffset="Product:N",
                    tooltip=["Season", "Product", "Doses (000)", "Attainment %"],
                )
            )
            attainment_avg = (
                dummy["sales"]
                .groupby("Season", as_index=False)["Attainment %"].mean()
                .rename(columns={"Attainment %": "Avg attainment %"})
            )
            line = (
                alt.Chart(attainment_avg)
                .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5, color="#1F1F29")
                .encode(
                    x=alt.X("Season:O"),
                    y=alt.Y(
                        "Avg attainment %:Q",
                        title="Avg attainment %",
                        scale=alt.Scale(domain=[60, 100]),
                    ),
                    tooltip=["Season", alt.Tooltip("Avg attainment %:Q", format=".1f")],
                )
            )
            chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        with chart_cols[1]:
            st.markdown(
                "<div class='small-label'>Market share trend — manufacturer share within this customer</div>",
                unsafe_allow_html=True,
            )
            mfg_chart = (
                alt.Chart(dummy["market_share"])
                .mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=70),
                    strokeWidth=2.6,
                    interpolate="monotone",
                )
                .encode(
                    x=alt.X("Season:O", title=None),
                    y=alt.Y("Share %:Q", title="Share % of customer vaccine spend", scale=alt.Scale(domain=[0, 70])),
                    color=alt.Color(
                        "Manufacturer:N",
                        scale=alt.Scale(
                            domain=["Sanofi", "Sequirus", "Pfizer", "GSK"],
                            range=["#7432A8", "#F37000", "#4285F4", "#36A39B"],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Season", "Manufacturer", alt.Tooltip("Share %:Q", format=".1f")],
                )
                .properties(height=260)
            )
            st.altair_chart(mfg_chart, use_container_width=True)

        # Row 3 of Customer Profile: Current Purchasing | Customer Relationship Level
        st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
        row3 = st.columns(2, gap="medium")
        with row3[0]:
            st.markdown(
                "<div class='small-label'>Current purchasing — manufacturer split (latest season)</div>",
                unsafe_allow_html=True,
            )
            purchasing_chart = (
                alt.Chart(dummy["current_purchasing"])
                .mark_arc(innerRadius=55, outerRadius=95, stroke="#FFFFFF", strokeWidth=2)
                .encode(
                    theta=alt.Theta("Share:Q", stack=True),
                    color=alt.Color(
                        "Manufacturer:N",
                        scale=alt.Scale(
                            domain=["Sanofi", "Sequirus", "Pfizer", "GSK"],
                            range=["#7432A8", "#F37000", "#4285F4", "#36A39B"],
                        ),
                        legend=alt.Legend(orient="right", title=None),
                    ),
                    tooltip=["Manufacturer", alt.Tooltip("Share:Q", format=".0f", title="Share %")],
                )
                .properties(height=220)
            )
            st.altair_chart(purchasing_chart, use_container_width=True)
            st.markdown(
                "<div style='color: var(--text-secondary); font-size:.8rem; margin-top:-.2rem;'>"
                "<strong>Sanofi-primary</strong> account &mdash; 62% of vaccine spend; Sequirus remains the main competitor."
                "</div>",
                unsafe_allow_html=True,
            )

        with row3[1]:
            st.markdown(
                """
                <div class='driver-card' style='min-height: 220px;'>
                    <div class='driver-header'>
                        <div class='driver-title'>Customer Relationship Level</div>
                        <div class='driver-meta'>Engagement intensity (TTM)</div>
                    </div>
                    <div style='display:flex; align-items:center; gap:.9rem; margin: .35rem 0 .8rem 0;'>
                        <span style='font-size:1.5rem; font-weight:820; color:#19766F; line-height:1;'>High</span>
                        <span style='color: var(--text-secondary); font-size:.85rem; line-height:1.4;'>
                            Strong cross-functional engagement across the last 12 months.
                        </span>
                    </div>
                    <div style='display:grid; grid-template-columns: repeat(2, 1fr); gap:.6rem;'>
                        <div style='background:#F8F7FC; border:1px solid #E5E2EF; border-radius:11px; padding:.65rem .8rem;'>
                            <div style='font-size:.72rem; font-weight:800; color:#6B6878; text-transform:uppercase; letter-spacing:.03rem;'># of Calls</div>
                            <div style='font-size:1.85rem; font-weight:820; color:#1F1F29; line-height:1.1; margin-top:.2rem;'>7</div>
                            <div style='font-size:.75rem; color:#6B6878; margin-top:.15rem;'>TTM field calls</div>
                        </div>
                        <div style='background:#F8F7FC; border:1px solid #E5E2EF; border-radius:11px; padding:.65rem .8rem;'>
                            <div style='font-size:.72rem; font-weight:800; color:#6B6878; text-transform:uppercase; letter-spacing:.03rem;'># of Stakeholders Engaged</div>
                            <div style='font-size:1.85rem; font-weight:820; color:#1F1F29; line-height:1.1; margin-top:.2rem;'>12</div>
                            <div style='font-size:.75rem; color:#6B6878; margin-top:.15rem;'>Across procurement, medical, exec</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =================== Section 2: Offer Intelligence ===================
    with st.container(border=True):
        st.markdown("<div class='section-title'>Offer Intelligence — Current Contract</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Active contract terms and prior-year performance.</div>",
            unsafe_allow_html=True,
        )
        contract_id = str(selected_row.get("contract_name", "—"))
        product = str(selected_row.get("brand_name", "—"))
        net_price = float(selected_row.get("prior_net_price", 0) or 0)
        rebate = float(selected_row.get("prior_rebate_pct", 0) or 0)
        reimbursement = float(selected_row.get("reimbursement_rate_pct", 0) or 0)
        volume_attain = float(selected_row.get("prior_year_attainment_pct", 0) or 0)
        returnability = float(selected_row.get("prior_returnability_pct", 0) or 0)
        contract_duration = "2 years"

        top_html = (
            "<div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:.6rem; margin-top:.4rem;'>"
            + _profile_card("Contract ID", contract_id)
            + _profile_card("Products", product)
            + _profile_card("Contract duration", contract_duration)
            + "</div>"
        )
        st.markdown(top_html, unsafe_allow_html=True)

        st.markdown(
            "<div class='small-label' style='margin-top:.7rem;'>Current offer on the table</div>",
            unsafe_allow_html=True,
        )
        lever_html = (
            "<div style='display:grid; grid-template-columns: repeat(5, 1fr); gap:.6rem; margin-top:.3rem;'>"
            + _profile_card("Net Price", f"${net_price:,.2f}")
            + _profile_card("Rebate", f"{rebate:.1f}%")
            + _profile_card("Reimbursement", f"{reimbursement:.1f}%")
            + _profile_card("Volume attainment", f"{volume_attain:.1f}%")
            + _profile_card("Returnability", f"{returnability:.1f}%")
            + "</div>"
        )
        st.markdown(lever_html, unsafe_allow_html=True)

        # Previous Offer block — only when negotiating in round 2 or later
        current_round = int(selected_row.get("round_number", 1) or 1)
        if current_round >= 2:
            prev_round = current_round - 1
            # Dummy prior-round terms: slightly worse than current
            prev_net = net_price * 1.04
            prev_rebate = max(rebate - 1.5, 0)
            prev_reimburse = max(reimbursement - 3, 0)
            prev_volume = max(volume_attain - 4, 0)
            prev_returnability = max(returnability - 0.5, 0)

            def _previous_card(label: str, value: str) -> str:
                return (
                    "<div style='background:#F3F1F8; border:1.5px dashed #B8AEDA; border-radius:11px;"
                    " padding:.55rem .75rem; min-height:62px; display:flex; flex-direction:column;"
                    " justify-content:center; opacity:.92;'>"
                    f"<div class='lever-value-label' style='color:#7B6D9C;'>{escape(label)}</div>"
                    f"<div style='font-weight:760; color:#3E3759; font-size:1rem; text-decoration: line-through;"
                    f" text-decoration-color:#B8AEDA;'>{escape(value)}</div></div>"
                )

            st.markdown(
                f"""
                <div style='display:flex; align-items:center; gap:.6rem; margin-top:1rem;'>
                    <span class='badge' style='background:#EFEAF8; color:#5A3E89; border-color:#D7CCEF;'>
                        Previous Offer &middot; Round {prev_round}
                    </span>
                    <span style='color: var(--text-secondary); font-size:.82rem;'>
                        Terms put forward in the prior round (not accepted)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prev_html = (
                "<div style='display:grid; grid-template-columns: repeat(5, 1fr); gap:.6rem; margin-top:.35rem;'>"
                + _previous_card("Net Price", f"${prev_net:,.2f}")
                + _previous_card("Rebate", f"{prev_rebate:.1f}%")
                + _previous_card("Reimbursement", f"{prev_reimburse:.1f}%")
                + _previous_card("Volume attainment", f"{prev_volume:.1f}%")
                + _previous_card("Returnability", f"{prev_returnability:.1f}%")
                + "</div>"
            )
            st.markdown(prev_html, unsafe_allow_html=True)

    # =================== Section 3: Negotiation Context ===================
    with st.container(border=True):
        st.markdown("<div class='section-title'>Negotiation Context</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Market position, competitive intensity, and recent activity.</div>",
            unsafe_allow_html=True,
        )

        wac_price = net_price * 1.15  # dummy: assume WAC ~15% above net
        expected_reimbursement = reimbursement
        round_num = int(selected_row.get("round_number", 1) or 1)

        m = st.columns(3, gap="small")
        with m[0]:
            render_metric_card("WAC Price", f"${wac_price:,.2f}", "Wholesale Acquisition Cost", "metric-card-green")
        with m[1]:
            render_metric_card("Expected Reimbursement", f"{expected_reimbursement:.1f}%", "Projected reimbursement", "metric-card-blue")
        with m[2]:
            render_metric_card("Negotiation Round", f"Round {round_num}", "Active negotiation round", "metric-card-amber", compact_value=True)

        st.markdown("<div style='height:.7rem;'></div>", unsafe_allow_html=True)

        share_cols = st.columns(2, gap="medium")
        with share_cols[0]:
            st.markdown(
                "<div class='small-label'>Brand share — this customer's portfolio</div>",
                unsafe_allow_html=True,
            )
            donut = (
                alt.Chart(dummy["brand_share"])
                .mark_arc(innerRadius=55, outerRadius=100, stroke="#FFFFFF", strokeWidth=2)
                .encode(
                    theta=alt.Theta("Share:Q", stack=True),
                    color=alt.Color(
                        "Brand:N",
                        scale=alt.Scale(range=palette),
                        legend=alt.Legend(orient="right", title=None),
                    ),
                    tooltip=["Brand", alt.Tooltip("Share:Q", format=".0f", title="Share %")],
                )
                .properties(height=260)
            )
            st.altair_chart(donut, use_container_width=True)

        with share_cols[1]:
            st.markdown(
                "<div class='small-label'>Segment share — vs peer accounts</div>",
                unsafe_allow_html=True,
            )
            donut = (
                alt.Chart(dummy["segment_share"])
                .mark_arc(innerRadius=55, outerRadius=100, stroke="#FFFFFF", strokeWidth=2)
                .encode(
                    theta=alt.Theta("Share:Q", stack=True),
                    color=alt.Color(
                        "Account:N",
                        scale=alt.Scale(range=palette),
                        legend=alt.Legend(orient="right", title=None),
                    ),
                    tooltip=["Account", alt.Tooltip("Share:Q", format=".0f", title="Share %")],
                )
                .properties(height=260)
            )
            st.altair_chart(donut, use_container_width=True)

        st.markdown("<div style='height:.7rem;'></div>", unsafe_allow_html=True)

        intensity_cols = st.columns(2, gap="medium")
        with intensity_cols[0]:
            intensity = "High"
            intensity_color = {"High": "#DC2626", "Medium": "#F37000", "Low": "#36A39B"}[intensity]
            st.markdown(
                f"""
                <div class='driver-card' style='min-height: 200px;'>
                    <div class='driver-header'>
                        <div class='driver-title'>Competition Intensity</div>
                        <div class='driver-meta'>Account-level signal</div>
                    </div>
                    <div style='font-size: 2.1rem; font-weight: 820; color: {intensity_color}; line-height: 1; margin: .35rem 0 .55rem 0;'>{intensity}</div>
                    <ul class='driver-list'>
                        <li><strong>3 active competitors</strong> currently engaged in this account.</li>
                        <li>Aggressive pricing pressure flagged from <strong>Pfizer</strong> and <strong>GSK</strong> over the last two rounds.</li>
                        <li>Customer has explicitly benchmarked against <strong>competitor rebate terms</strong>.</li>
                        <li>Negotiation cycle running ~2 weeks shorter than segment average.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with intensity_cols[1]:
            st.markdown(
                """
                <div class='driver-card' style='min-height: 200px;'>
                    <div class='driver-header'>
                        <div class='driver-title'>Recent Wins / Losses</div>
                        <div class='driver-meta'>Last 4 quarters &nbsp;|&nbsp; 3 wins, 1 loss</div>
                    </div>
                    <ul class='driver-list'>
                        <li><span class='positive-chip'>Won</span> <strong>Walmart Pharmacy</strong> — Flublok, $2.1M annual</li>
                        <li><span class='positive-chip'>Won</span> <strong>HEB</strong> — Fluzone HD, $0.9M annual</li>
                        <li><span class='negative-chip'>Lost</span> <strong>CVS Health</strong> — Fluzone SD, displaced by competitor on rebate</li>
                        <li><span class='positive-chip'>Won</span> <strong>Kroger</strong> — Fluzone HD pilot, expansion expected</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_customer_deep_dive(selected_row: pd.Series) -> None:
    customer_name = escape(str(selected_row.get("customer_name", "—")))
    segment = escape(str(selected_row.get("customer_segment", "—")))
    tier = escape(str(selected_row.get("customer_tier", selected_row.get("customer_class", "—"))))
    region = escape(str(selected_row.get("region", "—")))
    with st.expander("Customer Deep Dive  •  Work in Progress", expanded=False):
        st.markdown(
            f"""
            <div class='section-subtitle' style='margin-top:0;'>Customer-level intelligence — relationship history, attainment trends, and competitive footprint. Coming soon.</div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:.6rem; margin-top:.4rem;">
                <div class='current-card' style='flex-direction:column; align-items:flex-start; min-height:62px; padding:.5rem .75rem;'>
                    <div class='lever-value-label'>Customer</div>
                    <div class='current-value'>{customer_name}</div>
                </div>
                <div class='current-card' style='flex-direction:column; align-items:flex-start; min-height:62px; padding:.5rem .75rem;'>
                    <div class='lever-value-label'>Segment</div>
                    <div class='current-value'>{segment}</div>
                </div>
                <div class='current-card' style='flex-direction:column; align-items:flex-start; min-height:62px; padding:.5rem .75rem;'>
                    <div class='lever-value-label'>Tier</div>
                    <div class='current-value'>{tier}</div>
                </div>
                <div class='current-card' style='flex-direction:column; align-items:flex-start; min-height:62px; padding:.5rem .75rem;'>
                    <div class='lever-value-label'>Region</div>
                    <div class='current-value'>{region}</div>
                </div>
            </div>
            <div style='margin-top:.7rem; color:var(--text-secondary); font-size:.82rem; font-style:italic;'>
                Planned: relationship score trend, prior-season attainment curve, competitive share-of-voice, escalation history.
            </div>
            """,
            unsafe_allow_html=True,
        )


def choose_context(data: pd.DataFrame) -> tuple[pd.Series, str, list[str]]:
    with st.container(border=True):
        st.markdown("<div class='section-title'>Scenario Context</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Select the customer, negotiation round, and brand(s) to simulate.</div>", unsafe_allow_html=True)
        top = st.columns([1, 1.3, .8, 1.6], gap="small")
        segment = safe_selectbox(top[0], "Customer Segment", sorted(data["customer_segment"].unique()), "filter_customer_segment")
        segment_data = data[data["customer_segment"] == segment]
        customer = safe_selectbox(top[1], "Customer", sorted(segment_data["customer_name"].unique()), "filter_customer")
        customer_data = segment_data[segment_data["customer_name"] == customer]

        # Contract and Season are no longer user-selectable — auto-resolve to the
        # customer's most recent (upcoming) season and its associated contract.
        latest_season = sorted(customer_data["season_id"].unique())[-1]
        season_data = customer_data[customer_data["season_id"] == latest_season]
        latest_contract = sorted(season_data["contract_name"].unique())[0]
        contract_data = season_data[season_data["contract_name"] == latest_contract]

        round_options = sorted(contract_data["round_number"].unique())
        round_number = safe_selectbox(top[2], "Round Number", round_options, "filter_round_number")
        round_data = contract_data[contract_data["round_number"] == round_number]

        # Brand options are scoped to the customer's full brand history (not just
        # the latest contract or round), so the dropdown always offers a useful
        # set of choices instead of "No results" once one or two are selected.
        available_brands = sorted(customer_data["brand_name"].dropna().unique())
        if not available_brands:
            available_brands = sorted(contract_data["brand_name"].dropna().unique())
        if not available_brands:
            available_brands = sorted(data["brand_name"].dropna().unique())
        default_brands = available_brands[: min(1, len(available_brands))]
        selected_brands = safe_multiselect(top[3], "Brand / Brands", available_brands, default_brands, "filter_multi_brands")
        if not selected_brands:
            selected_brands = available_brands[:1]

        product_scope = "Single Product" if len(selected_brands) <= 1 else "Multi Product"

    selected_row = round_data.iloc[0] if not round_data.empty else contract_data.iloc[0]
    return selected_row, product_scope, selected_brands


def get_product_rows(data: pd.DataFrame, selected_row: pd.Series, selected_brands: list[str]) -> dict[str, pd.Series]:
    product_rows: dict[str, pd.Series] = {}
    for brand in selected_brands:
        match_levels = [
            data[
                (data["customer_name"] == selected_row["customer_name"])
                & (data["contract_name"] == selected_row["contract_name"])
                & (data["season_id"] == selected_row["season_id"])
                & (data["round_number"] == selected_row["round_number"])
                & (data["brand_name"] == brand)
            ],
            data[
                (data["customer_name"] == selected_row["customer_name"])
                & (data["contract_name"] == selected_row["contract_name"])
                & (data["brand_name"] == brand)
            ],
            data[(data["customer_name"] == selected_row["customer_name"]) & (data["brand_name"] == brand)],
            data[data["brand_name"] == brand],
        ]
        brand_matches = next((matches for matches in match_levels if not matches.empty), pd.DataFrame())
        product_rows[brand] = brand_matches.iloc[0] if not brand_matches.empty else selected_row
    return product_rows


def _lever_table_column_widths(num_brands: int) -> list[float]:
    return [1.25, 1.05] + [1.0] * num_brands


def render_lever_table_header(selected_brands: list[str]) -> None:
    cols = st.columns(_lever_table_column_widths(len(selected_brands)), gap="small")
    labels = ["Metric", "Season"] + list(selected_brands)
    for col, label in zip(cols, labels):
        with col:
            st.markdown(f"<div class='lever-table-header-cell'>{escape(label)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='lever-table-header'></div>", unsafe_allow_html=True)


def render_lever_table_row(
    config: dict[str, Any],
    selected_brands: list[str],
    product_rows: dict[str, pd.Series],
    row_index: int,
) -> None:
    parity = "even" if row_index % 2 == 0 else "odd"
    row_container = st.container(key=f"lever-row-{parity}-{config['key']}")
    with row_container:
        col_widths = _lever_table_column_widths(len(selected_brands))

        # Upcoming Season row — editable, mirrors the current "Proposed" behavior.
        row1 = st.columns(col_widths, gap="small")
        with row1[0]:
            st.markdown(f"<div class='lever-table-metric'>{escape(config['label'])}</div>", unsafe_allow_html=True)
        with row1[1]:
            st.markdown("<div class='lever-table-season-label'>Upcoming Season</div>", unsafe_allow_html=True)
        for col, brand in zip(row1[2:], selected_brands):
            with col:
                st.number_input(
                    f"{brand} {config['label']} upcoming season",
                    min_value=config["min_value"],
                    max_value=config["max_value"],
                    step=config["step"],
                    format=config["format"],
                    key=lever_state_key(brand, config["key"]),
                    label_visibility="collapsed",
                )

        # Previous Season row — read-only, mirrors the current "Current" value.
        row2 = st.columns(col_widths, gap="small")
        with row2[1]:
            st.markdown("<div class='lever-table-season-label lever-table-season-label--prev'>Previous Season</div>", unsafe_allow_html=True)
        for col, brand in zip(row2[2:], selected_brands):
            with col:
                previous_value = default_for_lever(product_rows[brand], config)
                st.markdown(
                    f"<div class='lever-table-prev-value'>{escape(format_value(previous_value, config['value_type']))}</div>",
                    unsafe_allow_html=True,
                )


def render_pricing_levers(product_scope: str, selected_brands: list[str], product_rows: dict[str, pd.Series]) -> tuple[bool, bool, bool]:
    st.markdown("<div class='section-title'>Pricing Levers</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Compare previous-season terms with the upcoming-season proposal, by brand.</div>",
        unsafe_allow_html=True,
    )

    render_lever_table_header(selected_brands)
    for row_index, config in enumerate(LEVER_CONFIGS):
        render_lever_table_row(config, selected_brands, product_rows, row_index)

    buttons = st.columns([1.2, 1, 1], gap="small")
    run_clicked = buttons[0].button("Run Simulation", type="primary", width="stretch")
    reset_clicked = buttons[1].button("Reset Inputs", width="stretch", on_click=reset_scenario, args=(product_rows,))
    save_clicked = buttons[2].button("Save Scenario", width="stretch")
    return run_clicked, reset_clicked, save_clicked


def _decisiveness_factor(p_accept: float, p_reject: float) -> tuple[str, str]:
    gap_pp = abs(p_accept - p_reject) * 100
    accept_pct = f"{p_accept * 100:.0f}%"
    reject_pct = f"{p_reject * 100:.0f}%"
    if gap_pp >= 40:
        return "High", f"Accept {accept_pct}, Reject {reject_pct} — clear separation between the two outcomes."
    if gap_pp >= 15:
        return "Medium", f"Accept {accept_pct}, Reject {reject_pct} — moderate separation between outcomes."
    return "Low", f"Accept {accept_pct}, Reject {reject_pct} — outcomes clustered, no clear winner."


def _historical_similarity_factor(user_inputs: dict[str, Any]) -> tuple[str, str]:
    match = stable_match_score(user_inputs)
    is_existing = str(user_inputs.get("new_vs_existing_customer", "Existing")).strip().lower() == "existing"
    if match >= 85 and is_existing:
        return "High", "Existing customer with similar Fluzone price range, volume commitment, and prior attainment seen across many prior deals."
    if match >= 78:
        return "Medium", f"Similar deals found in training data ({match}% match), but with some scenario differences."
    return "Low", "New customer type or unusual offer structure (e.g., very high volume ask with no comparable historical deals)."


def _feature_reliability_factor(user_inputs: dict[str, Any]) -> tuple[str, str]:
    key_features = [
        "net_price",
        "rebate_pct",
        "volume_commitment",
        "market_share_commitment_pct",
        "reimbursement_rate_pct",
        "prior_year_attainment_pct",
        "competitive_pressure_score",
    ]
    missing = []
    for key in key_features:
        value = user_inputs.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            missing.append(key)
    if not missing:
        return "High", "Net price, volume commitment, prior-year volume, market share, and customer segment are all available from source systems."
    if len(missing) <= 1:
        return "Medium", f"Most inputs are available; {len(missing)} field is being estimated."
    return "Low", "Key inputs such as prior-year volume, share, or competitive pressure are missing or estimated."


def _prediction_consistency_factor(
    model,
    user_inputs: dict[str, Any],
    feature_metadata: dict[str, Any],
) -> tuple[str, str, dict[str, float]]:
    key_num_features = ["net_price", "rebate_pct", "volume_commitment", "reimbursement_rate_pct", "market_share_commitment_pct"]
    accept_pcts: list[float] = []
    deltas = [-0.06, -0.03, 0.0, 0.03, 0.06]
    feature_columns = feature_metadata["feature_columns"]
    for delta in deltas:
        perturbed = {**user_inputs}
        for key in key_num_features:
            value = perturbed.get(key)
            if isinstance(value, (int, float)) and not pd.isna(value):
                perturbed[key] = value * (1 + delta)
        row = {**perturbed, **calculate_derived_fields(perturbed)}
        try:
            input_df = pd.DataFrame([{c: row[c] for c in feature_columns}])
            proba = model.predict_proba(input_df)[0]
            accept_pcts.append(float(proba[OUTCOME_LABELS.index("Accepted")]) * 100)
        except Exception:
            continue
    if not accept_pcts:
        return "Medium", "Perturbation test could not be run — treating as moderate consistency.", {}
    lo, hi = min(accept_pcts), max(accept_pcts)
    spread = hi - lo
    detail = {"lo": lo, "hi": hi, "spread": spread}
    band_text = f"Across five validation runs, Accept ranges {lo:.0f}%–{hi:.0f}%"
    if spread <= 6:
        return "High", f"{band_text} (narrow spread — the model is stable to small input changes).", detail
    if spread <= 20:
        return "Medium", f"{band_text} (moderate spread — some sensitivity to input changes).", detail
    return "Low", f"{band_text} (wide spread — prediction is sensitive to small input changes).", detail


def compute_confidence_factors(
    model,
    user_inputs: dict[str, Any],
    probabilities: dict[str, float],
    feature_metadata: dict[str, Any],
) -> dict[str, dict[str, str]]:
    p_accept = probabilities["Accepted"]
    p_reject = probabilities["Negotiated"] + probabilities["Rejected"]
    d_band, d_desc = _decisiveness_factor(p_accept, p_reject)
    h_band, h_desc = _historical_similarity_factor(user_inputs)
    f_band, f_desc = _feature_reliability_factor(user_inputs)
    c_band, c_desc, c_detail = _prediction_consistency_factor(model, user_inputs, feature_metadata)
    return {
        "decisiveness":  {"label": "Prediction Decisiveness",  "band": d_band, "description": d_desc},
        "historical":    {"label": "Historical Similarity",    "band": h_band, "description": h_desc},
        "reliability":   {"label": "Feature Reliability",      "band": f_band, "description": f_desc},
        "consistency":   {"label": "Prediction Consistency",   "band": c_band, "description": c_desc, "detail": c_detail},
    }


def compute_overall_confidence_band(factors: dict[str, dict[str, str]]) -> str:
    weights = {"High": 2, "Medium": 1, "Low": 0}
    total = sum(weights.get(f["band"], 1) for f in factors.values())
    if total >= 6:
        return "High"
    if total >= 3:
        return "Medium"
    return "Low"


def build_articulated_narrative(result: dict[str, Any]) -> str:
    probabilities = result["probabilities"]
    p_accept = probabilities["Accepted"]
    p_reject = probabilities["Negotiated"] + probabilities["Rejected"]
    is_accepted = p_accept >= p_reject
    gap_pp = abs(p_accept - p_reject) * 100

    customer_name = str(result.get("inputs", {}).get("customer_name", "The customer"))

    drivers = result.get("drivers", {})
    all_drivers: list[dict[str, Any]] = []
    for category in ("commercial_offer", "customer_relationship", "competitive_pressure", "deal_context"):
        all_drivers.extend(drivers.get(category, []))
    top_positive = max(
        (d for d in all_drivers if d.get("direction") == "Positive"),
        key=lambda d: d.get("impact_pp", 0),
        default=None,
    )
    top_negative = min(
        (d for d in all_drivers if d.get("direction") == "Negative"),
        key=lambda d: d.get("impact_pp", 0),
        default=None,
    )

    parts: list[str] = []

    if is_accepted:
        parts.append(
            f"<strong>{escape(customer_name)}</strong> is projected to accept this offer at "
            f"<strong>{p_accept:.0%}</strong> probability — a {gap_pp:.0f}-point lead over the {p_reject:.0%} rejection scenario."
        )
    else:
        parts.append(
            f"<strong>{escape(customer_name)}</strong> is more likely to reject or renegotiate at "
            f"<strong>{p_reject:.0%}</strong>, versus {p_accept:.0%} acceptance — a {gap_pp:.0f}-point lean toward pushback."
        )

    if top_positive:
        pos_text = top_positive["explanation"].strip().rstrip(".")
        parts.append(f"{escape(pos_text)}.")

    if top_negative:
        neg_text = top_negative["explanation"].strip().rstrip(".")
        if neg_text:
            neg_text_lower = neg_text[0].lower() + neg_text[1:]
        else:
            neg_text_lower = neg_text
        if top_positive:
            parts.append(f"However, {escape(neg_text_lower)}.")
        else:
            parts.append(f"The principal risk: {escape(neg_text)}.")

    if is_accepted:
        parts.append("Recommend advancing the offer while monitoring the risk factor above.")
    else:
        name_lower = (top_negative or {}).get("name", "").lower()
        if "price" in name_lower or "rebate" in name_lower:
            parts.append("Recommend tightening the pricing or rebate terms before re-engagement.")
        elif "volume" in name_lower or "share" in name_lower:
            parts.append("Recommend revising the volume or share commitment before re-engagement.")
        else:
            parts.append("Recommend refining the offer terms and re-scoring before re-engagement.")
    return " ".join(parts)


def build_confidence_narrative(
    result: dict[str, Any],
    factors: dict[str, dict[str, str]],
    overall_band: str,
) -> str:
    """Return an HTML bullet list summarising the confidence bucket — short, numeric, no filler."""
    probabilities = result["probabilities"]
    p_accept = probabilities["Accepted"]
    p_reject = probabilities["Negotiated"] + probabilities["Rejected"]
    gap_pp = abs(p_accept - p_reject) * 100
    match_score = stable_match_score(result.get("inputs", {}))

    d_band = factors.get("decisiveness", {}).get("band", "Medium")
    h_band = factors.get("historical", {}).get("band", "Medium")
    f_band = factors.get("reliability", {}).get("band", "Medium")
    c_band = factors.get("consistency", {}).get("band", "Medium")
    c_detail = factors.get("consistency", {}).get("detail", {})
    lo = c_detail.get("lo")
    hi = c_detail.get("hi")
    perturb_range = f"{lo:.0f}–{hi:.0f}%" if lo is not None and hi is not None else "n/a"

    decisive_tag = {"High": "decisive", "Medium": "moderate", "Low": "ambiguous"}[d_band]
    hist_tag = {"High": "strong precedent", "Medium": "partial precedent", "Low": "limited precedent"}[h_band]
    stability_tag = {"High": "stable", "Medium": "some sensitivity", "Low": "sensitive"}[c_band]
    reliability_tag = {
        "High": "all key inputs measured",
        "Medium": "most inputs measured, 1 estimated",
        "Low": "several inputs estimated",
    }[f_band]
    action = {
        "High": "Proceed with standard risk margin.",
        "Medium": "Validate key drivers with the account team before finalizing.",
        "Low": "Treat as a hypothesis — gather more intel before deciding.",
    }[overall_band]

    bullets = [
        f"<strong>Probability split:</strong> {p_accept:.0%} vs {p_reject:.0%} ({gap_pp:.0f}-pt gap) — {decisive_tag}",
        f"<strong>Historical match:</strong> {match_score}% similarity — {hist_tag}",
        f"<strong>Stability:</strong> Accept range {perturb_range} under stress-test — {stability_tag}",
        f"<strong>Data quality:</strong> {reliability_tag}",
        f"<strong>Action:</strong> {action}",
    ]
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"<ul style='margin:.1rem 0 0 1.1rem; padding:0; line-height:1.65;'>{items}</ul>"


def render_confidence_bucket(factors: dict[str, dict[str, str]], overall_band: str) -> None:
    band_color = {"High": "#19766F", "Medium": "#B44E00", "Low": "#DC2626"}[overall_band]
    band_bg    = {"High": "#EAF8F6", "Medium": "#FFF3E8", "Low": "#FEE2E2"}[overall_band]
    band_border= {"High": "#BDE4E0", "Medium": "#FFC58F", "Low": "#FCA5A5"}[overall_band]
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:.9rem; margin: .3rem 0 .55rem 0;">
            <span style='font-size:.72rem; font-weight:800; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.03rem;'>Confidence</span>
            <span style='display:inline-flex; align-items:center; padding:.35rem .85rem; border-radius:999px;
                background:{band_bg}; color:{band_color}; border:1px solid {band_border};
                font-size:1.05rem; font-weight:820; letter-spacing:.02rem;'>{escape(overall_band)}</span>
            <span style='color:var(--text-secondary); font-size:.85rem;'>Bucket derived from four sub-factors below.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="small")
    factor_order = ["decisiveness", "historical", "reliability", "consistency"]
    for col, key in zip(cols, factor_order):
        factor = factors.get(key, {})
        band = factor.get("band", "Medium")
        desc = factor.get("description", "")
        label = factor.get("label", key)
        pill_color = {"High": "#19766F", "Medium": "#B44E00", "Low": "#DC2626"}[band]
        pill_bg    = {"High": "#EAF8F6", "Medium": "#FFF3E8", "Low": "#FEE2E2"}[band]
        pill_border= {"High": "#BDE4E0", "Medium": "#FFC58F", "Low": "#FCA5A5"}[band]
        with col:
            st.markdown(
                f"""
                <div style='border:1px solid var(--border); border-radius:12px; padding:.75rem .85rem;
                            background:#FFFFFF; min-height:150px; box-shadow: 0 1px 2px rgba(31,31,41,.04);'>
                    <div style='display:flex; align-items:center; justify-content:space-between; gap:.4rem; margin-bottom:.5rem;'>
                        <div style='font-size:.82rem; font-weight:800; color:var(--text-primary); line-height:1.15;'>{escape(label)}</div>
                        <span style='display:inline-flex; padding:.16rem .55rem; border-radius:999px;
                            background:{pill_bg}; color:{pill_color}; border:1px solid {pill_border};
                            font-size:.72rem; font-weight:800; letter-spacing:.02rem;'>{escape(band)}</span>
                    </div>
                    <div style='font-size:.82rem; color:var(--text-secondary); line-height:1.4;'>{escape(desc)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_outcome(result: dict[str, Any] | None) -> None:
    st.markdown("<div class='section-title'>Pricing Simulation Outcome</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Run a simulation to calculate the likely outcome and confidence.</div>", unsafe_allow_html=True)
    if not result:
        st.info("Adjust the proposed levers and click Run Simulation.")
        return

    probabilities = result["probabilities"]
    p_accept = probabilities["Accepted"]
    p_reject = probabilities["Negotiated"] + probabilities["Rejected"]
    factors = result.get("confidence_factors")
    overall_band = result.get("overall_confidence_band", result.get("confidence_band", "Medium"))
    confidence_card_class = {
        "High":   "metric-card-green",
        "Medium": "metric-card-amber",
        "Low":    "metric-card-red",
    }.get(overall_band, "metric-card-blue")

    metric_cols = st.columns(3, gap="small")
    with metric_cols[0]:
        render_metric_card("Accepted", f"{p_accept:.0%}", "Likelihood customer accepts proposed offer", "metric-card-green")
    with metric_cols[1]:
        render_metric_card("Rejected", f"{p_reject:.0%}", "Likelihood customer rejects or renegotiates", "metric-card-red")
    with metric_cols[2]:
        render_metric_card("Confidence", overall_band, "Overall confidence in the forecast", confidence_card_class, compact_value=True)

    narrative_html = build_articulated_narrative(result)
    st.markdown("<div class='small-label' style='margin-top:.9rem;'>Narrative Summary</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='narrative-card'>{narrative_html}</div>", unsafe_allow_html=True)

    if factors:
        confidence_narrative_html = build_confidence_narrative(result, factors, overall_band)
        st.markdown("<div class='small-label' style='margin-top:.9rem;'>Confidence Interpretation</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='narrative-card'>{confidence_narrative_html}</div>", unsafe_allow_html=True)

    product_results = result.get("product_results", [])
    if len(product_results) > 1:
        st.markdown("<div class='small-label' style='margin-top:.9rem;'>Product-level simulation</div>", unsafe_allow_html=True)
        product_df = pd.DataFrame(
            [
                {
                    "Product": item["product"],
                    "Accepted": f"{item['probabilities']['Accepted']:.1%}",
                    "Rejected": f"{(item['probabilities']['Negotiated'] + item['probabilities']['Rejected']):.1%}",
                    "Recommendation": "Accepted" if item['probabilities']['Accepted'] >= (item['probabilities']['Negotiated'] + item['probabilities']['Rejected']) else "Rejected",
                }
                for item in product_results
            ]
        )
        st.dataframe(product_df, width="stretch", hide_index=True)

    st.markdown("<div style='height:.55rem;'></div>", unsafe_allow_html=True)
    render_driver_section(result["drivers"])

    shap_contributions = result.get("shap_contributions")
    if shap_contributions is not None:
        render_shap_chart(shap_contributions)


def save_scenario(customer: str, contract: str, scenario_inputs: dict[str, Any], result: dict[str, Any]) -> None:
    probabilities = result["probabilities"]
    st.session_state["saved_scenarios"].append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Customer": customer,
            "Contract": contract,
            "Brand scope": scenario_inputs["brand_scope"],
            "Proposed net price": scenario_inputs["net_price"],
            "Proposed rebate": scenario_inputs["rebate_pct"],
            "Accepted probability": round(probabilities["Accepted"], 4),
            "Negotiated probability": round(probabilities["Negotiated"], 4),
            "Rejected probability": round(probabilities["Rejected"], 4),
            "Recommended outcome": result["recommended_outcome"],
            "Confidence score": result["confidence_score"],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Customer Decision Engine", layout="wide")
    inject_css()

    missing = [path for path in [DATA_PATH, MODEL_PATH] if not path.exists()]
    if missing:
        render_header("v0.1")
        show_setup_block(missing)

    try:
        data = load_contracting_data()
        model = load_model()
        feature_metadata = load_feature_metadata()
        model_metadata = load_model_metadata()
    except (FileNotFoundError, ValueError) as exc:
        render_header("v0.1")
        st.error(str(exc))
        st.code("python src/generate_dummy_data.py\npython src/train_model.py\nstreamlit run app.py", language="powershell")
        st.stop()

    render_header(model_metadata.get("model_version", "v0.1"))

    st.session_state.setdefault("saved_scenarios", [])
    st.session_state.setdefault("latest_result", None)

    selected_row, product_scope, selected_brands = choose_context(data)
    context_signature = "|".join(
        [
            str(selected_row["customer_name"]),
            str(selected_row["contract_name"]),
            str(selected_row["season_id"]),
            str(selected_row["round_number"]),
            product_scope,
            ",".join(selected_brands),
        ]
    )
    context_changed = st.session_state.get("context_signature") != context_signature
    if context_changed:
        st.session_state["context_signature"] = context_signature
        st.session_state["latest_result"] = None
    product_rows = get_product_rows(data, selected_row, selected_brands)
    if context_changed:
        reset_product_values(product_rows)
    else:
        initialize_product_values(product_rows)

    tab_simulator, tab_deep_dive = st.tabs(["Pricing Simulator", "Customer Deep Dive"])

    with tab_simulator:
        # Computed up front (reads already-initialized session_state) so the entire
        # left column — levers panel + watchouts — can render inside one sticky wrapper.
        scenario_inputs_by_brand = {
            brand: build_scenario_inputs(product_rows[brand], product_scope, selected_brands, brand)
            for brand in selected_brands
        }
        watchouts = [
            f"{brand}: {watchout}"
            for brand, inputs in scenario_inputs_by_brand.items()
            for watchout in collect_scenario_watchouts(inputs)
        ]

        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            with st.container(key="lever-sticky-panel"):
                with st.container(border=True):
                    run_clicked, reset_clicked, save_clicked = render_pricing_levers(product_scope, selected_brands, product_rows)
                render_watchouts(watchouts)

        if run_clicked:
            st.session_state["latest_result"] = score_selected_products(
                model,
                feature_metadata,
                product_scope,
                selected_brands,
                product_rows,
            )

        if save_clicked:
            if st.session_state.get("latest_result") is None:
                st.toast("Run a simulation before saving a scenario.")
            else:
                save_scenario(
                    str(selected_row["customer_name"]),
                    str(selected_row["contract_name"]),
                    st.session_state["latest_result"]["inputs"],
                    st.session_state["latest_result"],
                )
                st.toast("Scenario saved.")

        with right:
            with st.container(border=True):
                render_outcome(st.session_state.get("latest_result"))

        render_rationale(model_metadata, st.session_state.get("latest_result"))
        render_saved_scenarios()

    with tab_deep_dive:
        render_customer_deep_dive_screen(selected_row)

    st.caption("Synthetic PoC model for demonstration only. Do not use for real contracting decisions.")


if __name__ == "__main__":
    main()
