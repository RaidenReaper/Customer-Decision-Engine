from __future__ import annotations

import hashlib
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.scoring import (
    DRIVER_CATEGORIES,
    load_feature_metadata,
    load_model,
    load_model_metadata,
    score_customer_response,
)
from src.utils import DATA_PATH, MODEL_PATH, calculate_derived_fields, load_contracting_data


BRAND_FAMILY = {
    "FluShield QIV": "Influenza",
    "FluShield HD": "Influenza",
    "VaxiGuard RSV": "Respiratory",
    "PneuProtect": "Pneumococcal",
    "TravelVax": "Travel Vaccine",
}

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
        "label": "Reimbursement Rate %",
        "key": "reimbursement_rate_pct",
        "current_column": "reimbursement_rate_pct",
        "value_type": "percent",
        "helper": "Expected reimbursement",
        "min_value": 0.0,
        "max_value": 150.0,
        "step": 0.5,
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
            background: #FFFFFF; min-height: 112px; box-shadow: 0 1px 2px rgba(31,31,41,.04);
        }
        .metric-card-green {border-left: 5px solid var(--teal); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-teal) 100%);}
        .metric-card-amber {border-left: 5px solid var(--orange); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-orange) 100%);}
        .metric-card-red {border-left: 5px solid var(--red); background: linear-gradient(180deg, #FFFFFF 0%, #FFF1F2 100%);}
        .metric-card-blue {border-left: 5px solid var(--blue); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-blue) 100%);}
        .metric-label {font-size: .76rem; color: var(--text-secondary); font-weight: 760; text-transform: uppercase;}
        .metric-value {font-size: 1.65rem; color: var(--text-primary); font-weight: 820; margin-top: .24rem;}
        .metric-help {font-size: .78rem; color: var(--text-secondary); margin-top: .22rem;}
        .current-card {
            background: #F8F7FC; border: 1px solid var(--border); border-radius: 12px;
            padding: .54rem .62rem; min-height: 68px;
        }
        .current-value {font-size: 1rem; font-weight: 780; color: var(--text-primary); margin-top: .18rem;}
        .helper-text {font-size: .75rem; color: var(--text-secondary); margin-top: .08rem;}
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
        .lever-input-wrap {padding-top: .05rem;}

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


def render_metric_card(label: str, value: str, help_text: str, css_class: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-help">{escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"""
        <div class="current-card">
            <div class="small-label">Current {escape(label)}</div>
            <div class="current-value">{escape(value)}</div>
            <div class="helper-text">{escape(helper)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lever(
    label: str,
    state_key: str,
    current_value: float | int,
    value_type: str,
    helper: str,
    min_value: float | int,
    max_value: float | int,
    step: float | int,
    format_string: str | None = None,
) -> None:
    left, right = st.columns([0.38, 0.62], gap="small")
    with left:
        render_current_card(label, format_value(current_value, value_type), helper)
    with right:
        st.number_input(
            f"Proposed {label}",
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=format_string,
            key=state_key,
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
        return result

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
    return aggregate_product_results(product_results, portfolio_inputs)


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


def choose_context(data: pd.DataFrame) -> tuple[pd.Series, str, list[str]]:
    with st.container(border=True):
        st.markdown("<div class='section-title'>Scenario Context</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Select the customer, contract, product scope, season, and negotiation round.</div>", unsafe_allow_html=True)
        top = st.columns([1, 1.15, 1.5, .9], gap="small")
        segment = top[0].selectbox("Customer Segment", sorted(data["customer_segment"].unique()))
        segment_data = data[data["customer_segment"] == segment]
        customer = top[1].selectbox("Customer", sorted(segment_data["customer_name"].unique()))
        customer_data = segment_data[segment_data["customer_name"] == customer]
        contract = top[2].selectbox("Contract Name", sorted(customer_data["contract_name"].unique()))
        contract_data = customer_data[customer_data["contract_name"] == contract]
        season = top[3].selectbox("Season", sorted(contract_data["season_id"].unique()))

        bottom = st.columns([1, 1.15, 1.5], gap="small")
        season_data = contract_data[contract_data["season_id"] == season]
        round_options = sorted(season_data["round_number"].unique())
        round_number = bottom[0].selectbox("Round Number", round_options)
        round_data = season_data[season_data["round_number"] == round_number]
        product_scope = bottom[1].selectbox("Product Scope", ["Single Product", "Multi Product"])
        available_brands = sorted(data["brand_name"].unique())
        if product_scope == "Single Product":
            selected_brand = bottom[2].selectbox("Brand / Brands", available_brands)
            selected_brands = [selected_brand]
        else:
            default_brands = available_brands[: min(2, len(available_brands))]
            selected_brands = bottom[2].multiselect("Brand / Brands", available_brands, default=default_brands)
            if not selected_brands:
                selected_brands = default_brands

    selected_row = round_data.iloc[0] if not round_data.empty else contract_data.iloc[0]
    return selected_row, product_scope, selected_brands


def get_product_rows(data: pd.DataFrame, selected_row: pd.Series, selected_brands: list[str]) -> dict[str, pd.Series]:
    product_rows: dict[str, pd.Series] = {}
    customer_matches = data[data["customer_name"] == selected_row["customer_name"]]
    for brand in selected_brands:
        brand_matches = customer_matches[customer_matches["brand_name"] == brand]
        if brand_matches.empty:
            brand_matches = data[data["brand_name"] == brand]
        product_rows[brand] = brand_matches.iloc[0] if not brand_matches.empty else selected_row
    return product_rows


def render_product_levers(row: pd.Series, brand: str) -> None:
    st.markdown(f"<div class='small-label'>Proposed terms for {escape(brand)}</div>", unsafe_allow_html=True)
    for config in LEVER_CONFIGS:
        current_value = default_for_lever(row, config)
        render_lever(
            config["label"],
            lever_state_key(brand, config["key"]),
            current_value,
            config["value_type"],
            config["helper"],
            config["min_value"],
            config["max_value"],
            config["step"],
            config["format"],
        )


def render_pricing_levers(product_scope: str, selected_brands: list[str], product_rows: dict[str, pd.Series]) -> tuple[bool, bool, bool]:
    st.markdown("<div class='section-title'>Pricing Levers</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Compare current contract terms with proposed scenario values.</div>",
        unsafe_allow_html=True,
    )

    if product_scope == "Multi Product" and len(selected_brands) > 1:
        st.caption("Each selected product has its own proposed values. The simulation scores products separately and aggregates the portfolio view.")
        tabs = st.tabs(selected_brands)
        for tab, brand in zip(tabs, selected_brands):
            with tab:
                render_product_levers(product_rows[brand], brand)
    else:
        brand = selected_brands[0]
        render_product_levers(product_rows[brand], brand)

    buttons = st.columns([1.2, 1, 1], gap="small")
    run_clicked = buttons[0].button("Run Simulation", type="primary", width="stretch")
    reset_clicked = buttons[1].button("Reset Inputs", width="stretch", on_click=reset_scenario, args=(product_rows,))
    save_clicked = buttons[2].button("Save Scenario", width="stretch")
    return run_clicked, reset_clicked, save_clicked


def render_outcome(result: dict[str, Any] | None) -> None:
    st.markdown("<div class='section-title'>Pricing Simulation Outcome</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Run a simulation to calculate probabilities and confidence.</div>", unsafe_allow_html=True)
    if not result:
        st.info("Adjust the proposed levers and click Run Simulation.")
        return

    probabilities = result["probabilities"]
    metric_cols = st.columns(5, gap="small")
    with metric_cols[0]:
        render_metric_card("Accepted", f"{probabilities['Accepted']:.0%}", "Likelihood customer accepts proposed offer", "metric-card-green")
    with metric_cols[1]:
        render_metric_card("Negotiated", f"{probabilities['Negotiated']:.0%}", "Likelihood customer requests changes", "metric-card-amber")
    with metric_cols[2]:
        render_metric_card("Rejected", f"{probabilities['Rejected']:.0%}", "Likelihood customer rejects the offer", "metric-card-red")
    with metric_cols[3]:
        render_metric_card("Recommended", result["recommended_outcome"], "Highest probability outcome", "metric-card-blue")
    with metric_cols[4]:
        render_metric_card("Confidence", f"{result['confidence_score']:.0f}%", f"{result['confidence_band']} confidence band", "metric-card-blue")

    st.markdown("<div class='small-label' style='margin-top:.9rem;'>Probability distribution</div>", unsafe_allow_html=True)
    render_stacked_bar(probabilities)

    match_score = stable_match_score(result["inputs"])
    st.markdown("<div class='small-label' style='margin-top:.9rem;'>Confidence interpretation</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="narrative-card">
            <strong>{result['confidence_score']:.1f}% confidence | {escape(result['confidence_band'])}</strong><br>
            Similar deal match: <strong>{match_score}%</strong>. High confidence means the proposed scenario closely matches historical synthetic deal patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='small-label' style='margin-top:.9rem;'>Narrative Summary</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='narrative-card'>{escape(result['narrative_summary'])}</div>", unsafe_allow_html=True)

    product_results = result.get("product_results", [])
    if len(product_results) > 1:
        st.markdown("<div class='small-label' style='margin-top:.9rem;'>Product-level simulation</div>", unsafe_allow_html=True)
        product_df = pd.DataFrame(
            [
                {
                    "Product": item["product"],
                    "Accepted": f"{item['probabilities']['Accepted']:.1%}",
                    "Negotiated": f"{item['probabilities']['Negotiated']:.1%}",
                    "Rejected": f"{item['probabilities']['Rejected']:.1%}",
                    "Recommendation": item["recommended_outcome"],
                    "Confidence": f"{item['confidence_score']:.1f}%",
                }
                for item in product_results
            ]
        )
        st.dataframe(product_df, width="stretch", hide_index=True)

    st.markdown("<div style='height:.55rem;'></div>", unsafe_allow_html=True)
    render_driver_section(result["drivers"])


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

    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        with st.container(border=True):
            run_clicked, reset_clicked, save_clicked = render_pricing_levers(product_scope, selected_brands, product_rows)

    scenario_inputs_by_brand = {
        brand: build_scenario_inputs(product_rows[brand], product_scope, selected_brands, brand)
        for brand in selected_brands
    }
    watchouts = [
        f"{brand}: {watchout}"
        for brand, inputs in scenario_inputs_by_brand.items()
        for watchout in collect_scenario_watchouts(inputs)
    ]

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

    with left:
        render_watchouts(watchouts)

    with right:
        with st.container(border=True):
            render_outcome(st.session_state.get("latest_result"))

    render_rationale(model_metadata, st.session_state.get("latest_result"))
    render_saved_scenarios()

    st.caption("Synthetic PoC model for demonstration only. Do not use for real contracting decisions.")


if __name__ == "__main__":
    main()
