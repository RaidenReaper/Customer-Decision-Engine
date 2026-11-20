"""Customer Decision Engine — win/loss simulator.

Same screen structure as the synthetic PoC: scenario context, a sticky
per-product lever table, and an outcome column with metric cards, narrative,
driver influence and SHAP. The data, model and levers are real.

One XGBoost model per product (Viresta / Immunexa / Noventra), each with its own
decision threshold.
"""
from __future__ import annotations

import base64
import os
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any




import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from landing import active_variant, go_to, page_switch_effects, render_landing
from src import narrative_business, narrative_technical
from src.access import resolve_views, roles_from_claim, view_accounts
from src.confidence import BAND_HIGH, BAND_MODERATE
from src.report import build_drivers_pdf, build_scenario_pdf, report_filename
from src.scoring import (
    compute_shap_contributions,
    confidence_at_probability,
    confidence_band,
    load_fold_models,
    load_models,
    price_sensitivity_curve,
    score_scenario,
    symmetric_price_sensitivity,
)
from src.utils import (
    ACCOUNT_ID_COLUMN,
    ADDRESS_COLUMN,
    ADVOCACY_COLUMN,
    APP_VERSION,
    build_date,
    LEVER_FEATURES,
    lever_label_for_value,
    lever_options,
    lever_value_for_label,
    CONTRACT_ID_COLUMN,
    CURRENT_SHARE_COLUMN,
    CURRENT_VOLUME_COLUMN,
    CUSTOMER_COLUMN,
    DDD_DOSE_COLUMNS,
    DEFAULT_SCENARIO,
    EMPTY_CATEGORICAL_VALUES,
    MARGIN_COLUMNS,
    PREMIUM_COLUMN,
    PRICE_BASE_COLUMN,
    PRIOR_INCUMBENT_COLUMN,
    PRIOR_SHARE_COLUMN,
    REFERENCE_PRICE_COLUMN,
    STANDARD_PRICE_BENCHMARK_COLUMN,
    SHIPMENTS_COLUMN,
    TOTAL_PREBOOK_COLUMN,
    FEATURE_LABELS,
    NUMERIC_FEATURE_COLUMNS,
    NARRATIVE_EXCLUDED_FEATURES,
    HISTORY_SEASONS,
    PRICE_YOY_COLUMN,
    PRIOR_PREBOOK_COLUMN,
    QUADRANT_COLUMN,
    SENSITIVITY_COLUMN,
    SENSITIVITY_BAND_LMH_COLUMN,
    ACCOUNT_SENSITIVITY_BAND_COLUMN,
    ACCOUNT_CONFIDENCE_LABEL_COLUMN,
    PRIOR_VOLUME_COLUMN,
    PRODUCT_COLUMN,
    PRODUCT_KEYS,
    PRODUCT_VALUES,
    SEASON,
    SHARE_CHANGE_COLUMN,
    shap_chart_contributions,
    merge_shap_features,
    redistribute_shap,
    load_features,
    load_history,
    load_model_metadata,
    product_key,
)

PRICE_GROUP_COLUMN = "cy_rmus_price_group"

# Stroke icons rather than emoji: emoji render at different sizes and weights
# per platform, so the four row headers never matched each other.
_ICON_OPEN = (
    "<svg viewBox='0 0 24 24' width='19' height='19' fill='none' "
    "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
    "stroke-linejoin='round'>"
)
SIGNAL_ICONS = {
    "customer": _ICON_OPEN + "<circle cx='12' cy='8' r='3.4'/>"
    "<path d='M5 20c0-3.9 3.1-7 7-7s7 3.1 7 7'/></svg>",
    "negotiation": _ICON_OPEN + "<path d='m11 17 2 2a1 1 0 1 0 3-3'/>"
    "<path d='m14 14 2.5 2.5a1 1 0 1 0 3-3l-3.88-3.88a3 3 0 0 0-4.24 0l-.88.88"
    "a1 1 0 1 1-3-3l2.81-2.81a5.79 5.79 0 0 1 7.06-.87l.47.28a2 2 0 0 0 1.42.25L21 4'/>"
    "<path d='m21 3 1 11h-2'/><path d='M3 3 2 14l6.5 6.5a1 1 0 1 0 3-3'/>"
    "<path d='M3 4h8'/></svg>",
    "competitive": _ICON_OPEN + "<path d='M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6z'/></svg>",
    "offer": _ICON_OPEN + "<path d='M3 12V4h8l9 9-8 8-9-9z'/>"
    "<circle cx='7.6' cy='7.6' r='1.3'/></svg>",
    "drivers": _ICON_OPEN + "<circle cx='12' cy='12' r='9'/>"
    "<circle cx='12' cy='12' r='3.4' fill='currentColor' stroke='none'/></svg>",
    "signal": _ICON_OPEN + "<path d='M4 13h2.4l1.8-6 3.2 12 2.4-14 2.2 8H20'/></svg>",
    "trophy": _ICON_OPEN + "<path d='M8 21h8'/><path d='M12 17v4'/>"
    "<path d='M7 4h10v4a5 5 0 0 1-10 0z'/>"
    "<path d='M7 6H4a3 3 0 0 0 3 3'/><path d='M17 6h3a3 3 0 0 1-3 3'/></svg>",
    "heart": _ICON_OPEN + "<path d='M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 "
    "5.5 0 1 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z'/></svg>",
    "pulse": _ICON_OPEN + "<path d='M22 12h-4l-3 9L9 3l-3 9H2'/></svg>",
    "trending": _ICON_OPEN + "<path d='M3 17l6-6 4 4 8-8'/><path d='M15 7h6v6'/></svg>",
    "dollar": _ICON_OPEN + "<path d='M12 1v22'/>"
    "<path d='M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'/></svg>",
    "refresh": _ICON_OPEN + "<path d='M23 4v6h-6'/><path d='M1 20v-6h6'/>"
    "<path d='M3.51 9a9 9 0 0 1 14.85-3.36L23 10'/>"
    "<path d='M1 14l4.64 4.36A9 9 0 0 0 20.49 15'/></svg>",
    "percent": _ICON_OPEN + "<line x1='19' y1='5' x2='5' y2='19'/>"
    "<circle cx='6.5' cy='6.5' r='2.5'/><circle cx='17.5' cy='17.5' r='2.5'/></svg>",
}

_KPI_ICON_OPEN = (
    "<svg viewBox='0 0 24 24' width='16' height='16' fill='none' "
    "stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
    "stroke-linejoin='round'>"
)
KPI_ICONS = {
    "quadrant": _KPI_ICON_OPEN + "<circle cx='12' cy='12' r='9'/>"
    "<circle cx='12' cy='12' r='5.2'/><circle cx='12' cy='12' r='1.4' fill='currentColor'/></svg>",
    "price": _KPI_ICON_OPEN + "<path d='M3 12V4h8l9 9-8 8-9-9z'/>"
    "<circle cx='7.6' cy='7.6' r='1.3'/></svg>",
    "shipments": _KPI_ICON_OPEN + "<rect x='1' y='7' width='13' height='9' rx='1'/>"
    "<path d='M14 10h4l3 3v3h-7z'/><circle cx='5.5' cy='18.2' r='1.6'/>"
    "<circle cx='17.5' cy='18.2' r='1.6'/></svg>",
    "prebooks": _KPI_ICON_OPEN + "<rect x='3' y='5' width='18' height='16' rx='2'/>"
    "<path d='M3 9.5h18'/><path d='M8 3v4'/><path d='M16 3v4'/></svg>",
    "share": _KPI_ICON_OPEN + "<circle cx='12' cy='12' r='9'/>"
    "<path d='M12 3v9l7.8 4.5'/></svg>",
    "tam": _KPI_ICON_OPEN + "<circle cx='12' cy='12' r='9'/>"
    "<ellipse cx='12' cy='12' rx='4' ry='9'/><path d='M3 12h18'/></svg>",
    "sensitivity": _KPI_ICON_OPEN + "<path d='M3 15a9 9 0 0 1 18 0'/>"
    "<path d='M12 15l4.5-5.5'/><circle cx='12' cy='15' r='1.4' fill='currentColor'/></svg>",
}

# One row per driver (replacing the earlier Customer Context / Negotiation
# Context / etc. grouping), as in Prod. v8 has no composite driver scores, so
# the Driver Value column is gone: every tile is one of the driver's own
# source columns, most of them now direct model inputs. "kind" selects the
# formatter in _signal_text.
# Rows ordered and split to match the four-category client taxonomy (11/6,
# same decision as DRIVER_CATEGORIES below): the SAM risk flag left the
# Competitive Pressure row for its own row under Relationship with Customer;
# Prior Year Wallet Share left Relationship Strength for its own Share of
# Wallet row under Customer Loyalty; the Contract Structure row split into
# Buying Group (Price Sensitivity) and Multi-year Contract (Customer
# Loyalty). Tiles themselves are unchanged.
SIGNAL_GROUPS = [
    # --- Price Sensitivity ---------------------------------------------------
    {
        "title": "Price vs. Standard Benchmark Offer",
        "icon": SIGNAL_ICONS["offer"],
        "tiles": [
            {"label": "Standard Price Benchmark", "column": "standard_price_benchmark", "kind": "currency"},
            {"label": "Dead-Net Price", "column": "dead_net_price_per_dose", "kind": "currency", "direction": "below_benchmark", "range_bar": False},
        ],
    },
    {
        "title": "Year-over-year Price Change",
        "icon": SIGNAL_ICONS["trending"],
        "tiles": [
            {"label": "CY Price", "column": "dead_net_price_per_dose", "kind": "currency", "direction": "below_py", "range_bar": False},
            {"label": "PY Price", "column": "py_dead_price_customer", "kind": "currency", "range_bar": False},
        ],
    },
    {
        "title": "Price vs. Cheapest Competitor Price",
        "icon": SIGNAL_ICONS["trophy"],
        "tiles": [
            # v27 measures the premium against the CHEAPEST competitor (Competitor A
            # for HD / Noventra, the cheaper of Competitor B / Competitor C for STD).
            {"label": "Cheapest Competitor Price", "column": "competitor_price_used", "kind": "currency", "range_bar": False},
            {"label": "Premium vs Cheapest", "column": "premium_gap_vs_min_competitor_pct", "kind": "pct1signed", "direction": "lower_better", "range_bar": False},
        ],
    },
    {
        "title": "Negotiation Receptivity",
        "icon": SIGNAL_ICONS["drivers"],
        "tiles": [
            {"label": "Negotiation Stance", "column": "negotiation_stance_bucket", "kind": "negotiation_stance", "direction": "negotiation"},
        ],
    },
    {
        "title": "Buying Group",
        "icon": SIGNAL_ICONS["offer"],
        "tiles": [
            # A v27 driver (native categorical). Not a lever.
            {"label": "Buying Group", "column": "buying_group_5", "kind": "text"},
        ],
    },
    # --- Customer Loyalty ----------------------------------------------------
    {
        "title": "Share of Wallet",
        "icon": SIGNAL_ICONS["heart"],
        "tiles": [
            {
                "label": "Prior Year Wallet Share",
                "column": "bob_Prior Season Shr of Wallet (L1BG)",
                "kind": "text",
                "direction": "wallet_share",
            },
        ],
    },
    {
        "title": "Competitive Pressure",
        "icon": SIGNAL_ICONS["trophy"],
        "tiles": [
            # Flips over the prior 2 seasons (current season excluded), the
            # Customer share change over prior seasons in percentage points, and
            # the YoY price move. The SAM risk flag has its own row under
            # Relationship with Customer (11/6 taxonomy).
            {"label": "Flip Count", "column": "flips_prior_2_seasons", "kind": "compact", "direction": "zero_good"},
            {"label": "Market Share Change", "column": "yoy_ms_change_pct_prior", "kind": "pts1signed", "direction": "higher_better", "range_bar": False},
            # v27's YoY driver is in % (it was a $ delta before v27).
            {"label": "YoY Price Change", "column": "yoy_price_change_pct", "kind": "pct1signed", "direction": "lower_better"},
        ],
    },
    {
        "title": "Multi-year Contract",
        "icon": SIGNAL_ICONS["offer"],
        "tiles": [
            # A v27 driver. Not a lever.
            {"label": "Multi-year Contract", "column": "multi_year_flag", "kind": "yesno"},
        ],
    },
    # --- Relationship with Customer --------------------------------------------
    {
        "title": "Relationship Strength",
        "icon": SIGNAL_ICONS["heart"],
        "tiles": [
            {
                "label": "Prior Year Doses",
                "column": "bob_L1BG Prior Season Customer Doses",
                "kind": "compact",
                "direction": "nonzero_good",
                "range_bar": False,
            },
        ],
    },
    {
        "title": "Field Engagement Health",
        "icon": SIGNAL_ICONS["customer"],
        "tiles": [
            {"label": "Engagement Score", "column": "cde_engagement_length_log", "kind": "num2"},
            {"label": "Net Call Sentiment", "column": "cde_net_sentiment_score", "kind": "num2", "direction": "higher_better"},
        ],
    },
    {
        "title": "SAM Risk Flag",
        "icon": SIGNAL_ICONS["signal"],
        "tiles": [
            # The product-level SAM survey risk flag — a v27 driver.
            {"label": "Risk Flag", "column": "sam_risk_flag_product", "kind": "yesno", "direction": "bad_if_yes"},
        ],
    },
    # --- Clinical Inclination ------------------------------------------------
    {
        "title": "Clinical Focus",
        "icon": SIGNAL_ICONS["pulse"],
        "tiles": [
            {"label": "Clinical Bucket", "column": "clinical_bucket", "kind": "clinical_bucket", "direction": "clinical"},
            {
                "label": "Partnership Advocacy",
                "column": "bob_Prior Season HS Partnership Advocacy (L1BG+Brand)",
                "kind": "text",
                "direction": "advocacy",
            },
        ],
    },
    # Provider Margin row hidden per request (10/8): customer_provider_margin is
    # no longer a model input (dropped in the 10/7 model). Commented out, not
    # deleted, so it can be restored as context if wanted.
    # {
    #     "title": "Provider Margin",
    #     "icon": SIGNAL_ICONS["dollar"],
    #     "tiles": [
    #         # customer_provider_margin = cms_reimbursement - customer_list_price,
    #         # verified to 7e-15 against the source.
    #         {"label": "CMS Reimbursement", "column": "cms_reimbursement", "kind": "currency"},
    #         {"label": "Customer List Price", "column": "customer_list_price", "kind": "currency"},
    #     ],
    # },
    # Prebook Conversion Ratio row hidden (10/22, v27): not a model input since
    # 10/8, and the shown season has no current-year prebook data at all, so
    # its tiles were blank on every account. Commented out, not deleted.
    # {
    #     "title": "Prebook Conversion Ratio",
    #     "icon": SIGNAL_ICONS["refresh"],
    #     "tiles": [
    #         {"label": "Contract Customer Doses", "column": "contract_level_total_final_sales", "kind": "compact"},
    #         {"label": "Total Prebook", "column": "total_prebook_cy", "kind": "compact"},
    #     ],
    # },
]

# The Deep Dive's category view (user decision 11/5): the SIGNAL_GROUPS rows
# grouped under the same categories as the simulation tab's accordion —
# re-grouped 11/6 to the four-category client taxonomy (see
# DRIVER_CATEGORIES). A category shows its member rows unchanged, so every
# row's "i" tooltip and formula stays. v1 renders the picker as a segmented
# control (option N), v2 as a list + detail layout (option O) — the same
# CDE_DRIVER_CATEGORY_STYLE switch as the accordion.
SIGNAL_CATEGORIES = [
    ("Price Sensitivity", [
        "Price vs. Standard Benchmark Offer",
        "Year-over-year Price Change",
        "Price vs. Cheapest Competitor Price",
        "Negotiation Receptivity",
        "Buying Group",
    ]),
    ("Customer Loyalty", [
        "Share of Wallet",
        "Competitive Pressure",
        "Multi-year Contract",
    ]),
    ("Relationship with Customer", [
        "Relationship Strength",
        "Field Engagement Health",
        "SAM Risk Flag",
    ]),
    ("Clinical Inclination", ["Clinical Focus"]),
]

# SHAP's own diverging pair, so the waterfall reads like the canonical plot
# (flipped from SHAP's own default red=positive/blue=negative per request).
# Verified CVD-safe: normal-vision OKLab dE 37.3, protanopia 21.7,
# deuteranopia 28.7, tritanopia 35.0 — all above the 8.0 separation target.
SHAP_POSITIVE = "#1E88E5"
SHAP_NEGATIVE = "#FF0D57"
# The waterfall's start/end anchors (11/3): deep slate for the model's own
# average, purple for this account's prediction. Used by the emphasised
# reference lines, their labels and the shaded lift band between them.
SHAP_ANCHOR_BASE = "#3D3654"
SHAP_ANCHOR_FINAL = "#6B21A8"

# Driver categories (user decision 11/5): the 16 drivers grouped into five
# business buckets, shown as a ranked, expandable accordion between the
# product picker and the (now collapsible) waterfall. Two approved visual
# styles run side by side for comparison; CDE_DRIVER_CATEGORY_STYLE picks one
# per instance: "ranked" (v1, rank chips + share-of-impact bars, the default)
# or "meters" (v2, signal-strength meters).
DRIVER_CATEGORY_STYLE = os.environ.get("CDE_DRIVER_CATEGORY_STYLE", "ranked")
# Re-grouped 11/6 to the client taxonomy (the four-category driver table):
# Price Sensitivity / Customer Loyalty / Relationship with Customer / Clinical
# Inclination, with the user's placements for the drivers that table omits —
# buying group under Price Sensitivity, multi-year under Customer Loyalty,
# prior-year Customer doses under Relationship with Customer. season_cat has no
# bucket: it is removed from every SHAP surface and its value redistributed
# into the other drivers (REDISTRIBUTED_FEATURES in src/utils.py, 11/6), so
# no Season row ever reaches this accordion.
DRIVER_CATEGORIES = [
    ("Price Sensitivity", [
        "price_vs_std_benchmark_updated",
        "premium_gap_vs_min_competitor_pct",
        "yoy_price_change_pct",
        "negotiation_receptivity_score",
        "buying_group_category_cat",
    ]),
    ("Customer Loyalty", [
        "bob_Prior Season Shr of Wallet_enc",
        "flips_prior_2_seasons",
        "yoy_ms_change_pct_prior",
        "multi_year_flag",
    ]),
    ("Relationship with Customer", [
        "cde_engagement_length_log",
        "cde_net_sentiment_score",
        "sam_risk_flag_product",
        "log_prior_customer_doses",
    ]),
    ("Clinical Inclination", [
        "apt_clinical_bucket_enc",
        "bob_Prior Season HS Partnership Advocacy_enc",
    ]),
]
# Importance bands on this scenario's own probability-space contributions
# (account-specific, so they move with the levers): a driver by its |pp|
# move, a category by its drivers' summed |pp|.
DRIVER_IMPORTANCE_BANDS = (6.0, 2.0)  # High >= 6pp, Medium >= 2pp, else Low
CATEGORY_IMPORTANCE_BANDS = (15.0, 6.0)
IMPORTANCE_COLORS = {
    "High": ("#803C05", "#FFECE2"),
    "Medium": ("#B45309", "#FEF3E2"),
    "Low": ("#6E6662", "#EFEDF4"),
}

LEVER_CONFIGS = [
    {
        "label": "Dead Net Price",
        "key": "dead_net_price",
        "kind": "number",
        "value_type": "currency",
        "helper": "$/dose",
        "prev_column": PRICE_BASE_COLUMN,
        # Matches "format" (2 decimals): step used to be 0.25, a grid the
        # rounded min/max bounds below don't always land on (46.50 did,
        # 44.17 doesn't) — off-grid entries like 46.00 were then rejected as
        # "not a valid value" even though they're within min/max. 0.01 makes
        # every 2-decimal price in range valid, matching what is displayed.
        "step": 0.01,
        "format": "%.2f",
        # Explains the min/max guardrail itself, not a driver definition —
        # Streamlit's own "Value must be greater than or equal to X" message
        # (baked into its compiled frontend, not overridable from Python)
        # gives the bound but never the reason for it.
        "tooltip_text": (
            "Dead Net Price scenarios are limited to the historical price range observed for this product, keeping simulations realistic and grounded in comparable account behavior."
        ),
    },
    # The five non-price levers are v8 model features themselves (key = the
    # feature name), so each choice is scored directly. Options are the
    # inputs' real categories (src/utils.py LEVER_SPECS), which lets an
    # Account Manager correct an input the data infers — e.g. a health system
    # they know is clinically inclined — and see the prediction move.
    {
        "label": "Share of Wallet",
        "key": "bob_Prior Season Shr of Wallet_enc",
        "kind": "choice",
        "prev_column": None,
        "options": lever_options("bob_Prior Season Shr of Wallet_enc"),
        "tooltip_text": (
            "Prior-season share of the health system's flu wallet, from Book of Business: "
            "from Customer Loyalist (most of the wallet) to Competitive Loyalist."
        ),
    },
    {
        "label": "Partnership Advocacy",
        "key": "bob_Prior Season HS Partnership Advocacy_enc",
        "kind": "choice",
        "prev_column": None,
        "options": lever_options("bob_Prior Season HS Partnership Advocacy_enc"),
        "tooltip_text": (
            "The health system's prior-season partnership advocacy stage: "
            "Advocate, Operationalize, Prioritize or Choose."
        ),
    },
    {
        "label": "Engagement Score",
        "key": "cde_engagement_length_log",
        "kind": "choice",
        "prev_column": None,
        "options": lever_options("cde_engagement_length_log"),
        "tooltip_text": (
            "Call engagement score derived from call details (number of calls, topics discussed, "
            "seniority of attendees). Low / Medium / High are the typical scores of the bottom, "
            "middle and top third of accounts; No Data means no call data for this account."
        ),
    },
    {
        "label": "Negotiation Stance",
        "key": "negotiation_receptivity_score",
        "kind": "choice",
        "prev_column": None,
        "options": lever_options("negotiation_receptivity_score"),
        "tooltip_text": (
            "The account's recorded negotiation stance from PSC / APT contracting inputs. "
            "Change it if you know the account's current stance differs."
        ),
    },
    {
        "label": "Clinical Focus",
        "key": "apt_clinical_bucket_enc",
        "kind": "choice",
        "prev_column": None,
        "options": lever_options("apt_clinical_bucket_enc"),
        "tooltip_text": (
            "Whether the health system is clinically inclined or non-clinically inclined in its "
            "vaccine decisions, inferred from PSC / APT data. If you know the account better than the "
            "data does, change it here and see how the prediction moves."
        ),
    },
]


# --- cached loaders ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_metadata() -> dict[str, Any]:
    return load_model_metadata()


@st.cache_resource(show_spinner=False)
def get_models() -> dict[str, Any]:
    return load_models(get_metadata())


@st.cache_resource(show_spinner=False)
def get_fold_models() -> dict[str, Any]:
    return load_fold_models()


@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    return load_features()


@st.cache_data(show_spinner=False)
def get_history() -> pd.DataFrame:
    return load_history()


@st.cache_data(show_spinner=False)
def product_medians() -> dict[str, dict[str, float]]:
    """Per-product median for every numeric model feature (which includes
    every lever's feature), read by the technical narrative's %-vs-median
    figures.

    Covers the whole list rather than a hand-picked subset. A missing entry
    here only raises for accounts where that value happens to be null, so a
    hardcoded list fails on some accounts and not others, which is a nasty way
    to find out.
    """
    frame = get_features()
    names = list(NUMERIC_FEATURE_COLUMNS)
    return {
        str(product): {name: float(pd.to_numeric(subset[name], errors="coerce").median()) for name in names}
        for product, subset in frame.groupby(PRODUCT_COLUMN)
    }


# The raw columns behind the composite driver scores (per the modelling
# team's formula table), for the Account Context driver detail: a raw column
# is cited alongside a driver only when the account's own value for it is
# above this per-product median — the median IS the threshold, per the
# agreed rule, not a separately tuned constant.
RAW_DETAIL_COLUMNS = ("bob_L1BG Prior Season Customer Doses", "flips_prior_2_seasons", "share_change",
    "premium_%_vs_competitor1", "dead_net_price_per_dose", "standard_price_benchmark")


@st.cache_data(show_spinner=False)
def raw_column_medians() -> dict[str, dict[str, float]]:
    """Per-product median for the raw columns behind the composite driver
    scores — the threshold narrative_technical._raw_detail() compares each
    account's own value against."""
    frame = get_features()
    return {
        str(product): {
            name: float(subset[name].median()) for name in RAW_DETAIL_COLUMNS if name in subset.columns
        }
        for product, subset in frame.groupby(PRODUCT_COLUMN)
    }


# --- formatting -------------------------------------------------------------


def format_currency(value: float) -> str:
    return f"${float(value):,.2f}"


def format_score(value: float) -> str:
    return f"{float(value):.2f}"


def format_volume(value: float) -> str:
    return f"{int(value):,} doses"


def format_value(value: Any, value_type: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    if value_type == "currency":
        return format_currency(float(value))
    if value_type == "volume":
        return format_volume(float(value))
    if value_type == "flag":
        return "Yes" if float(value) >= 0.5 else "No"
    return format_score(float(value))


def slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


# Simulation views (user, 11/17): three independent copies of the Simulation
# screen, shown as tabs for now so the team can agree what each audience sees;
# later the signed-in user's role picks one. They differ only in the
# Predicted Outcome block:
#   full        - win probability %, its bar and the band pill (leadership)
#   directional - the band only (selected PSC reviewers)
#   intel       - no Predicted Outcome block at all (SAMs / DNAs, broader PSC)
# Each view keeps its own levers, result and pickers: view_key() namespaces
# every session_state key the screen uses. "full" keeps the original,
# unprefixed keys, so the default view behaves exactly as before.
# Demo build: only the full view, shown as "Simulation". The other two views'
# code paths stay in place; add them back here to show their tabs again.
SIMULATION_VIEWS: tuple[tuple[str, str], ...] = (
    ("full", "Simulation"),
)
# Customer Deep Dive tab, shown after Simulation. False hides it.
SHOW_CUSTOMER_DEEP_DIVE = True
_ACTIVE_VIEW = {"id": "full"}


def set_active_view(view: str) -> None:
    _ACTIVE_VIEW["id"] = view


def active_view() -> str:
    return _ACTIVE_VIEW["id"]


def view_key(name: str) -> str:
    view = _ACTIVE_VIEW["id"]
    return name if view == "full" else f"{view}__{name}"


def lever_state_key(product: str, lever_key: str) -> str:
    return view_key(f"proposed_{slugify(product)}_{lever_key}")


def keep_lever_state() -> None:
    """Keep every view's lever selections while another tab is open.

    Only the open simulation tab renders (lazy tabs), and Streamlit drops a
    keyed widget's value on any run that does not draw it — so switching tabs
    would reset the other views' levers. Re-assigning each value at the top of
    the run turns it into plain session state, which survives (Streamlit's
    documented workaround). Lever widgets take no default argument, so this
    raises no "default value and Session State" warning.
    """
    for key in [k for k in st.session_state if isinstance(k, str) and "proposed_" in k]:
        st.session_state[key] = st.session_state[key]


def safe_selectbox(
    container, label: str, options: list[Any], key: str, preferred: Any = None
) -> Any:
    """Selectbox that opens on ``preferred`` when it exists, else the first option."""
    if not options:
        st.error(f"No options are available for {label}.")
        st.stop()
    if st.session_state.get(key) not in options:
        st.session_state[key] = preferred if preferred in options else options[0]
    return container.selectbox(label, options, key=key)


def safe_multiselect(
    container, label: str, options: list[Any], default: list[Any], key: str, width: Any = "stretch"
) -> list[Any]:
    if not options:
        st.error(f"No options are available for {label}.")
        st.stop()
    valid_default = [value for value in default if value in options] or options[:1]
    current = st.session_state.get(key, valid_default)
    if not isinstance(current, list):
        current = valid_default
    selected = [value for value in current if value in options]
    st.session_state[key] = selected or valid_default
    return container.multiselect(label, options, key=key, width=width)


# --- css --------------------------------------------------------------------


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F8F7FB;
            --primary: #EC7200;
            --purple: #EC7200;
            --orange: #EC7200;
            --teal: #36A39B;
            --blue: #4285F4;
            --light-purple: #FBE8DD;
            --light-orange: #FFF3E8;
            --light-teal: #EAF8F6;
            --light-blue: #EAF2FF;
            --border: #EFE1DA;
            --text-primary: #1F1F29;
            --text-secondary: #716964;
            --green: #36A39B;
            --amber: #EC7200;
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
        /* Light-to-dark banner: brand purple on the left fading to near-black
           on the right, so the logo's own dark background sits
           flush instead of showing as a pasted-on box. */
        .app-header {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: .55rem 1.15rem; border-radius: 14px;
            background: linear-gradient(
                90deg, #B25502 0%, #EC7200 22%, #5F2B01 48%, #2F1100 72%, #060008 100%
            );
            box-shadow: 0 10px 26px rgba(46, 17, 0, .28); margin-bottom: 1rem;
            position: relative; overflow: hidden;
        }
        /* Logo and title left-aligned; the badges take the remaining space and
           sit hard right. */
        .app-header-side {flex: 1 1 0; display: flex; align-items: center; min-width: 0;}
        .app-header-side--right {justify-content: flex-end;}
        .app-header-brand {
            display: flex; align-items: center; gap: .65rem;
            flex: 0 0 auto; min-width: 0;
        }
        .app-header-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 34px; height: 34px; flex: none; border-radius: 10px;
            background: rgba(255, 255, 255, .16);
            border: 1px solid rgba(255, 255, 255, .26);
            font-size: 1.05rem; line-height: 1;
        }
        .app-title {
            font-size: 1.22rem; line-height: 1.15; font-weight: 800; margin: 0;
            color: #FFFFFF; letter-spacing: .005rem;
        }
        .header-badges {display: flex; gap: .4rem; flex-wrap: wrap; justify-content: flex-end;}
        .badge {
            display: inline-flex; align-items: center; border-radius: 999px; padding: .28rem .6rem;
            background: rgba(255, 255, 255, .14); color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, .3);
            font-size: .72rem; font-weight: 700; white-space: nowrap;
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
        /* Hover-only info tip: an "i" circle that reveals a panel. No chevron
           and no click, unlike st.popover. */
        /* No help cursor anywhere (11/16): its "?" beside the arrow read as stray. */
        .info-tip {position: relative; display: inline-flex; cursor: default; margin-top: .45rem;}
        /* The SHAP heading's info icon sits right after its text, not off in
           its own far-right column. */
        .shap-heading-row {display: flex; align-items: center; gap: .4rem;}
        .shap-heading-row .info-tip {margin-top: 0;}
        .info-tip-icon {
            width: 17px; height: 17px; border-radius: 999px;
            border: 1.5px solid var(--text-secondary); color: var(--text-secondary);
            display: inline-flex; align-items: center; justify-content: center;
            font-family: Georgia, "Times New Roman", serif; font-style: italic;
            font-weight: 700; font-size: .68rem; line-height: 1;
        }
        .info-tip:hover .info-tip-icon {
            border-color: var(--purple); color: var(--purple);
        }
        .info-tip-body {
            visibility: hidden; opacity: 0; transition: opacity .12s ease;
            position: absolute; z-index: 60; top: calc(100% + 8px); left: 0;
            width: 430px; max-width: 60vw;
            background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px;
            box-shadow: 0 6px 20px rgba(31,31,41,.14);
            padding: .7rem .85rem; font-size: .78rem; line-height: 1.55;
            color: var(--text-primary); font-weight: 400; text-transform: none;
            letter-spacing: normal;
        }
        .info-tip-body ul {margin: 0; padding-left: 1.05rem;}
        .info-tip-body li {margin-bottom: .35rem;}
        .info-tip-body li:last-child {margin-bottom: 0;}
        .info-tip:hover .info-tip-body {visibility: visible; opacity: 1;}
        /* Driver-tile variant: centred under a narrow tile rather than
           left-anchored, and no help cursor (that read as a stray "?" on
           hover, which was flagged and removed once already). */
        .info-tip-body--center {left: 50%; transform: translateX(-50%); width: 300px;}
        .signal-tile .info-tip, .signal-head .info-tip, .signal-row-formula .info-tip, .kpi-value-tip {cursor: default; margin-top: 0;}
        /* KPI-card hover tip: anchored under the value, not the icon-shaped
           default, and narrower so it doesn't dwarf a card's own width. */
        .kpi-value-tip {display: inline-flex;}
        /* .kpi-row-right sets white-space: nowrap for its own single-line
           layout; that inherits into the tooltip body and rendered the
           whole description as one unwrapped, overflowing line. */
        /* width: max-content sizes the box to its own short text (e.g. "8.6%
           decrease vs prior year") instead of always claiming a fixed 260px;
           max-width caps it once text is long enough to need wrapping. */
        .kpi-value-tip .info-tip-body {width: max-content; max-width: 260px; left: 0; top: calc(100% + 6px); white-space: normal;}
        /* Strategic Quadrant's description is the longest of these tips —
           1.5x the default width so it wraps to noticeably fewer lines. */
        /* max-width is set explicitly (not just width) because the general
           .kpi-value-tip rule above caps max-width at 260px — max-width
           always wins over a plain width when it's smaller, which is why
           bumping width alone never visibly grew this box. */
        .kpi-value-tip--quadrant .info-tip-body {width: 550px; max-width: 550px;}
        /* display: block puts the (purple) first sentence on its own line
           regardless of how much text would otherwise fit beside it at
           this width — the rest of the description always starts fresh on
           line 2, not wherever the text happens to wrap. */
        .kpi-value-tip--quadrant .info-tip-body strong {color: var(--purple); display: block; margin-bottom: .3rem;}
        /* Formula icon sits at the row's far right edge — the panel opens
           leftward from it (right: 0) so it doesn't overflow the viewport. */
        .signal-row-formula {flex: none; display: flex; align-items: center; margin-left: .6rem;}
        .info-tip-body--formula {left: auto; right: 0; width: 320px;}
        /* Hover-preview + click-to-pin variant (the driver "i" icons):
           hovering still shows the panel via the normal .info-tip:hover
           rule above, but a hidden checkbox additionally lets a click pin
           it open so it survives the mouse leaving — the checked rule below
           keeps it visible independently of :hover. The backdrop label is a
           full-viewport click target behind the panel that unpins/closes it
           on any outside click; the icon label toggles the same checkbox,
           so clicking the icon again also closes it. */
        .info-tip--click .info-tip-toggle {position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;}
        .info-tip--click .info-tip-backdrop {display: none;}
        .info-tip--click .info-tip-toggle:checked ~ .info-tip-backdrop {
            display: block; position: fixed; inset: 0; z-index: 59; cursor: default;
        }
        .info-tip--click .info-tip-toggle:checked ~ .info-tip-body {visibility: visible; opacity: 1;}
        .info-tip--click .info-tip-icon {cursor: pointer;}
        .filter-label {
            font-size: .72rem; font-weight: 900; color: var(--text-primary);
            text-transform: uppercase; letter-spacing: .03rem; margin-bottom: .25rem;
        }
        .metric-card {
            border: 1px solid var(--border); border-radius: 14px; padding: .86rem .92rem;
            /* Tall enough for three product rows plus the label and footnote
               without clipping. */
            background: #FFFFFF; height: 172px; box-shadow: 0 1px 2px rgba(31,31,41,.04);
            display: flex; flex-direction: column;
        }
        .metric-card-green {border-left: 5px solid var(--teal); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-teal) 100%);}
        .metric-card-amber {border-left: 5px solid var(--orange); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-orange) 100%);}
        .metric-card-red {border-left: 5px solid var(--red); background: linear-gradient(180deg, #FFFFFF 0%, #FFF1F2 100%);}
        .metric-card-blue {border-left: 5px solid var(--blue); background: linear-gradient(180deg, #FFFFFF 0%, var(--light-blue) 100%);}
        .metric-label {font-size: .76rem; color: var(--text-secondary); font-weight: 760; text-transform: uppercase;}
        .metric-help {font-size: .78rem; color: var(--text-secondary); margin-top: .22rem;}
        .current-card {
            background: #FFFFFF; border: 1px solid #DDD4CF; border-radius: 11px;
            padding: 0 .75rem; min-height: 42px;
            display: flex; align-items: center;
            font-size: 1rem; font-weight: 780; color: var(--text-primary);
            box-shadow: none;
        }
        .current-value {font-size: 1rem; font-weight: 780; color: var(--text-primary);}
        /* Address field (11/12): plain text subtitle under the Health
           System selectbox, not a boxed field of its own — a bordered box
           next to two real dropdowns read as a third selectable field,
           which it isn't. Pin icon + text; shown in full, no truncation —
           wraps onto a second line rather than clipping with an ellipsis. */
        .current-card-address {
            display: flex; align-items: flex-start; gap: .4rem; margin: -.6rem 0 .2rem; padding-left: .5rem;
            font-weight: 400; font-size: .92rem; color: var(--text-secondary);
        }
        .current-card-address svg { flex: none; margin-top: .15rem; color: var(--purple); }
        .current-card-address span { white-space: nowrap; }
        div[data-testid="stColumn"]:has(.current-card-address),
        div[data-testid="stColumn"] div:has(> .current-card-address),
        div[data-testid="stColumn"] div:has(.current-card-address):not([data-testid="stColumn"]) {
            overflow: visible !important;
        }
        .stacked-bar {
            display: flex; width: 100%; height: 28px; border-radius: 999px; overflow: hidden;
            background: #F5E9E3; border: 1px solid var(--border);
        }
        /* Per-product rows inside the Win / Loss / Confidence cards, and one
           win-probability bar per product beneath them. */
        /* One shared row style across Win, Loss and Confidence so the product
           labels and values line up horizontally from card to card. */
        .outcome-row {
            display: flex; align-items: baseline; justify-content: space-between;
            gap: .4rem; margin-bottom: .42rem; line-height: 1.25;
        }
        .outcome-row:last-child {margin-bottom: 0;}
        .outcome-name {
            font-size: .72rem; font-weight: 800; color: var(--text-secondary);
            text-transform: uppercase; white-space: nowrap;
        }
        .outcome-value {
            font-size: .95rem; font-weight: 800; color: var(--text-primary);
            white-space: nowrap; line-height: 1.25;
        }
        /* One card per product: the name on its own line, then the outcome
           band (Comfortable / Watch / At Risk) with a one-line meaning, then
           the confidence row. No percentage or track (removed 11/11). */
        .outcome-card {
            border: 1px solid var(--border); border-radius: 14px; padding: .9rem 1rem;
            background: #FFFFFF; box-shadow: 0 1px 2px rgba(31,31,41,.04);
            display: flex; flex-direction: column; height: 100%;
        }
        .outcome-card-name {
            font-size: 1rem; font-weight: 800; color: var(--text-primary);
            margin-bottom: .5rem;
        }
        .outcome-band {
            display: inline-flex; align-items: center; gap: .5rem; align-self: flex-start;
            border-radius: 999px; padding: .32rem .85rem; white-space: nowrap;
            font-size: 1.05rem; font-weight: 800; letter-spacing: .02em; line-height: 1.2;
        }
        .outcome-band-dot {width: 10px; height: 10px; border-radius: 999px; background: currentColor; flex: none;}
        .outcome-band-comfortable {background: #E6F6EC; color: #15803D;}
        .outcome-band-watch {background: #FFF1E5; color: #EA580C;}
        .outcome-band-risk {background: #FDE8E8; color: #DC2626;}
        /* Empty spacer (11/12): kept purely for the gap it preserves between
           the band and the confidence row below — the "Over 20 pts below
           threshold" style caption text it used to hold was removed. */
        .outcome-card-caption {margin: .45rem 0 .6rem;}
        /* Full Simulation view only (11/17): the pre-11/11 card — percentage
           with the band as a small pill, then a filled win-probability bar. */
        .outcome-card-head {
            display: flex; align-items: center; justify-content: space-between;
            gap: .4rem; margin-bottom: .55rem;
        }
        .outcome-card-percent {font-size: 1.7rem; font-weight: 800; color: var(--text-primary); line-height: 1.1;}
        .outcome-pill {
            display: inline-flex; align-items: center; border-radius: 999px; white-space: nowrap;
            padding: .18rem .6rem; font-size: .68rem; font-weight: 800;
        }
        .outcome-pill-win {background: var(--light-teal); color: var(--teal);}
        .outcome-pill-watch {background: #FEF3E2; color: #B45309;}
        .outcome-pill-loss {background: #FDE8E8; color: var(--red);}
        .outcome-card-track {
            width: 100%; height: 8px; border-radius: 999px; background: var(--border);
            margin: 0 0 .45rem 0; overflow: hidden;
        }
        .outcome-card-fill {height: 100%; border-radius: 999px;}
        .outcome-card-fill-win {background: var(--teal);}
        .outcome-card-fill-watch {background: #D97706;}
        .outcome-card-fill-loss {background: var(--red);}
        .outcome-card-caption-text {font-size: .78rem; color: var(--text-secondary); margin: 0 0 .6rem;}
        .outcome-confidence-row {
            display: flex; align-items: center; justify-content: space-between;
            margin-top: auto; padding-top: .55rem; border-top: 1px solid var(--border);
        }
        .outcome-confidence-left {display: flex; align-items: center; gap: .5rem;}
        .outcome-confidence-icon {
            display: inline-flex; align-items: center; justify-content: center;
            width: 26px; height: 26px; border-radius: 50%;
            background: var(--light-teal); color: var(--teal); flex: none;
        }
        .outcome-confidence-label {font-size: .82rem; font-weight: 700; color: var(--text-secondary);}
        .outcome-confidence-right {display: flex; align-items: center; gap: .45rem;}
        .outcome-confidence-value {font-size: .92rem; font-weight: 800;}
        /* Hover card on the confidence row (11/16): how the score is built. */
        .outcome-confidence-row {position: relative;}
        .outcome-confidence-info {
            display: inline-flex; align-items: center; justify-content: center;
            width: 15px; height: 15px; border-radius: 50%; border: 1.5px solid currentColor;
            font-size: .62rem; font-weight: 800; font-style: normal; color: var(--text-secondary);
        }
        .confidence-tip {
            visibility: hidden; opacity: 0; transition: opacity .12s ease-out;
            position: absolute; z-index: 1000; top: calc(100% + 6px); left: 0;
            width: min(400px, 86vw); padding: .7rem .85rem;
            background: #FFFFFF; border: 1px solid var(--border); border-radius: 12px;
            box-shadow: 0 8px 24px rgba(31,31,41,.16);
            font-size: .8rem; line-height: 1.5; color: var(--text-primary); cursor: default;
        }
        .outcome-confidence-row:hover .confidence-tip {visibility: visible; opacity: 1;}
        .confidence-points {margin: 0; padding-left: 1.1rem;}
        .confidence-points li {margin: 0 0 .35rem;}
        .confidence-points li:last-child {margin-bottom: 0;}
        /* The last product card sits at the page's right edge: open leftwards. */
        .outcome-confidence-row.tip-right .confidence-tip {left: auto; right: 0;}
        /* Top-2-driver mini card, one per product, next to the narrative.
           Fixed height, not 100%: tall enough for a two-line driver name
           like "Doses-to-prebook ratio" so all three match regardless of
           content length. */
        .drivers-card {
            border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.1rem;
            background: #FFFFFF; box-shadow: 0 1px 2px rgba(31,41,41,.04);
            height: 160px; box-sizing: border-box; overflow: hidden;
        }
        .drivers-card-head {display: flex; align-items: center; gap: .5rem; margin-bottom: .9rem;}
        .drivers-dot {width: 8px; height: 8px; border-radius: 999px; flex: none;}
        .drivers-dot-win {background: var(--teal);}
        .drivers-dot-loss {background: var(--red);}
        .drivers-card-name {font-size: 1rem; font-weight: 800; color: var(--text-primary);}
        .drivers-row {
            display: flex; align-items: baseline; justify-content: space-between;
            gap: .5rem; margin-bottom: .85rem;
        }
        .drivers-row:last-child {margin-bottom: 0;}
        .drivers-row-left {display: flex; align-items: baseline; gap: .4rem; min-width: 0;}
        .drivers-num {font-size: .74rem; font-weight: 800; color: var(--text-secondary); flex: none;}
        .drivers-name {font-size: .78rem; font-weight: 700; color: var(--text-primary);}
        .drivers-pill {
            display: inline-flex; align-items: center; border-radius: 999px;
            padding: .1rem .5rem; font-size: .66rem; font-weight: 800; flex: none;
        }
        .drivers-pill-positive {background: var(--light-teal); color: var(--teal);}
        .drivers-pill-negative {background: #FDE8E8; color: var(--red);}
        .bar-row {display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem;}
        .bar-row-name {
            font-size: .72rem; font-weight: 800; color: var(--text-secondary);
            text-transform: uppercase; white-space: nowrap; width: 74px; flex: none;
        }
        .bar-row-track {flex: 1 1 auto;}
        .bar-row-value {font-size: .76rem; font-weight: 800; white-space: nowrap; width: 118px; flex: none; text-align: right;}
        /* One row per model driver: name, a range track with a marker per
           product, then the raw columns the feature is derived from. */
        /* The first product chip rendered flush against the input's rounded
           border and lost its first character ("LU HD").

           The left offset belongs on the value container, not on the chips: a
           margin big enough to clear the border applies to every chip and shows
           up as a wide gap between them. So the container is padded once, and
           the chips keep only the small margin that separates them.

           padding-left needs !important — BaseWeb sets its own 5px inline and
           the neighbouring input rules below are themselves !important.
           12px clears the 11px border-radius plus the 1px border.

           Wrapping is allowed so a fourth chip drops to a second line instead
           of overflowing. */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
            margin-left: 3px !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
            flex-wrap: wrap;
            row-gap: .2rem;
            padding-left: 12px !important;
        }
        /* Sign-in screen styling lives in _render_login_screen(), injected only
           on that screen so it cannot reach any other surface. */

        /* --- account signals --------------------------------------------- */
        /* Palette from the supplied design: teal values, purple category
           names, grey captions. No column-divider lines — spacing alone
           (equal gaps, equal tile widths) carries the table structure. */
        /* Column widths shared between the header row and every signal row,
           so the two line up: SIGNAL | DRIVER VALUE | UNDERLYING FEATURES. */
        .signal-table-header {
            display: flex; gap: 1rem; padding: 0 .9rem; margin-top: 1.1rem; margin-bottom: .4rem;
            font-size: .68rem; font-weight: 800; color: #9B938F;
            text-transform: uppercase; letter-spacing: .03rem;
            align-items: center;
        }
        /* Left-aligned since the per-row icon badges were removed (user,
           11/5) — the names now start at the row's left edge. */
        .signal-table-header-signal {width: 224px; flex: none; text-align: left;}
        .signal-table-header-features {
            flex: 1 1 auto; display: flex; min-width: 0;
        }
        .signal-table-header-features span {flex: 0 0 25%; text-align: center;}
        .signal-table-header-formula {width: 24px; flex: none; margin-left: .6rem;}
        .signal-row {
            display: flex; align-items: stretch; gap: 1rem;
            background: #FFFFFF; border: 1px solid #F4E8E1; border-radius: 12px;
            padding: .7rem .9rem; margin-bottom: .55rem;
            box-shadow: 0 1px 2px rgba(31, 31, 41, .04);
        }
        .signal-head {
            display: flex; align-items: center; gap: .55rem;
            width: 224px; flex: none;
        }
        .signal-head-icon {
            display: inline-flex; align-items: center; justify-content: center;
            color: #A74F03; flex: none; width: 34px; height: 34px;
            border-radius: 999px; background: #FFECE1; border: 1px solid #FEC6A5;
        }
        .signal-head-icon svg {width: 18px; height: 18px;}
        .signal-head-text-wrap {display: flex; flex-direction: column; min-width: 0;}
        .signal-head-text {
            font-size: .88rem; font-weight: 800; color: var(--primary); line-height: 1.15;
        }
        .signal-head-subtitle {
            font-size: .7rem; color: #9B938F; line-height: 1.2; margin-top: .1rem;
        }
        /* Fixed 4-column grid for the raw-column tiles, same tile width on
           every row regardless of how many tiles that row has — a row with
           fewer tiles fills the first columns left-aligned and leaves the
           rest empty, rather than stretching to fill the width (that uneven
           stretching was the earlier complaint). */
        .signal-tiles {display: flex; flex: 1 1 auto; min-width: 0;}
        /* Fixed tile height so all rows are the same depth whatever their
           labels wrap to — the table look depends on it. */
        .signal-tile {
            flex: 0 0 25%; min-width: 0; text-align: center; padding: 0 .5rem;
            display: flex; flex-direction: column; justify-content: center;
            min-height: 46px;
        }
        .signal-value {
            font-size: 1.02rem; font-weight: 800; color: #10B3A6; line-height: 1.15;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .signal-value--empty {color: #BEB5B0;}
        /* Direction colours (user request 11/3): the SHAP pair, so a signal
           value reads like the waterfall — blue helps the win, red hurts it.
           Undirected values keep the teal above. */
        .signal-value--positive {color: #1E88E5;}
        .signal-value--negative {color: #FF0D57;}
        .signal-label {
            font-size: .68rem; color: #6E6662; margin-top: .22rem; line-height: 1.2;
        }
        /* Mini range bar under a numeric tile's value (user request 11/11):
           the brand's lowest-to-highest value this season and this account's
           dot, so a bare number like 10.37 has a scale. (A median tick was
           removed by request, 11/11.) */
        .range-bar {width: 82%; max-width: 150px; margin: .35rem auto 0;}
        .range-track {position: relative; height: 4px; border-radius: 999px; background: #F5E6DD;}
        .range-dot {
            position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 999px;
            background: #10B3A6; border: 2px solid #FFFFFF; box-shadow: 0 0 0 1px #10B3A6;
            transform: translate(-50%, -50%);
        }
        .range-ends {
            display: flex; justify-content: space-between; margin-top: .22rem;
            font-size: .6rem; color: #9B938F; line-height: 1;
        }
        /* The drivers export button (11/11), right-aligned in the Account
           signals header and the same size as the Simulation tab's report
           button. */
        .st-key-drivers_download {max-width: 78px; margin-left: auto;}
        /* Streamlit's markdown container has a -1rem bottom margin, which
           made the centred header row sit the "Account signals" title 8px
           below the brand picker and the download button. */
        [data-testid="stMarkdownContainer"]:has(> .signal-section-title) {margin-bottom: 0;}
        .signal-range-note {font-size: .72rem; color: var(--text-secondary); margin: .3rem 0 .6rem;}
        .signal-section-title {
            display: flex; align-items: center; gap: .85rem;
        }
        .signal-section-copy {display: flex; flex-direction: column; gap: .2rem;}
        .signal-section-copy .subsection-title {font-size: 1.05rem; margin: 0;}
        .signal-section-subtitle {
            font-size: .82rem; color: var(--text-secondary); line-height: 1.25;
        }
        .signal-section-icon {
            display: inline-flex; align-items: center; justify-content: center;
            width: 42px; height: 42px; border-radius: 999px;
            color: #A74F03; background: #FFE6D8; border: 1px solid #FEB88F;
            flex: none;
        }
        .signal-section-icon svg {width: 21px; height: 21px;}
        /* Left-align the brand toggle so it sits just after the "Account
           signals" heading (with the column gap as its only spacing) instead
           of being pushed to the far right, where it was easy to miss.
           Streamlit adds st-key-<key> to a keyed widget's container, which is
           stabler than the generated testids. The outer container alone is
           not enough: the widget renders a chain of full-width block
           wrappers, so it has to be pinned left at each level. */
        .st-key-signal_product,
        .st-key-signal_product > div,
        .st-key-signal_product > div > div {
            display: flex !important;
            justify-content: flex-start !important;
            width: 100% !important;
        }
        .st-key-signal_product [role="radiogroup"],
        .st-key-signal_product [data-baseweb="button-group"] {
            margin-right: auto !important;
            justify-content: flex-start !important;
            padding-top: 6px !important;
            padding-left: 80px !important;
        }
        /* Each brand pill's own width (the pill sizes to its label + this
           padding — bump these two values further for even more width) and
           push the whole group down so the toggle's row lines up with the
           "Account signals" text's vertical centre instead of sitting above it. */
        .st-key-signal_product [data-baseweb="button-group"] button {
            /* 1.6rem, not 2.8rem: sized for full brand names, not HD/STD/FLK. */
            padding-left: 1.6rem !important;
            padding-right: 1.6rem !important;
        }
        /* Account signals framed panel (S1, user decision 11/5, v1 only — the
           keyed container renders only on the "ranked" style): one bordered
           card holds the header row, the category tab strip and the rows, so
           the section no longer floats between its neighbours. */
        .st-key-signals_panel {
            background: #FFFFFF; border: 1px solid #F0E1D9; border-radius: 14px;
            padding: .85rem 1.1rem .65rem;
            box-shadow: 0 1px 3px rgba(45, 38, 35, .05);
        }
        .signals-panel-divider {height: 1px; background: #F0E1D9; margin: .55rem -1.1rem .45rem;}
        /* The category picker as an attached tab strip: joined square tabs,
           the active one white with a purple top bar (buttons carry
           aria-checked). */
        .st-key-signal_category [data-testid="stButtonGroup"] {gap: 0 !important;}
        .st-key-signal_category button {
            border-radius: 0 !important; border: none !important;
            border-top: 3px solid transparent !important;
            border-right: 1px solid #F0E1D9 !important;
            background: #F6F4FA !important; color: #4D4642 !important;
            padding: .6rem 1.45rem !important; margin: 0 !important;
        }
        /* Tab labels sized up so the categories read as real navigation
           (user, 11/5: "make the tabs a bit more evident"). */
        .st-key-signal_category button p {
            font-size: .95rem !important; font-weight: 600 !important;
            letter-spacing: .01em;
        }
        .st-key-signal_category button:last-child {border-right: none !important;}
        .st-key-signal_category button:hover {background: #F6EBE4 !important;}
        /* The selected tab. Two selectors because the marker moved between
           Streamlit builds: 1.58 (the image's pin — what dev runs) marks it
           kind="segmented_controlActive" and has NO aria-checked; later
           builds (1.60, the local env) carry aria-checked="true". Targeting
           only aria-checked left dev's selected tab unhighlighted (11/6). */
        .st-key-signal_category button[aria-checked="true"],
        .st-key-signal_category button[kind="segmented_controlActive"] {
            background: #FFFFFF !important; color: #803C05 !important;
            border-top: 3px solid #803C05 !important;
            box-shadow: 0 1px 0 #FFFFFF;
        }
        .st-key-signal_category button[aria-checked="true"] p,
        .st-key-signal_category button[kind="segmented_controlActive"] p {
            font-weight: 800 !important;
        }
        /* Shrink the heading column to its own content width (icon + "Account
           signals" text, no subtitle) instead of the fixed fraction Streamlit
           gives it, so the picker column right after it starts immediately
           past the text — no leftover reserved whitespace in between. */
        div[data-testid="stHorizontalBlock"]:has(> div .signal-section-title)
            > div:first-child {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        /* Section headings below the page title: same weight and colour as
           "Customer Deep Dive", a step down in size. */
        .subsection-title {
            font-size: .98rem; font-weight: 800; color: var(--text-primary);
            margin: 0 0 .45rem 0;
        }
        .driver-row {
            display: flex; align-items: center; gap: .8rem;
            padding: .45rem 0 .95rem 0; border-bottom: 1px solid #F6ECE6;
        }
        .driver-row:last-child {border-bottom: none;}
        /* Left: the feature name only. Right of the bar: its derivation. */
        .driver-name {
            width: 190px; flex: none;
            font-size: .8rem; font-weight: 800; color: var(--text-primary);
            line-height: 1.25;
        }
        /* Single-hue sequential track, light (low) to dark (high). One hue
           carries magnitude without implying good or bad, which a red-to-green
           ramp did — for competitive intensity a high value is not favourable. */
        .driver-track {
            position: relative; flex: 1 1 auto; height: 34px; border-radius: 5px;
            border: 1px solid #BEB5B0;
            /* Mid stop kept at #7BAFE2 rather than darker: below that the grey
               marker fill and its white rim both fall under 3:1 there. */
            background: linear-gradient(90deg,
                #EAF2FC 0%, #B9D3F0 38%, #7BAFE2 72%, #14508F 100%);
        }
        /* Straddles the bottom edge: half inside the bar, half below it. An SVG
           rather than a CSS border triangle so it can carry a white rim — grey
           fill alone drops to 1.3:1 contrast against the dark end of the ramp,
           while the rim keeps it visible at both ends. */
        .driver-marker {
            position: absolute; bottom: -12px; width: 22px; height: 24px;
            transform: translateX(-11px); line-height: 0;
        }
        /* Values sit inside the track, right-aligned on one line. */
        .driver-inline {
            position: absolute; right: .5rem; top: 50%; transform: translateY(-50%);
            font-size: .68rem; font-weight: 800; white-space: nowrap;
            color: #FFFFFF; text-shadow: 0 1px 2px rgba(0,0,0,.55);
        }
        .driver-inline span {margin-left: .5rem;}
        /* How the selected driver is calculated, above its charts. */
        .driver-definition {
            border: 1px solid var(--border); border-left: 4px solid var(--purple);
            border-radius: 10px; padding: .55rem .7rem; background: #FFFFFF;
            margin: .1rem 0 .8rem 0;
        }
        .driver-definition-title {font-size: .8rem; font-weight: 800; color: var(--text-primary);}
        .driver-definition-formula {font-size: .74rem; color: var(--text-secondary); margin-top: .12rem;}
        .driver-definition-source {font-size: .7rem; color: var(--text-secondary); margin-top: .2rem; line-height: 1.5;}
        .driver-definition-source code {font-size: .68rem; background: #FAF0EA; padding: 0 .18rem; border-radius: 3px;}
        .driver-unverified {color: var(--red); font-weight: 800;}
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
        /* Sub-heading and divider inside the narrative card: drivers first,
           then the description. */
        .narrative-sub {
            font-size: .72rem; font-weight: 800; color: var(--text-secondary);
            text-transform: uppercase; letter-spacing: .03rem; margin-bottom: .4rem;
        }
        .narrative-rule {
            border-top: 1px solid #F6ECE6; margin: .75rem 0 .6rem 0;
        }
        /* Confidence Summary card (below the Narrative Summary). */
        .confidence-head {display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-bottom: .45rem;}
        .confidence-score {font-size: 1.55rem; font-weight: 800; line-height: 1;}
        .confidence-score-max {font-size: .8rem; font-weight: 700; color: var(--text-secondary); margin-left: .1rem;}
        .confidence-band {
            font-size: .7rem; font-weight: 800; text-transform: uppercase; letter-spacing: .03rem;
            border: 1px solid; border-radius: 999px; padding: .12rem .55rem;
        }
        .confidence-headline {font-weight: 700; color: var(--text-primary);}
        .confidence-factor {margin: 0 0 .65rem 0;}
        .confidence-factor-top {display: flex; justify-content: space-between; align-items: baseline; gap: .6rem;}
        .confidence-factor-name {font-size: .84rem; font-weight: 700; color: var(--text-primary);}
        .confidence-factor-weight {font-size: .72rem; font-weight: 600; color: var(--text-secondary); margin-left: .45rem;}
        .confidence-factor-value {font-size: .84rem; font-weight: 800; color: var(--text-primary);}
        .confidence-factor-track {height: 6px; border-radius: 999px; background: #F6ECE6; margin: .25rem 0 .3rem 0; overflow: hidden;}
        .confidence-factor-fill {height: 100%; border-radius: 999px; background: var(--purple);}
        .confidence-factor-detail {font-size: .8rem; color: var(--text-secondary); line-height: 1.45;}
        .confidence-caveat {
            font-size: .74rem; color: var(--text-secondary); border-top: 1px solid #F6ECE6;
            padding-top: .55rem; margin-top: .2rem; line-height: 1.45;
        }
        /* Customer Deep Dive KPI cards */
        .kpi-card {
            border: 1px solid var(--border); border-radius: 14px;
            padding: .8rem .9rem; background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(31,31,41,.04);
            /* Fixed height, not min-height, so all five cards are identical
               regardless of how many products are selected. */
            height: 165px; display: flex; flex-direction: column;
            overflow: hidden;
        }
        /* The card's own overflow:hidden (kept for uniform card height) was
           clipping the hover tooltip box at the card's edge, cutting its
           text off mid-line instead of letting it float above the
           neighbouring card. Lifted only while a tooltip inside is actually
           open. */
        .kpi-card:has(.kpi-value-tip:hover) {overflow: visible; position: relative; z-index: 60;}
        .kpi-card--purple {border-left: 5px solid var(--purple); background: linear-gradient(180deg,#FFFFFF 0%,var(--light-purple) 100%);}
        .kpi-card--teal   {border-left: 5px solid var(--teal);   background: linear-gradient(180deg,#FFFFFF 0%,var(--light-teal) 100%);}
        .kpi-card--blue   {border-left: 5px solid var(--blue);   background: linear-gradient(180deg,#FFFFFF 0%,var(--light-blue) 100%);}
        .kpi-card--orange {border-left: 5px solid var(--orange); background: linear-gradient(180deg,#FFFFFF 0%,var(--light-orange) 100%);}
        /* Icon, heading and product name all pick up the card's own accent
           color instead of a flat grey, so each card reads as one unit. */
        .kpi-card--purple .kpi-label-icon, .kpi-card--purple .kpi-label {color: var(--purple);}
        .kpi-card--teal   .kpi-label-icon, .kpi-card--teal   .kpi-label {color: var(--teal);}
        .kpi-card--blue   .kpi-label-icon, .kpi-card--blue   .kpi-label {color: var(--blue);}
        .kpi-card--orange .kpi-label-icon, .kpi-card--orange .kpi-label {color: var(--orange);}
        .kpi-label-row {display: flex; align-items: center; gap: .4rem; margin-bottom: .45rem;}
        .kpi-label-icon {display: inline-flex; flex: none;}
        .kpi-label {
            font-size: .92rem; font-weight: 800;
            text-transform: uppercase; letter-spacing: .03rem;
        }
        .kpi-value {
            font-size: 1.55rem; font-weight: 820; color: var(--text-primary);
            line-height: 1.15;
        }
        .kpi-value--text {font-size: 1.05rem; line-height: 1.3;}
        .kpi-delta {font-size: .74rem; font-weight: 700; margin-top: .1rem;}
        .kpi-delta--up   {color: #19766F;}
        .kpi-delta--down {color: var(--red);}
        .kpi-delta--flat {color: #918984;}
        /* One line per product: name on the left, price and its YoY together on
           the right, so the price card stays the same height as the others. */
        .kpi-row {
            display: flex; align-items: baseline; justify-content: space-between;
            gap: .35rem; margin-bottom: .34rem;
        }
        .kpi-row:last-child {margin-bottom: 0;}
        .kpi-row-name {
            font-size: .8rem; font-weight: 800; color: var(--text-primary);
            text-transform: uppercase; white-space: nowrap;
        }
        .kpi-row-right {display: flex; align-items: baseline; gap: .3rem; white-space: nowrap;}
        .kpi-row-value {font-size: .8rem; font-weight: 800; color: var(--text-primary);}
        .kpi-delta-inline {font-size: .7rem; font-weight: 800; white-space: nowrap;}
        .kpi-foot {font-size: .72rem; color: var(--text-secondary); margin-top: auto; padding-top: .4rem;}
        /* Coloured chip for the four account-posture buckets. */
        .bucket-pill {
            font-size: .68rem; font-weight: 800; color: #FFFFFF;
            padding: .1rem .45rem; border-radius: 999px;
            text-transform: capitalize; white-space: nowrap;
        }
        .lever-table-header {
            display: flex; padding: .3rem .1rem; margin-bottom: .2rem;
            border-bottom: 2px solid var(--border);
        }
        .lever-table-header-cell {
            font-size: .78rem; font-weight: 800; color: var(--text-primary);
            text-transform: none;
        }
        div[class*="st-key-lever-row-even-"],
        div[class*="st-key-lever-row-odd-"] {
            border-radius: 12px; margin-bottom: .3rem;
            padding: .5rem .6rem .55rem .6rem;
            /* The container is the flex column holding both season rows. Its
               default gap pushed the Previous Season row past the padding, so
               it rendered outside the tinted box. */
            gap: .1rem !important;
        }
        div[class*="st-key-lever-row-even-"] {background: #FAF9FC;}
        div[class*="st-key-lever-row-odd-"]  {background: #FFFFFF;}
        /* Lever dropdowns: let the selected option wrap onto a second line
           (slightly smaller) rather than clip to "Customer…" / "Priorit…" —
           the product columns are too narrow for "Competitive Loyalist" or
           "Non-Clinically Inclined" on one line. Scoped to the lever rows. */
        /* The value column (first child of the control) shrinks and wraps;
           the chevron (its sibling) keeps its own width, so it stays inside
           the box. */
        div[class*="st-key-lever-row-"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            padding-left: .4rem !important;
            padding-right: 0 !important;
        }
        div[class*="st-key-lever-row-"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child div {
            white-space: normal !important;
            text-overflow: clip !important;
            height: auto !important;
            max-width: 100% !important;
            font-size: .76rem !important;
            line-height: 1.15 !important;
            word-break: normal !important;
            /* Breaks a word only when it alone is wider than the box (very
               narrow screens) — clipping it would hide the value entirely. */
            overflow-wrap: break-word !important;
        }
        div[class*="st-key-lever-row-"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child {
            flex: 0 0 auto !important;
            padding-left: 0 !important;
            padding-right: .2rem !important;
        }
        .lever-table-metric {
            font-size: .9rem; font-weight: 800; color: var(--text-primary);
            padding-top: .45rem; line-height: 1.25;
        }
        /* Icon sits right after the lever name, not stacked under it. */
        .lever-table-metric--with-tip {display: flex; align-items: center; gap: .4rem;}
        .lever-table-metric--with-tip .info-tip {margin-top: 0; cursor: default;}
        .lever-table-season-label {
            font-size: .74rem; font-weight: 800; color: var(--text-primary);
            padding-top: .5rem; line-height: 1.2;
        }
        .lever-table-season-label--prev {
            font-size: .72rem; font-weight: 700; color: #918984;
            padding-top: .4rem; white-space: nowrap;
        }
        .lever-table-prev-value {
            font-size: .88rem; font-style: italic; color: #918984;
            padding-top: .4rem; padding-left: .85rem;
        }
        div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-lever-sticky-panel"]) {
            height: 100% !important;
        }
        div[class*="st-key-lever-sticky-panel"] {
            position: sticky;
            top: 1rem;
            flex: 0 0 auto !important;
        }
        label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important; font-weight: 700 !important;
        }
        /* :first-child on the multiselect, not every > div. BaseWeb renders more
           than one direct child inside the select, and painting them all white
           with a border and a 42px min-height put an opaque rounded box over the
           left of the first product chip — the "white box above the product". */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stMultiSelect"] div[role="combobox"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="input"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DDD4CF !important;
            border-radius: 11px !important;
            box-shadow: none !important;
            min-height: 42px !important;
        }
        /* The typing input inside the multiselect must not carry the blanket
           white background set for inputs below, or it paints over the chips. */
        [data-testid="stMultiSelect"] input {
            background-color: transparent !important;
            min-height: 0 !important;
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
            color: #918984 !important;
            -webkit-text-fill-color: #918984 !important;
        }
        [data-testid="stNumberInput"] input {
            background-color: #FFFFFF !important;
            font-weight: 760 !important;
            min-height: 42px !important;
            text-align: left !important;
            padding-left: .85rem !important;
        }
        /* The selected value sits in the first child of the combobox row; the
           chevron icon is a separate sibling, so centering this alone leaves
           the icon where it was. */
        [data-testid="stSelectbox"] div[role="combobox"] > div:first-child {
            text-align: center !important;
            width: 100%;
        }
        /* The SHAP product picker sits beside the Log-odds/Probability
           segmented control and should match its shorter native height,
           rather than the 42px every other selectbox in the app uses. */
        div[class*="st-key-shap-product-picker"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[class*="st-key-shap-product-picker"] [data-testid="stSelectbox"] div[role="combobox"] {
            min-height: 27px !important;
        }
        /* Right-align the Log-odds/Probability toggle within its column,
           rather than narrowing the column itself — a narrow column wrapped
           the two pill labels onto a second line and grew the control's
           height, taller than the picker beside it. */
        /* st.container(key=...) puts the key class ON the vertical block, so
           this element is the flex parent of its two children (the explainer
           icon, then the segmented control) — flex-direction must be set
           here, and explicitly to row: Streamlit's own stVerticalBlock rule
           sets column, which is what stacked the icon above the pills. */
        .st-key-shap-view-toggle {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-end !important;
            gap: .45rem !important;
            width: 100% !important;
        }
        /* The two element containers default to width:100%, which on a flex
           row gives each half the width and leaves a gap between the icon
           and the toggle. Sizing them to their content keeps the pair snug
           and right-aligned as one group. */
        .st-key-shap-view-toggle > div {
            width: auto !important;
            flex: none !important;
            display: flex !important;
            align-items: center !important;
        }
        .st-key-shap-view-toggle [data-baseweb="button-group"] {
            justify-content: flex-end !important;
        }
        /* Markdown blocks carry their own margins and line-height, which sat
           the icon a few pixels low even with align-items:center. Zero them,
           then nudge up the small remainder so the icon's centre lines up
           with the pill labels rather than the pill box. */
        .st-key-shap-view-toggle [data-testid="stMarkdown"],
        .st-key-shap-view-toggle [data-testid="stMarkdownContainer"] {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
        .st-key-shap-view-toggle .info-tip {
            margin-top: 0; cursor: default; top: -2px;
        }
        /* Opens leftward: the icon sits near the right edge of the page, so a
           left-anchored panel would run off-screen. */
        .st-key-shap-view-toggle .info-tip-body {
            left: auto; right: 0;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button[aria-label],
        [data-testid="stNumberInput"] [role="button"] {
            background-color: var(--light-purple) !important;
            color: var(--purple) !important;
            border-color: #DDD4CF !important;
        }
        [data-testid="stNumberInput"] button svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {
            fill: var(--primary) !important;
            color: var(--primary) !important;
        }
        /* Same 135deg gradient treatment as the teal/blue chips, run through
           light purple instead. The label and clear icon go dark purple: white
           text on a tint this light falls to about 1.6:1, well under the 4.5:1
           it needs, so the colour swap has to come with it. */
        [data-baseweb="tag"] {
            background: linear-gradient(135deg, #FBDCCA 0%, #EAA57C 100%) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"] span, [data-baseweb="tag"] svg {
            color: #592701 !important;
            fill: #592701 !important;
            -webkit-text-fill-color: #592701 !important;
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
            background: #FAF1EB !important;
        }
        div[data-testid="stButton"] > button {
            border-radius: 11px !important;
            font-weight: 780 !important;
            min-height: 42px !important;
            border: 1px solid #DDD4CF !important;
            background: #FFFFFF !important;
            color: var(--purple) !important;
        }
        /* Same gradient as the app header, so the primary action reads as part
           of the same system rather than a second brand colour. */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(100deg, #2E1100 0%, #522401 42%, #EC7200 100%) !important;
            color: #FFFFFF !important;
            border-color: #522401 !important;
        }
        div[data-testid="stButton"] > button:disabled {
            opacity: .55 !important;
            background: #F3F1F8 !important;
            color: #918984 !important;
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


@st.cache_resource(show_spinner=False)
def _logo_data_uri(filename: str) -> str | None:
    path = Path(__file__).with_name("logos") / filename
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_header() -> None:
    """Banner, app version and the date this build was deployed.

    The row count and model version that used to sit here were dropped by
    request; the version below is the application's own, bumped per release.
    """
    # The ZS logo is the header's brand mark. It sits on its own white tile
    # (the logo's background), since its grey wordmark would vanish on the
    # dark end of the header gradient. Aspect ratio kept, not cropped square.
    flu_logo = _logo_data_uri("zs-logo.png")
    mark = (
        f"<img src='{flu_logo}' alt='ZS' style='height:40px; border-radius:8px; "
        f"background:#FFFFFF; display:block;'/>"
        if flu_logo
        else "<span class='app-header-mark'>&#128200;</span>"
    )
    # The badge spans the first training season (2024-25) to the shown season —
    # "2024-2026" for v31 (user request 11/3). Update the 2024 if a future
    # model trains on different seasons.
    season_badge = f"2024-{SEASON.split('-')[0]}"
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-brand">
                {mark}
                <p class="app-title">Customer Response Prediction Engine</p>
            </div>
            <div class="app-header-side app-header-side--right">
                <div class="header-badges">
                    <span class="badge">Season &middot; {escape(season_badge)}</span>
                    <!-- Static, not tied to any real release process yet -- commented out for now.
                    <span class="badge">Version {escape(APP_VERSION)}</span>
                    -->
                    <span class="badge">Last updated &middot; {escape(build_date())}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- context ----------------------------------------------------------------


def choose_context(data: pd.DataFrame) -> tuple[str, pd.DataFrame, list[str]]:
    with st.container(border=True):
        # Health System wide enough for its longest name (the address
        # subtitle sits below it), Products beside it wide enough for all
        # three brand chips plus the clear/dropdown icons; the last column
        # is left empty so neither stretches across the row. Season moved
        # into the header badges, so it no longer needs a tile here. The
        # contract / price group selector that used to sit here was removed:
        # 342 of the 343 accounts hold exactly one price group, so it was a
        # single-option dropdown on all but one of them.
        top = st.columns([1.1, 1.7, 2.5], gap="small")

        customers = sorted(data[CUSTOMER_COLUMN].dropna().unique())
        with top[0]:
            customer = safe_selectbox(
                st, "Health System", customers, "filter_customer",
                preferred=DEFAULT_SCENARIO["customer"],
            )
            customer_data = data[data[CUSTOMER_COLUMN] == customer]

            # Read-only (11/12): the account's address, by IDN against DDD —
            # see merge_address() in prepare_real_data.py. "—" (ADDRESS_COLUMN's
            # own placeholder for the 2/366 accounts DDD has no address for)
            # renders as-is rather than as an empty line. Shown as a subtitle
            # under the selector itself (not its own column/field) so it
            # reads as metadata about the selection, not a second control.
            address = str(customer_data[ADDRESS_COLUMN].iloc[0]) if not customer_data.empty else "—"
            st.markdown(
                f"<div class='current-card-address' title='{escape(address)}'>"
                "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
                "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
                "<path d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'/>"
                "<circle cx='12' cy='10' r='3'/></svg>"
                f"<span>{escape(address)}</span></div>",
                unsafe_allow_html=True,
            )

        # Every row for the account, unfiltered. One account holds two price
        # groups; get_product_rows() already takes the first row per product, so
        # that account scores its first price group without any row being
        # excluded from the frame the deep-dive charts aggregate over.
        contract_data = customer_data

        available = [p for p in PRODUCT_VALUES if p in set(contract_data[PRODUCT_COLUMN])]
        # Open with every product the default scenario offers, so the
        # multi-product path and the differing thresholds are visible up front.
        preferred_products = [p for p in DEFAULT_SCENARIO["products"] if p in available]

        # safe_multiselect() otherwise just intersects whatever was already
        # selected with the new account's own products — so switching from an
        # account with 1 product to one with 3 would silently stay at 1
        # instead of showing all 3. Only reset on an actual customer change
        # (not the first render, where "_customer_for_products" is unset and
        # the preferred_products default above should still apply).
        if (
            st.session_state.get("_customer_for_products") is not None
            and st.session_state["_customer_for_products"] != customer
        ):
            st.session_state["filter_products"] = available
        st.session_state["_customer_for_products"] = customer

        selected_products = safe_multiselect(
            top[1],
            "Products",
            available,
            preferred_products or available[:1],
            "filter_products",
        )
        if not selected_products:
            selected_products = available[:1]

    return customer, contract_data, selected_products


def get_product_rows(contract_data: pd.DataFrame, selected_products: list[str]) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for product in selected_products:
        matches = contract_data[contract_data[PRODUCT_COLUMN] == product]
        if not matches.empty:
            rows[product] = matches.iloc[0]
    return rows


def context_signature(customer: str, contract_data: pd.DataFrame, products: list[str]) -> str:
    group = contract_data.iloc[0][PRICE_GROUP_COLUMN] if not contract_data.empty else ""
    return "|".join([customer, str(group), ",".join(products)])


# --- levers -----------------------------------------------------------------


def default_for_lever(row: pd.Series, config: dict[str, Any]) -> Any:
    """The value a lever opens on: the account's own dead-net price, or the
    dropdown option matching the account's own value for the input (a
    missing input opens on its "No Data" / "Unknown" option)."""
    key = config["key"]
    if key == "dead_net_price":
        value = row["dead_net_price_per_dose"]
        return float(value) if not pd.isna(value) else float(row["ref_product_median_price"])
    return lever_label_for_value(key, row.get(key))


def initialize_product_values(product_rows: dict[str, pd.Series]) -> None:
    for product, row in product_rows.items():
        for config in LEVER_CONFIGS:
            key = lever_state_key(product, config["key"])
            if key not in st.session_state:
                st.session_state[key] = default_for_lever(row, config)


def reset_product_values(product_rows: dict[str, pd.Series]) -> None:
    for product, row in product_rows.items():
        for config in LEVER_CONFIGS:
            st.session_state[lever_state_key(product, config["key"])] = default_for_lever(row, config)


def reset_scenario(product_rows: dict[str, pd.Series]) -> None:
    reset_product_values(product_rows)
    st.session_state[view_key("latest_result")] = None
    st.session_state[view_key("latest_result_at")] = None


def collect_overrides(product: str, row: pd.Series) -> dict[str, Any]:
    """Turn the lever widgets' current selections into model overrides,
    keyed by model feature (plus "dead_net_price").

    A lever still on the option the account's own value opened on feeds that
    value straight back, untouched — so Run Simulation on an untouched panel
    reproduces the baseline card exactly. That matters most for Engagement
    Score: its options are bands, and a band's representative value is never
    the account's own score, so feeding it for an untouched lever would move
    the prediction without the user doing anything. A moved lever feeds the
    option's value (the model's own encoding, or NaN for No Data / Unknown).
    """
    overrides: dict[str, Any] = {}
    for config in LEVER_CONFIGS:
        value = st.session_state[lever_state_key(product, config["key"])]
        key = config["key"]
        if key == "dead_net_price":
            overrides[key] = float(value)
        elif str(value) == str(default_for_lever(row, config)):
            overrides[key] = row.get(key)  # the account's own value, NaN included
        else:
            overrides[key] = lever_value_for_label(key, str(value))
    return overrides


def _lever_table_column_widths(count: int) -> list[float]:
    # Metric wraps fine over two lines (its text is one or two words), so it
    # gives up some width to the product columns — on a narrower laptop
    # screen the previous ratio still squeezed the dropdowns enough to
    # truncate "Medium" to "M...". 10/8: product columns widened a little
    # (from Metric) so the longer lever options — "Competitive Loyalist",
    # "Non-Clinically Inclined", "Unfavourable" — fit on up to two lines.
    # Season stays 0.8: narrower splits "Upcoming Season" mid-word.
    return [0.78, 0.84] + [1.5] * count


def render_lever_table_header(selected_products: list[str]) -> None:
    cols = st.columns(_lever_table_column_widths(len(selected_products)), gap="small")
    for col, label in zip(cols, ["Metric", "Season"] + list(selected_products)):
        with col:
            st.markdown(
                f"<div class='lever-table-header-cell'>{escape(label)}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div class='lever-table-header'></div>", unsafe_allow_html=True)


def _lever_metric_label_html(config: dict[str, Any]) -> str:
    """Lever name in the table's Metric column, with a Deep-Dive-style "i"
    icon right after the name whenever the lever has explanatory text: either
    "tooltip_key" (looked up in DRIVER_SIGNAL_TOOLTIPS, for the four levers
    that are also driver-tab signals) or a literal "tooltip_text" (Dead Net
    Price, whose icon explains its min/max guardrail rather than a driver)."""
    # A connector word ("of", "vs.") is held to the word before it, so a
    # narrow column wraps "Share of Wallet" as "Share of / Wallet" rather than
    # one word per line.
    label_text = re.sub(r" (of|vs\.?|to|and)\b", r"&nbsp;\1", escape(config["label"]))
    label_html = f"<span>{label_text}</span>"
    # tooltip_text (Dead Net Price's guardrail note) is plain text and gets
    # escaped; a DRIVER_SIGNAL_TOOLTIPS hit is raw HTML (built by
    # _driver_tooltip_html's <ul>/<li>/<strong> markup) and is not.
    plain_tooltip = config.get("tooltip_text")
    html_tooltip = DRIVER_SIGNAL_TOOLTIPS.get(config.get("tooltip_key", ""))
    if not plain_tooltip and not html_tooltip:
        return f"<div class='lever-table-metric'>{label_html}</div>"
    tooltip_inner = escape(plain_tooltip) if plain_tooltip else html_tooltip
    tip_body = (
        f"<div class='info-tip-body'>"
        f"<strong>{escape(config['label'])}</strong>"
        f"<div style='margin-top:.35rem;'>{tooltip_inner}</div>"
        f"</div>"
    )
    return (
        f"<div class='lever-table-metric lever-table-metric--with-tip'>"
        f"{label_html}"
        f"<span class='info-tip'>"
        f"<span class='info-tip-icon'>i</span>"
        f"{tip_body}"
        f"</span>"
        f"</div>"
    )


def render_lever_table_row(
    config: dict[str, Any],
    selected_products: list[str],
    product_rows: dict[str, pd.Series],
    metadata: dict[str, Any],
    row_index: int,
) -> None:
    parity = "even" if row_index % 2 == 0 else "odd"
    with st.container(key=f"lever-row-{parity}-{config['key']}"):
        widths = _lever_table_column_widths(len(selected_products))

        row1 = st.columns(widths, gap="small")
        with row1[0]:
            st.markdown(
                _lever_metric_label_html(config),
                unsafe_allow_html=True,
            )
        with row1[1]:
            # Only dead net price carries a prior-season comparison, so it is
            # the only row where "which season is this" is worth stating; the
            # others leave the cell blank rather than repeat it five times.
            label_text = "Upcoming Season" if config["key"] == "dead_net_price" else ""
            st.markdown(
                f"<div class='lever-table-season-label'>{escape(label_text)}</div>",
                unsafe_allow_html=True,
            )
        for col, product in zip(row1[2:], selected_products):
            with col:
                state_key = lever_state_key(product, config["key"])
                if config["kind"] == "choice":
                    st.selectbox(
                        f"{product} {config['label']} upcoming season",
                        config["options"],
                        key=state_key,
                        label_visibility="collapsed",
                    )
                elif config["key"] == "dead_net_price":
                    bounds = metadata["products"][product_key(product)]["lever_ranges"]
                    st.number_input(
                        f"{product} {config['label']} upcoming season",
                        min_value=round(float(bounds["dead_net_price_p01"]), 2),
                        max_value=round(float(bounds["dead_net_price_p99"]), 2),
                        step=config["step"],
                        format=config["format"],
                        key=state_key,
                        label_visibility="collapsed",
                    )
                else:
                    st.number_input(
                        f"{product} {config['label']} upcoming season",
                        min_value=config["min_value"],
                        max_value=config["max_value"],
                        step=config["step"],
                        format=config["format"],
                        key=state_key,
                        label_visibility="collapsed",
                    )

        # Only dead net price has a prior-season value in the source. The other
        # levers used to render this row as a line of em dashes, which read as
        # "the data is missing" rather than "there is no such measure" — so the
        # row is omitted entirely for them.
        if config["prev_column"] is None:
            return

        row2 = st.columns(widths, gap="small")
        with row2[1]:
            st.markdown(
                "<div class='lever-table-season-label lever-table-season-label--prev'>Prior Season</div>",
                unsafe_allow_html=True,
            )
        for col, product in zip(row2[2:], selected_products):
            with col:
                display = format_value(
                    product_rows[product][config["prev_column"]], config["value_type"]
                )
                # Matches the plain "%.2f" the upcoming-season box above it
                # uses, rather than the "$"-prefixed form format_value gives
                # other currency displays.
                display = display.lstrip("$")
                st.markdown(
                    f"<div class='lever-table-prev-value'>{escape(display)}</div>",
                    unsafe_allow_html=True,
                )


def changed_lever_css(selected_products: list[str], product_rows: dict[str, pd.Series]) -> str:
    """CSS that gives every lever whose current value differs from the
    account's own a purple border and tint (user decision 11/3), so the
    untouched levers read as defaults at a glance. Streamlit exposes each
    keyed widget as an ``st-key-<key>`` class on its element container, with
    whitespace in the key turned into dashes. Reset Inputs restores the
    defaults, which empties the selector list and clears every highlight."""
    selectors: list[str] = []
    for product in selected_products:
        row = product_rows.get(product)
        if row is None:
            continue
        for config in LEVER_CONFIGS:
            state_key = lever_state_key(product, config["key"])
            if state_key not in st.session_state:
                continue
            value = st.session_state[state_key]
            default = default_for_lever(row, config)
            if config["key"] == "dead_net_price":
                # Same half-cent rule as build_feature_row(): the box shows 2
                # decimals but prices carry up to 4, so re-typing the shown
                # value must not light the lever up.
                try:
                    changed = abs(float(value) - float(default)) >= 0.005
                except (TypeError, ValueError):
                    changed = False
            else:
                changed = str(value) != str(default)
            if changed:
                css_key = re.sub(r"\s+", "-", state_key)
                selectors.append(f".st-key-{css_key} [data-testid='stNumberInputContainer']")
                selectors.append(f".st-key-{css_key} .react-aria-ComboBox [role='group']")
    if not selectors:
        return ""
    return (
        "<style>\n"
        + ",\n".join(selectors)
        + " {\n"
        "    border: 2px solid #AC5207;\n"
        "    background-color: #FFECE2;\n"
        "    border-radius: 8px;\n"
        "    box-shadow: none;\n"
        "}\n"
        "</style>"
    )


def render_pricing_levers(
    selected_products: list[str],
    product_rows: dict[str, pd.Series],
    metadata: dict[str, Any],
    customer: str = "",
) -> tuple[bool, bool, bool]:
    header_left, header_right = st.columns([5, 1], gap="small")
    with header_left:
        st.markdown("<div class='section-title'>Contracting Levers</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Set upcoming-season terms by product. Only dead net price has a "
            "prior-season comparison in the source data.</div>",
            unsafe_allow_html=True,
        )

    # st.download_button hands over the bytes it was rendered with, so the PDF
    # has to be built now rather than on click. That is also why Run Simulation
    # scores in an on_click callback in main(): callbacks run before the script
    # re-executes, so the result below is the one the user just produced rather
    # than the previous run's.
    #
    # Rebuilding the PDF is expensive, and this function reruns on every widget
    # interaction on the page (e.g. the SHAP Log-odds/Probability toggle), not
    # just when the scenario changes — without memoizing, that made every such
    # toggle visibly lag. `result` is only ever replaced (not mutated) when a
    # new scenario is scored, so its object identity is a cheap, reliable cache
    # key: unrelated widget reruns keep the same `result` object and reuse the
    # cached bytes instead of re-rendering the report.
    result = st.session_state.get(view_key("latest_result"))
    # The scenario report (PDF) is Full Simulation only (user, 11/18): it
    # carries the win probability and the waterfall, which Directional
    # Prediction and Customer Intelligence do not show. Not built there either.
    with header_right:
        if active_view() != "full":
            pass
        elif result is None:
            st.button(
                "",
                icon=":material/download:",
                width="stretch",
                disabled=True,
                help="Run a simulation first.",
            )
        else:
            cache_key = (customer, id(result))
            if st.session_state.get(view_key("_pdf_cache_key")) != cache_key:
                st.session_state[view_key("_pdf_cache_bytes")] = build_scenario_pdf(
                    customer, SEASON, product_rows, result
                )
                st.session_state[view_key("_pdf_cache_key")] = cache_key
            st.download_button(
                "",
                icon=":material/download:",
                data=st.session_state[view_key("_pdf_cache_bytes")],
                file_name=report_filename(customer),
                mime="application/pdf",
                width="stretch",
                help="Download scenario report",
            )

    render_lever_table_header(selected_products)
    for row_index, config in enumerate(LEVER_CONFIGS):
        render_lever_table_row(config, selected_products, product_rows, metadata, row_index)

    highlight_css = changed_lever_css(selected_products, product_rows)
    if highlight_css:
        st.markdown(highlight_css, unsafe_allow_html=True)

    buttons = st.columns(2, gap="small")
    run_clicked = buttons[0].button("Run Simulation", type="primary", width="stretch")
    reset_clicked = buttons[1].button(
        "Reset Inputs", width="stretch", on_click=reset_scenario, args=(product_rows,)
    )
    return run_clicked, reset_clicked, False


# --- watchouts --------------------------------------------------------------


# --- scoring ----------------------------------------------------------------


def score_selected_products(
    models: dict[str, Any],
    metadata: dict[str, Any],
    product_rows: dict[str, pd.Series],
    fold_models: dict[str, Any] | None = None,
    baseline: bool = False,
) -> dict[str, Any]:
    """Score every selected product.

    ``baseline=True`` scores the account's real, unbanded feature values
    (empty overrides) rather than the lever selections, and reports the win
    probability from the modelling team's own pre-computed ``prob0`` column
    instead of a fresh model call — this is what the app shows before the
    user clicks Run Simulation, so it matches the shipped scored file exactly
    rather than an XGBoost-version-dependent recomputation.
    """
    product_results = []
    for product, row in product_rows.items():
        overrides = {} if baseline else collect_overrides(product, row)
        # fold_models drives the win/loss model-confidence score — passed only
        # here, not into the price-sweep callers, which don't display it.
        result = score_scenario(models, metadata, row, overrides, fold_models=fold_models)
        if baseline:
            prob0 = row.get("prob0")
            if prob0 is not None and not pd.isna(prob0):
                win_probability = float(prob0) / 100.0
                margin = abs(win_probability - result["threshold"])
                result["win_probability"] = win_probability
                result["loss_probability"] = 1.0 - win_probability
                result["predicted_outcome"] = "Win" if win_probability >= result["threshold"] else "Loss"
                result["margin_to_threshold"] = margin
                result["confidence"] = confidence_band(margin)
                # The prediction-confidence margin (M) follows the probability
                # actually shown; data, history and stability do not move.
                result.update(confidence_at_probability(result, win_probability))
        result["overrides"] = overrides
        result["is_baseline"] = baseline
        result["sensitivity_flag"] = row.get(SENSITIVITY_COLUMN)
        result["account_sensitivity_band"] = row.get(ACCOUNT_SENSITIVITY_BAND_COLUMN)
        result["account_sensitivity_basis"] = row.get("account_sensitivity_basis")
        result["account_driving_product"] = row.get("account_driving_product")
        result["account_confidence_label"] = row.get(ACCOUNT_CONFIDENCE_LABEL_COLUMN)
        result["prior_incumbent"] = row.get(PRIOR_INCUMBENT_COLUMN)
        result["share_change"] = row.get(SHARE_CHANGE_COLUMN)
        result["clinical_bucket"] = row.get("clinical_bucket")
        result["advocacy"] = row.get(ADVOCACY_COLUMN)
        result["negotiation_stance"] = row.get("negotiation_stance_bucket")
        # Raw columns behind the composite driver scores (per the modelling
        # team's formula table) — used for _raw_insight() so a driver bullet
        # can cite the real-world input, not just the engineered score.
        result["prior_customer_doses"] = row.get(PRIOR_VOLUME_COLUMN)
        result["prior_wallet_share"] = row.get("bob_Prior Season Shr of Wallet (L1BG)")
        result["flip_count"] = row.get("flips_prior_2_seasons")
        result["risk_flag"] = row.get("sam_risk_flag_product")
        result["premium_vs_competitor1"] = row.get(PREMIUM_COLUMN)
        result["current_customer_share"] = row.get(CURRENT_SHARE_COLUMN)
        result["prior_customer_share"] = row.get(PRIOR_SHARE_COLUMN)
        result["dead_net_price"] = row.get(PRICE_BASE_COLUMN)
        result["standard_price_benchmark"] = row.get(STANDARD_PRICE_BENCHMARK_COLUMN)

        current_price = float(overrides.get("dead_net_price", row[PRICE_BASE_COLUMN]))
        median_price = row.get(REFERENCE_PRICE_COLUMN)
        if median_price is not None and not pd.isna(median_price) and float(median_price):
            result["price_vs_median_pct"] = 100 * (current_price - float(median_price)) / float(median_price)
        else:
            result["price_vs_median_pct"] = None

        # Surface the reason rather than silently rendering an empty panel.
        try:
            # redistribute_shap() removes the REDISTRIBUTED_FEATURES drivers
            # (the prebook ratio, and Season since 11/6) as their own bars
            # and folds each value into the rest (60% same side, 40%
            # opposite) — applied here, the one place item["shap"] is set,
            # so every consumer (Narrative Summary, the waterfall, the
            # driver-categories accordion, the PDF report) sees the same
            # adjusted values. merge_shap_features() first adds
            # yoy_price_missing onto the Year-over-year Price Change bar, so
            # the base ("Average Win Probability") stays constant.
            result["shap"] = redistribute_shap(
                merge_shap_features(compute_shap_contributions(models, metadata, row, overrides))
            )
            result["shap_error"] = None
        except Exception as exc:
            result["shap"] = None
            result["shap_error"] = f"{type(exc).__name__}: {exc}"
        try:
            result["symmetric_sensitivity"] = symmetric_price_sensitivity(models, metadata, row, overrides)
        except Exception:
            result["symmetric_sensitivity"] = None
        product_results.append(result)

    # Simple average across products, per the agreed aggregation.
    win_probability = sum(item["win_probability"] for item in product_results) / len(product_results)
    threshold = sum(item["threshold"] for item in product_results) / len(product_results)
    margin = abs(win_probability - threshold)
    imputed = sorted({name for item in product_results for name in item["imputed_features"]})

    return {
        "product_results": product_results,
        "win_probability": win_probability,
        "loss_probability": 1.0 - win_probability,
        "threshold": threshold,
        "margin_to_threshold": margin,
        "predicted_outcome": "Win" if win_probability >= threshold else "Loss",
        "confidence": "High" if margin >= 0.25 else "Medium" if margin >= 0.10 else "Low",
        "imputed_features": imputed,
        "is_portfolio": len(product_results) > 1,
        "is_baseline": baseline,
    }


# --- outcome ----------------------------------------------------------------


# Which narrative style renders in the Narrative Summary card below. Both
# styles live permanently in the codebase (src/narrative_technical.py,
# src/narrative_business.py) — flip this constant and rerun the app to
# switch; no other code needs to change since both modules expose the same
# build_narrative(result, customer, product_filter) shape (the technical
# module also takes medians, supplied automatically below).
NARRATIVE_STYLE = "business"  # "business" or "technical"


def build_narrative(
    result: dict[str, Any],
    customer: str = "",
    product_filter: str | None = None,
    show_probability: bool = True,
) -> str:
    """Dispatches to the selected narrative style (NARRATIVE_STYLE above).
    Both styles cover the same four sections' worth of ground per product —
    outcome, account drivers, price sensitivity, recommended read — just at
    different technical depth. See the two modules' own docstrings for what
    each one shows.

    ``show_probability=False`` drops the "The model estimates a x% chance of
    winning" sentence (Directional Prediction / Customer Intelligence, user
    11/18). Only the business style supports it; the technical style is not
    in use and would still state the probability.
    """
    if NARRATIVE_STYLE == "technical":
        medians = product_medians()
        raw_medians = raw_column_medians()
        return narrative_technical.build_narrative(result, customer, product_filter, medians, raw_medians)
    return narrative_business.build_narrative(result, customer, product_filter, show_probability=show_probability)


def render_top_drivers(result: dict[str, Any]) -> None:
    """One mini card per product: its top 2 drivers overall by |SHAP|,
    regardless of sign, each tagged Positive or Negative. A compact card, so
    2 total (not the Narrative Summary's 2-best-and-2-worst) is what fits;
    see _driver_clause() for the fuller 4-driver text version.
    """
    product_results = result["product_results"]
    columns = st.columns(len(product_results), gap="small")
    for column, item in zip(columns, product_results):
        is_win = item["predicted_outcome"] == "Win"
        dot_class = "drivers-dot-win" if is_win else "drivers-dot-loss"
        shap = item.get("shap")
        rows = ""
        if shap is None or shap.empty:
            rows = "<div class='drivers-row'>Contributions unavailable</div>"
        else:
            eligible = shap[~shap["feature"].isin(NARRATIVE_EXCLUDED_FEATURES)]
            top = eligible.reindex(
                eligible["shap"].abs().sort_values(ascending=False).index
            ).head(2)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                positive = row["shap"] >= 0
                pill_class = "drivers-pill-positive" if positive else "drivers-pill-negative"
                pill_text = "Positive" if positive else "Negative"
                rows += (
                    f"<div class='drivers-row'>"
                    f"<span class='drivers-row-left'>"
                    f"<span class='drivers-num'>{rank}</span>"
                    f"<span class='drivers-name'>{escape(FEATURE_LABELS[row['feature']])}</span>"
                    f"</span>"
                    f"<span class='drivers-pill {pill_class}'>{pill_text}</span>"
                    f"</div>"
                )
            if not rows:
                rows = "<div class='drivers-row'>No drivers available</div>"
        column.markdown(
            f"<div class='drivers-card'>"
            f"<div class='drivers-card-head'>"
            f"<span class='drivers-dot {dot_class}'></span>"
            f"<span class='drivers-card-name'>{escape(str(item['product']))}</span>"
            f"</div>"
            f"{rows}"
            f"</div>",
            unsafe_allow_html=True,
        )


# Keys must match the confidence bands exactly ("Moderate", not "Medium" —
# band_for() in src/confidence.py emits High/Moderate/Low).
CONFIDENCE_COLORS = {"High": "var(--teal)", "Moderate": "var(--orange)", "Low": "var(--red)"}
_CONFIDENCE_ICON_SVG = (
    "<svg width='14' height='14' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'>"
    "<rect x='1.5' y='8.5' width='3' height='6' rx='.5' fill='currentColor'/>"
    "<rect x='6.5' y='4.5' width='3' height='10' rx='.5' fill='currentColor'/>"
    "<rect x='11.5' y='1.5' width='3' height='13' rx='.5' fill='currentColor'/>"
    "</svg>"
)

# Hover card on each product's confidence row (11/16): only the two-sentence
# confidence summary (src/confidence.py business_summary()), in the style of
# the user's examples — no score, factors or detail.
def confidence_points_html(text: str) -> str:
    # One bullet per sentence (user, 11/16: "two sentences, two bullet points").
    sentences = [s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip()) if s]
    if len(sentences) < 2:
        return escape(text)
    return "<ul class='confidence-points'>" + "".join(f"<li>{escape(s)}</li>" for s in sentences) + "</ul>"


def confidence_tip_html(detail: dict[str, Any] | None) -> str:
    text = (detail or {}).get("explanation", {}).get("summary") or "Confidence is unavailable for this product."
    return f"<div class='confidence-tip'>{confidence_points_html(text)}</div>"


# Outcome band (user decision 11/11, replacing the percentage and its bar;
# 20pp Watch width agreed 11/3): Comfortable at or above the product's
# threshold, Watch up to 20 points below it, At Risk further below.
# predicted_outcome stays the binary threshold decision the narrative, report
# and tests use.
WATCH_BAND_WIDTH = 0.20


def outcome_band(win_probability: float, threshold: float) -> tuple[str, str, str]:
    """(label, css class, one-line meaning) for a product's win probability."""
    if win_probability >= threshold:
        return "Comfortable Win", "outcome-band-comfortable", "At or above win threshold"
    # Rounded so a value exactly 20 points below is Watch despite float error
    # (0.55 - 0.20 == 0.35000000000000003).
    if round(win_probability - threshold, 9) >= -WATCH_BAND_WIDTH:
        return "Watch", "outcome-band-watch", "Within 20 pts below threshold"
    return "At Risk", "outcome-band-risk", "Over 20 pts below threshold"


def render_outcome(
    result: dict[str, Any] | None,
    customer: str = "",
    product_rows: dict[str, pd.Series] | None = None,
) -> None:
    view = active_view()
    if view == "intel":
        # Customer Intelligence: no Predicted Outcome block (no percentage, no
        # band, no confidence); the panel opens on the driver contributions.
        if not result:
            st.info("Adjust the levers and click Run Simulation.")
            return
    else:
        render_outcome_cards(result, view)
        if not result:
            return
    render_outcome_details(result, customer, product_rows)


# Band class -> (pill, bar fill) classes for the Full Simulation card, which
# shows the percentage and its bar with the band as a small pill (the pre-11/11
# card, restored for this view only).
_FULL_CARD_CLASSES = {
    "outcome-band-comfortable": ("outcome-pill-win", "outcome-card-fill-win"),
    "outcome-band-watch": ("outcome-pill-watch", "outcome-card-fill-watch"),
    "outcome-band-risk": ("outcome-pill-loss", "outcome-card-fill-loss"),
}


def render_outcome_cards(result: dict[str, Any] | None, view: str) -> None:
    """Predicted Outcome header and one card per product (full / directional views)."""
    subtitle = (
        "Model-predicted win probability and outcome for each selected product."
        if view == "full"
        else "Model-predicted outcome band for each selected product, against its win threshold."
    )
    header, refresh = st.columns([4, 1], gap="small")
    with header:
        st.markdown("<div class='section-title'>Predicted Outcome</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-subtitle'>{escape(subtitle)}</div>", unsafe_allow_html=True)
    run_at = st.session_state.get(view_key("latest_result_at"))
    with refresh:
        if result and result.get("is_baseline"):
            label = "Current account data"
        elif run_at:
            label = f"Last run: {run_at}"
        else:
            label = ""
        if label:
            st.markdown(
                f"<div style='text-align:right; font-size:.72rem; color:var(--text-secondary); "
                f"margin-top:.2rem;'>{escape(label)}</div>",
                unsafe_allow_html=True,
            )

    if not result:
        st.info("Adjust the levers and click Run Simulation.")
        return

    product_results = result["product_results"]
    cards = st.columns(len(product_results), gap="small")
    for index, (column, item) in enumerate(zip(cards, product_results)):
        name = str(item["product"])
        # Last of several cards: open the hover card leftwards, inside the page.
        tip_side = " tip-right" if index and index == len(product_results) - 1 else ""
        band, band_class, _meaning = outcome_band(item["win_probability"], float(item["threshold"]))
        # Win/loss prediction confidence (v5, src/confidence.py) — recomputed live per scenario.
        # None when the fold models are unavailable: shown as an em dash rather
        # than falling back to the margin-based confidence_band(), which is a
        # different measure and would silently redefine what "High" means.
        confidence = item.get("winloss_confidence_label") or "—"
        confidence_color = CONFIDENCE_COLORS.get(confidence, "var(--text-secondary)")
        if view == "full":
            pct = float(item["win_probability"])
            pill_class, fill_class = _FULL_CARD_CLASSES[band_class]
            outcome_html = (
                f"<div class='outcome-card-head'>"
                f"<span class='outcome-card-percent'>{pct:.0%}</span>"
                f"<span class='outcome-pill {pill_class}'>{escape(band).upper()}</span>"
                f"</div>"
                f"<div class='outcome-card-track'>"
                f"<div class='outcome-card-fill {fill_class}' style='width:{pct * 100:.0f}%;'></div>"
                f"</div>"
                f"<div class='outcome-card-caption outcome-card-caption-text'>Win probability</div>"
            )
        else:
            outcome_html = (
                f"<span class='outcome-band {band_class}'>"
                f"<span class='outcome-band-dot'></span>{escape(band).upper()}</span>"
                f"<div class='outcome-card-caption'></div>"
            )
        column.markdown(
            f"<div class='outcome-card'>"
            f"<div class='outcome-card-name'>{escape(name)}</div>"
            f"{outcome_html}"
            f"<div class='outcome-confidence-row{tip_side}'>"
            f"<div class='outcome-confidence-left'>"
            f"<span class='outcome-confidence-icon'>{_CONFIDENCE_ICON_SVG}</span>"
            f"<span class='outcome-confidence-label'>Confidence</span>"
            f"<i class='outcome-confidence-info'>i</i>"
            f"</div>"
            f"<div class='outcome-confidence-right'>"
            f"<span class='outcome-confidence-value' style='color:{confidence_color};'>{escape(confidence)}</span>"
            f"</div>"
            f"{confidence_tip_html(item.get('winloss_confidence_detail'))}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Top Drivers (per product) cards — commented out for now, per request.
    # The shared product filter below is kept regardless, since the SHAP
    # curve and Narrative Summary both still need it.
    # st.markdown(
    #     "<div class='small-label' style='margin-top:.9rem; color:var(--text-primary); font-size:.8rem;'>Top Drivers (per product)</div>",
    #     unsafe_allow_html=True,
    # )
    # render_top_drivers(result)


def render_outcome_details(
    result: dict[str, Any],
    customer: str = "",
    product_rows: dict[str, pd.Series] | None = None,
) -> None:
    """Driver contributions, Narrative Summary and Confidence Summary (every view)."""
    product_results = result["product_results"]

    # SHAP contribution and the Narrative Summary below it share one product
    # picker, so the two sub-sections always describe the same product. It
    # opens on the first of PRODUCT_VALUES the account actually has — Viresta,
    # else Immunexa, else Noventra — walked explicitly rather than taking
    # products[0], so the order is this rule and not an incidental
    # consequence of how get_product_rows() happened to build the dict.
    #
    # The pick is remembered in "shap_product_choice", a plain session_state
    # entry that is deliberately NOT the widget's own key.
    #
    # Streamlit discards a keyed widget's state on any script run that never
    # instantiates the widget, and Run Simulation is exactly that case: main()
    # st.rerun()s as soon as it has scored, before this column renders (the
    # rerun is there so the Download Report button above picks up the new
    # result). So "shap_product" is gone by the time the page redraws, and
    # anything derived from it — the widget's own state, or a value seeded
    # into that same key — falls back to the default and silently snaps the
    # picker to Viresta, or to whichever product sits at position 0 when Viresta
    # is not selected. A non-widget key survives that collection, so it is the
    # thing the widget's index is read from, and it is written back from the
    # widget's return value below. The `not in products` guard re-seeds when
    # the remembered pick is genuinely gone (its product de-selected), and
    # main() clears the key outright on a health-system change so hopping
    # between accounts always opens on Viresta again.
    products = [str(item["product"]) for item in product_results]
    choice_key = view_key("shap_product_choice")
    if st.session_state.get(choice_key) not in products:
        st.session_state[choice_key] = next(
            (product for product in PRODUCT_VALUES if product in products), products[0]
        )
    default = products.index(st.session_state[choice_key])

    if active_view() != "intel":  # intel has no cards above to space from
        st.markdown("<div style='height:.9rem;'></div>", unsafe_allow_html=True)
    shap_heading, shap_toggle = st.columns([2.55, 1.45], gap="small", vertical_alignment="center")
    with shap_heading:
        st.markdown(
            "<div class='shap-heading-row'>"
            "<span class='small-label' style='margin:0; color:var(--text-primary); font-size:.8rem;'>"
            "Contribution to predicted Win probability</span>"
            f"{WATERFALL_HELP}"
            "</div>",
            unsafe_allow_html=True,
        )
    # Log-odds/Probability toggle disabled per request — Probability only,
    # no user-facing switch. Left commented (not deleted) so it can be
    # restored by uncommenting this block and removing the `view_mode =
    # "Probability"` line below.
    # with shap_toggle:
    #     # Icon first, toggle second — the container is a flex row (see the
    #     # .st-key-shap-view-toggle CSS), so this order puts the icon
    #     # immediately left of the pills.
    #     with st.container(key="shap-view-toggle"):
    #         st.markdown(SHAP_TOGGLE_HELP, unsafe_allow_html=True)
    #         view_mode = st.segmented_control(
    #             "SHAP view",
    #             ["Log-odds", "Probability"],
    #             default="Probability",
    #             key="shap_view_mode",
    #             label_visibility="collapsed",
    #         ) or "Probability"
    view_mode = "Probability"
    with st.container(key="shap-product-picker"):
        chosen = st.selectbox(
            "Narrative & SHAP product",
            products,
            index=default,
            key=view_key("shap_product"),
            label_visibility="collapsed",
        )
    # Mirror the widget back into the surviving key, so the next run that has
    # lost the widget's state still opens on what the user last chose.
    st.session_state[choice_key] = chosen
    selected = next(item for item in product_results if str(item["product"]) == chosen)
    base_row = (product_rows or {}).get(chosen)
    render_shap_chart(
        shap_chart_contributions(selected.get("shap")),
        selected.get("shap_error"),
        view_mode,
        base_row=base_row,
        overrides=selected.get("overrides"),
    )

    st.markdown("<div style='height:.55rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-label' style='margin:0; color:var(--text-primary); font-size:.8rem;'>"
        "Narrative Summary</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='narrative-card'>"
        f"{build_narrative(result, customer, product_filter=chosen, show_probability=active_view() == 'full')}"
        "</div>",
        unsafe_allow_html=True,
    )
    # Confidence Summary card removed from all three views (user, 11/17); the
    # two-bullet summary stays in the Confidence hover on the outcome cards.
    # render_confidence_summary(selected)


def confidence_summary_html(detail: dict[str, Any]) -> str:
    """The Confidence Summary card for one product (v5, src/confidence.py):
    score and band, the rule-based explanation, one suggested action, and the
    four factors behind the score — same picker as the Narrative Summary."""
    explanation = detail["explanation"]
    band = detail["band"]
    color = CONFIDENCE_COLORS.get(band, "var(--text-secondary)")
    parts = [
        "<div class='confidence-head'>"
        f"<span class='confidence-score' style='color:{color};'>{detail['score']:.0f}"
        "<span class='confidence-score-max'>/100</span></span>"
        f"<span class='confidence-band' style='color:{color}; border-color:{color};'>{escape(band)}</span>"
        f"<span class='confidence-headline'>{escape(explanation['headline'])}</span>"
        "</div>"
    ]
    if explanation["summary"]:
        parts.append(f"<div>{confidence_points_html(explanation['summary'])}</div>")
    if explanation["action"]:
        parts.append(f"<p><strong>Suggested action:</strong> {escape(explanation['action'])}</p>")
    parts.append("<div class='narrative-rule'></div><div class='narrative-sub'>What the score is made of</div>")
    for factor in explanation["factors"]:
        score = max(0.0, min(100.0, float(factor["score"])))
        parts.append(
            "<div class='confidence-factor'>"
            "<div class='confidence-factor-top'>"
            f"<span class='confidence-factor-name'>{escape(factor['name'])}"
            f"<span class='confidence-factor-weight'>{factor['weight']:.0%} of score</span></span>"
            f"<span class='confidence-factor-value'>{score:.0f}</span>"
            "</div>"
            f"<div class='confidence-factor-track'><div class='confidence-factor-fill' style='width:{score:.1f}%;'></div></div>"
            f"<div class='confidence-factor-detail'>{escape(factor['detail'])}</div>"
            "</div>"
        )
    parts.append(
        f"<div class='confidence-caveat'>{escape(explanation['caveat'])} "
        f"High: {BAND_HIGH:.0f} and above · Moderate: {BAND_MODERATE:.0f}–{BAND_HIGH - 1:.0f} · "
        f"Low: below {BAND_MODERATE:.0f}.</div>"
    )
    return "".join(parts)


def render_confidence_summary(item: dict[str, Any]) -> None:
    st.markdown("<div style='height:.55rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-label' style='margin:0; color:var(--text-primary); font-size:.8rem;'>"
        "Confidence Summary</div>",
        unsafe_allow_html=True,
    )
    detail = item.get("winloss_confidence_detail")
    if detail is None:
        st.markdown(
            "<div class='narrative-card'>Confidence is unavailable for this product "
            "(the model versions it needs could not be loaded).</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(f"<div class='narrative-card'>{confidence_summary_html(detail)}</div>", unsafe_allow_html=True)


def _waterfall_frame(contributions: pd.DataFrame, base_value: float) -> pd.DataFrame:
    """Chain the SHAP contributions into waterfall start/end pairs.

    ``contributions`` arrives sorted by absolute impact, so the largest driver is
    row 0 and is drawn at the top. Accumulation runs the other way — from
    E[f(X)] at the bottom row upward — so the top bar ends exactly at f(x),
    which is the TreeSHAP additivity identity made visible.
    """
    rows = contributions.to_dict("records")
    running = base_value
    for row in reversed(rows):
        row["start"] = running
        running += float(row["shap"])
        row["end"] = running

    frame = pd.DataFrame(rows)
    # Driver name only. The feature's value is still in the hover tooltip.
    frame["axis_label"] = frame["label"].astype(str)
    # Display text, not raw values: the segmentation flag's value is a
    # category label, and a column mixing it with numbers can't be serialised
    # to Arrow for the chart.
    frame["value"] = [
        str(value) if isinstance(value, str) else _trim_number(value) for value in frame["value"]
    ]

    lowest = float(frame[["start", "end"]].min().min())
    highest = float(frame[["start", "end"]].max().max())
    span = max(highest - lowest, 1e-6)

    frame["mid"] = (frame["start"] + frame["end"]) / 2
    frame["arrow"] = np.where(frame["shap"] >= 0, "triangle-right", "triangle-left")
    # Wide enough to carry the number inside the bar, as SHAP does; otherwise
    # the number sits just beyond the tip.
    frame["inside"] = frame["magnitude"] / span >= 0.16
    pad = span * 0.015
    frame["label_x"] = np.where(
        frame["inside"],
        frame["mid"],
        np.where(frame["shap"] >= 0, frame["end"] + pad, frame["end"] - pad),
    )
    return frame


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _probability_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-express the same waterfall in probability space.

    SHAP additivity holds in log-odds, not probability — sigmoid is
    non-linear, so a feature's probability-space "contribution" is not its
    SHAP value transformed, it is the change in sigmoid(running total) that
    step causes. This is the same link="logit" technique SHAP's own waterfall
    plot uses: transform the cumulative start/end, not the raw values.
    """
    view = frame.copy()
    view["start"] = _sigmoid(frame["start"])
    view["end"] = _sigmoid(frame["end"])
    view["delta"] = view["end"] - view["start"]
    view["mid"] = (view["start"] + view["end"]) / 2

    lowest = float(view[["start", "end"]].min().min())
    highest = float(view[["start", "end"]].max().max())
    span = max(highest - lowest, 1e-6)

    view["inside"] = view["delta"].abs() / span >= 0.16
    pad = span * 0.015
    view["label_x"] = np.where(
        view["inside"],
        view["mid"],
        np.where(view["delta"] >= 0, view["end"] + pad, view["end"] - pad),
    )
    return view


def _trim_number(value: Any) -> str:
    """Compact numeric label: 0.778, 2.98, 34.372 — no trailing zeros."""
    if value is None or pd.isna(value):
        return "n/a"
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _importance_band(value: float, bands: tuple[float, float]) -> str:
    high, medium = bands
    return "High" if value >= high else ("Medium" if value >= medium else "Low")


def _pp_html(pp: float) -> str:
    colour = SHAP_POSITIVE if pp >= 0 else SHAP_NEGATIVE
    return f"<span style='color:{colour}; font-weight:800;'>{pp:+.1f}pp</span>"


def _meter_html(level: str, colour: str) -> str:
    filled = {"High": 3, "Medium": 2, "Low": 1}[level]
    bars = "".join(
        f"<span style='display:inline-block; width:5px; height:{6 + k * 4}px; "
        f"background:{colour if k < filled else '#F0E1D9'}; border-radius:2px; "
        f"margin-left:2px; vertical-align:bottom;'></span>"
        for k in range(3)
    )
    return f"<span style='display:inline-flex; align-items:flex-end; width:25px;'>{bars}</span>"


def render_driver_categories(view: pd.DataFrame) -> None:
    """The ranked driver-categories accordion (user decision 11/5): each of
    DRIVER_CATEGORIES as a native <details> row — collapsed by default, a
    click shows its drivers with this scenario's probability contribution
    and an H/M/L importance band. Two styles (see DRIVER_CATEGORY_STYLE).
    Contributions are the same probability-space deltas the waterfall draws,
    so the two sections always agree."""
    rows = view[["feature", "label", "delta"]].copy()
    rows["pp"] = rows["delta"] * 100

    categories = []
    seen: set[str] = set()
    for name, features in DRIVER_CATEGORIES:
        subset = rows[rows["feature"].isin(features)]
        if subset.empty:
            continue
        seen.update(subset["feature"])
        drivers = subset.reindex(subset["pp"].abs().sort_values(ascending=False).index)
        categories.append((name, drivers))
    leftover = rows[~rows["feature"].isin(seen)]
    if not leftover.empty:  # a future model input not yet mapped to a bucket
        categories.append(("Other", leftover))
    categories.sort(key=lambda item: item[1]["pp"].abs().sum(), reverse=True)

    total_abs = max(sum(c[1]["pp"].abs().sum() for c in categories), 1e-6)
    max_pp = max(rows["pp"].abs().max(), 1e-6)
    is_ranked = DRIVER_CATEGORY_STYLE == "ranked"

    def driver_rows(drivers: pd.DataFrame) -> str:
        out = ""
        for _, d in drivers.iterrows():
            pp = float(d["pp"])
            band = _importance_band(abs(pp), DRIVER_IMPORTANCE_BANDS)
            colour = SHAP_POSITIVE if pp >= 0 else SHAP_NEGATIVE
            fg = IMPORTANCE_COLORS[band][0]
            if is_ranked:
                width = max(5, abs(pp) / max_pp * 150)
                out += (
                    "<div class='drvcat-driver'>"
                    f"<span class='drvcat-driver-name'>{escape(str(d['label']))}</span>"
                    f"<span><span style='display:inline-block; width:{width:.0f}px; height:10px; "
                    f"background:{colour}; border-radius:3px;'></span></span>"
                    f"<span style='text-align:right;'>{_pp_html(pp)}</span>"
                    f"<span style='font-size:.62rem; font-weight:800; color:{fg}; "
                    f"text-transform:uppercase;'>{band}</span>"
                    "</div>"
                )
            else:
                verb = "raises win" if pp > 0 else ("lowers win" if pp < 0 else "no effect")
                out += (
                    "<div class='drvcat-driver drvcat-driver--meters'>"
                    f"{_meter_html(band, colour)}"
                    f"<span class='drvcat-driver-name'>{escape(str(d['label']))}</span>"
                    f"<span style='font-size:.68rem; color:#8D8581;'>{band} &middot; {verb}</span>"
                    f"<span style='text-align:right;'>{_pp_html(pp)}</span>"
                    "</div>"
                )
        return out

    blocks = ""
    for rank, (name, drivers) in enumerate(categories, start=1):
        net = float(drivers["pp"].sum())
        cat_abs = float(drivers["pp"].abs().sum())
        band = _importance_band(cat_abs, CATEGORY_IMPORTANCE_BANDS)
        net_colour = SHAP_POSITIVE if net >= 0 else SHAP_NEGATIVE
        if is_ranked:
            share = cat_abs / total_abs * 100
            summary = (
                "<summary class='drvcat-head'>"
                f"<span class='drvcat-rank'>{rank}</span>"
                "<span style='flex:1;'>"
                "<span class='drvcat-head-top'>"
                f"<span class='drvcat-name'>{escape(name)}</span>"
                f"<span class='drvcat-sub'>{share:.0f}% of total impact &middot; {band}</span>"
                "</span>"
                "<span class='drvcat-track'><span class='drvcat-fill' "
                f"style='width:{share:.0f}%;'></span></span>"
                "</span>"
                "<span class='drvcat-chev'>&#9656;</span>"
                "</summary>"
            )
            block_style = ""
        else:
            summary = (
                "<summary class='drvcat-head'>"
                f"{_meter_html(band, net_colour)}"
                f"<span class='drvcat-name' style='flex:1;'>{escape(name)}</span>"
                f"<span class='drvcat-sub'>{band} importance</span>"
                f"{_pp_html(net)}"
                "<span class='drvcat-chev'>&#9656;</span>"
                "</summary>"
            )
            block_style = f" style='border-left:4px solid {net_colour};'"
        blocks += f"<details class='drvcat'{block_style}>{summary}<div>{driver_rows(drivers)}</div></details>"

    legend = (
        "<span class='drvcat-legend'>"
        f"<span class='drvcat-dot' style='background:{SHAP_POSITIVE};'></span>Increases probability of win"
        f"<span class='drvcat-dot' style='background:{SHAP_NEGATIVE}; margin-left:.8rem;'></span>"
        "Decreases probability of win"
        "</span>"
    )
    st.markdown(
        """<style>
        .drvcat {background:#fff; border:1px solid #F0E1D9; border-radius:12px;
                 margin-bottom:8px; overflow:hidden;}
        .drvcat summary {list-style:none; cursor:pointer;}
        .drvcat summary::-webkit-details-marker {display:none;}
        .drvcat-head {display:flex; align-items:center; gap:12px; padding:11px 16px;}
        .drvcat-rank {width:26px; height:26px; border-radius:50%; background:#2D2623; color:#fff;
                 font-size:.78rem; font-weight:800; display:inline-flex; align-items:center;
                 justify-content:center; flex:none;}
        .drvcat-head-top {display:flex; justify-content:space-between; align-items:center;}
        .drvcat-name {font-weight:700; font-size:.88rem; color:var(--text-primary);}
        .drvcat-sub {font-size:.72rem; color:#8D8581;}
        .drvcat-track {display:block; margin-top:6px; height:6px; background:#EFEDF4; border-radius:4px;}
        .drvcat-fill {display:block; height:6px; border-radius:4px;
                 background:linear-gradient(90deg,#803C05,#C15C01);}
        .drvcat-chev {color:#803C05; font-size:.8rem;}
        .drvcat[open] .drvcat-chev {transform:rotate(90deg);}
        .drvcat-driver {display:grid; grid-template-columns:220px 1fr 56px 48px; align-items:center;
                 gap:10px; padding:7px 16px 7px 54px; border-top:1px dashed #EFEDF4;
                 font-size:.78rem; color:#4D4642; background:#FBFAFD;}
        .drvcat-driver--meters {grid-template-columns:25px 1fr auto 56px; padding-left:20px;}
        .drvcat-subhead {display:flex; justify-content:space-between; align-items:center;
                 margin:.3rem 0 .5rem;}
        .drvcat-subhead-title {font-size:.72rem; font-weight:800; letter-spacing:.5px;
                 color:#6E6662; text-transform:uppercase;}
        .drvcat-legend {font-size:.72rem; color:#6E6662; display:inline-flex; align-items:center;}
        .drvcat-dot {width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:.35rem;}
        </style>"""
        "<div class='drvcat-subhead'>"
        "<span class='drvcat-subhead-title'>Driver categories &middot; ranked by impact</span>"
        f"{legend}</div>"
        f"{blocks}",
        unsafe_allow_html=True,
    )


# The 9 fixed price steps the curve is evaluated at (11/12 decision — not a
# smooth sweep across the model's observed price range, which is what
# price_sensitivity_curve() in src/scoring.py builds: this is specifically
# +/-5/10/15/20% off the scenario's own current price, plus current itself).
PRICE_RESPONSE_STEPS_PCT = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def price_response_points(
    models: dict[str, Any], metadata: dict[str, Any], base_row: pd.Series, overrides: dict[str, Any]
) -> pd.DataFrame:
    """Win probability at 9 price points around the scenario's current price:
    +/-5/10/15/20% and current itself. Every other lever stays at the
    scenario's own overridden value — only price moves.

    Account- and product-specific by construction: base_row carries this
    account's own feature values and current price, and threshold_for() (via
    score_scenario) is this product's own threshold — a different account or
    product will trace a different line and cross the threshold at a
    different point, or not at all.
    """
    overrides = dict(overrides or {})
    current_price = float(overrides.get("dead_net_price", base_row[PRICE_BASE_COLUMN]))
    rows = []
    for pct in PRICE_RESPONSE_STEPS_PCT:
        price = current_price * (1 + pct / 100)
        result = score_scenario(models, metadata, base_row, {**overrides, "dead_net_price": price})
        rows.append(
            {
                "pct_change": pct,
                "dead_net_price": price,
                "win_probability": result["win_probability"],
                "threshold": result["threshold"],
                "is_current": pct == 0.0,
            }
        )
    return pd.DataFrame(rows)


def render_price_response_curve(
    models: dict[str, Any], metadata: dict[str, Any], base_row: pd.Series, overrides: dict[str, Any]
) -> None:
    """Price x win-probability line (11/12, full-visibility persona): the 9
    fixed price steps against the product's own win threshold, in the same
    expander as the SHAP waterfall, directly below it."""
    curve = price_response_points(models, metadata, base_row, overrides)
    threshold = float(curve["threshold"].iloc[0])

    st.markdown(
        "<div style='height:1rem;'></div>"
        "<div class='small-label' style='margin:0 0 .3rem; color:var(--text-primary); font-size:.8rem;'>"
        "Price-response curve</div>",
        unsafe_allow_html=True,
    )

    # Each tick shows its % step AND, in brackets, the $ price that step is
    # (11/12) — e.g. "0%\n($57.35)" — rather than leaving the reader to work
    # the price back out from the % themselves. tickValues pins exactly these
    # 9 points so Vega-Lite doesn't add any of its own in between.
    tick_values = curve["pct_change"].tolist()
    tick_labels = {
        # "\\n" (2 chars: backslash + n), not an actual newline: this string
        # is embedded as a JS string literal in labelExpr below, and a real
        # newline byte inside a single-quoted JS literal is a syntax error —
        # it silently broke the whole chart (0-height) until caught 11/12.
        float(pct): ("0%" if pct == 0 else f"{pct:+.0f}%") + f"\\n(${price:,.2f})"
        for pct, price in zip(curve["pct_change"], curve["dead_net_price"])
    }
    label_expr = " : ".join(f"datum.value == {pct} ? '{label}'" for pct, label in tick_labels.items())
    label_expr += " : ''"

    line = (
        alt.Chart(curve)
        .mark_line(color=SHAP_ANCHOR_FINAL, strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X(
                "pct_change:Q",
                title="Price change vs current scenario",
                axis=alt.Axis(
                    tickCount=9, values=tick_values, labelExpr=label_expr,
                    # All 9 must show (not Vega-Lite's own overlap thinning) —
                    # angled and small enough that the full set fits without
                    # Vega collapsing the \n inside labelExpr to a space.
                    labelOverlap=False, labelAngle=-35, labelFontSize=9.5,
                    labelLimit=90, labelOffset=4, grid=True, gridColor="#E6E2F0",
                ),
            ),
            y=alt.Y(
                "win_probability:Q",
                title="Win probability",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            tooltip=[
                alt.Tooltip("pct_change:Q", format="+.0f", title="Price change"),
                alt.Tooltip("dead_net_price:Q", format="$.2f", title="Price"),
                alt.Tooltip("win_probability:Q", format=".1%", title="Win probability"),
            ],
        )
    )
    # The current-price point, picked out from the rest of the line.
    current = curve[curve["is_current"]]
    current_dot = (
        alt.Chart(current)
        .mark_point(size=160, filled=True, color=SHAP_ANCHOR_FINAL, stroke="#FFFFFF", strokeWidth=2)
        .encode(x="pct_change:Q", y="win_probability:Q")
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color="#9891A6", strokeDash=[4, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    threshold_label = (
        alt.Chart(pd.DataFrame({"y": [threshold], "t": [f"Win threshold {threshold:.0%}"]}))
        .mark_text(align="left", baseline="bottom", dx=-180, dy=-4, fontSize=11, fontWeight="bold", color="#6B6479")
        .encode(y="y:Q", text="t:N")
    )

    st.altair_chart(
        alt.layer(line, current_dot, threshold_rule, threshold_label)
        .properties(height=240, padding={"top": 10, "right": 10, "left": 5, "bottom": 35}),
        width="stretch",
    )


_DEFERRED_KEY = "_deferred_renders"


def defer(draw: Any) -> None:
    """Queue slow, out-of-sight drawing for the end of this run (see run_deferred)."""
    st.session_state.setdefault(_DEFERRED_KEY, []).append(draw)


def run_deferred() -> None:
    """Draw everything defer() queued. Called last in main(), once the rest is on screen."""
    queued = st.session_state.pop(_DEFERRED_KEY, [])
    for draw in queued:
        draw()


def render_shap_chart(
    contributions: pd.DataFrame | None,
    error: str | None = None,
    mode: str = "Probability",
    base_row: pd.Series | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    if contributions is None or contributions.empty:
        if error:
            st.warning(f"SHAP contributions could not be computed — {error}")
        else:
            st.info("SHAP contributions are not available for this scenario.")
        return
    base_value = float(contributions.attrs.get("base_value", 0.0))
    frame = _waterfall_frame(contributions, base_value)
    final = float(frame["end"].iloc[0])
    is_probability = mode == "Probability"

    if is_probability:
        view = _probability_view(frame)
        view["display_value"] = view["delta"]
        x_base, x_final = float(_sigmoid(base_value)), float(_sigmoid(final))
        axis_title = "Model output (probability)"
        axis_format = ".0%"
        text_format = "+.1%"
        fx_text = f"Account Prediction = <strong>{x_final:.1%}</strong> &nbsp;·&nbsp; Average Win Probability = <strong>{x_base:.1%}</strong>"
    # Log-odds view disabled per request — render_outcome() now hardcodes
    # mode="Probability", so this branch is unreachable. Left commented
    # (not deleted) so it can be restored alongside the toggle above.
    # else:
    #     view = frame
    #     view["display_value"] = view["shap"]
    #     x_base, x_final = base_value, final
    #     axis_title = "Model output (log-odds)"
    #     axis_format = ".2f"
    #     text_format = "+.2f"
    #     fx_text = f"Account Prediction = <strong>{final:+.3f}</strong> &nbsp;·&nbsp; Average Win Probability = <strong>{base_value:+.3f}</strong>"

    # The "Account Prediction = x% · Average Win Probability = y%" line shows
    # only in Full Simulation (user, 11/17): it states the win probability,
    # which Directional Prediction and Customer Intelligence do not show.
    if active_view() == "full":
        st.markdown(
            f"<div style='color:var(--text-secondary); font-size:.8rem; margin-bottom:.2rem;'>"
            f"{fx_text}</div>",
            unsafe_allow_html=True,
        )

    # Driver categories (11/5): the ranked accordion summarising the waterfall
    # by business bucket. It reads the same probability-space view the chart
    # draws, so the numbers always agree.
    render_driver_categories(view)

    # The "SHAP waterfall — driver-level detail" panel (waterfall + price
    # response curve) shows only in Full Simulation (user, 11/18): both chart
    # the win probability itself. Directional Prediction and Customer
    # Intelligence keep the driver categories above.
    if active_view() != "full":
        return

    # The waterfall and price curve sit in a panel that is closed by default,
    # and building them is the slowest part of a run. Draw the panel's shell
    # now and fill it at the end of main() (run_deferred), so the rest of the
    # page, Customer Deep Dive included, is on screen first.
    expander = st.expander("SHAP waterfall — driver-level detail", expanded=False)
    slot = expander.empty()
    slot.caption("Loading driver-level detail…")
    defer(lambda: _draw_shap_detail(
        slot, frame, view, x_base, x_final, axis_title, axis_format, text_format, base_row, overrides
    ))


def _draw_shap_detail(
    slot: Any,
    frame: pd.DataFrame,
    view: pd.DataFrame,
    x_base: float,
    x_final: float,
    axis_title: str,
    axis_format: str,
    text_format: str,
    base_row: pd.Series | None,
    overrides: dict[str, Any] | None,
) -> None:
    """The SHAP waterfall and price-response curve, drawn into the panel's slot."""
    order = frame["axis_label"].tolist()
    scale = alt.Scale(zero=False, nice=True)
    y = alt.Y(
        "axis_label:N",
        title=None,
        sort=order,
        # labelLimit stops Altair truncating to "Relationship stre…";
        # labelOverlap=False forces every feature name to render.
        axis=alt.Axis(labelLimit=340, labelOverlap=False, labelFontSize=12, domain=False, ticks=False),
    )
    colour = alt.Color(
        "direction:N",
        scale=alt.Scale(
            domain=["Increases win", "Decreases win"],
            range=[SHAP_POSITIVE, SHAP_NEGATIVE],
        ),
        # Bottom, so the top margin belongs to the anchor-line labels alone.
        # The data values stay "Increases win"/"Decreases win" (tests and the
        # narrative share them); only the printed legend text is longer.
        legend=alt.Legend(
            orient="bottom",
            title=None,
            labelFontSize=11.5,
            labelExpr=(
                "datum.label == 'Increases win'"
                " ? 'Increases probability of win'"
                " : 'Decreases probability of win'"
            ),
        ),
    )
    tooltip = [
        alt.Tooltip("label:N", title="Feature"),
        alt.Tooltip("value:N", title="Value used"),
        alt.Tooltip("shap:Q", format="+.3f", title="SHAP contribution (log-odds)"),
        alt.Tooltip("end:Q", format="+.3f", title="Running total (log-odds)"),
    ]

    # Start/end anchors (user decision 11/3, "option C3"): the two reference
    # verticals carry their name and value in the top margin, and a soft band
    # spans average -> prediction with the lift labelled, so "started at the
    # average, these drivers moved it here" reads straight off the chart.
    lo = float(view[["start", "end"]].min().min())
    hi = float(view[["start", "end"]].max().max())
    span_x = max(hi - lo, 1e-6)
    # When the two lines sit close together their labels would overlap, so the
    # pair stacks vertically instead and the lift label is dropped (the header
    # line above the chart still shows both numbers).
    stacked = abs(x_final - x_base) / span_x < 0.35

    def _anchor_label_align(x_value: float) -> tuple[str, int]:
        if x_value >= hi - span_x * 0.20:
            return "right", -8
        if x_value <= lo + span_x * 0.20:
            return "left", 8
        return "center", 0

    def _anchor(x_value: float, colour_hex: str, name: str, dashed: bool, dys: tuple[int, int]) -> list[alt.Chart]:
        align, dx = _anchor_label_align(x_value)
        point = pd.DataFrame({"x": [x_value]})
        line = (
            alt.Chart(point)
            .mark_rule(
                color=colour_hex,
                strokeWidth=3 if dashed else 2.5,
                strokeDash=[6, 4] if dashed else [1, 0],
                opacity=0.95,
            )
            .encode(x=alt.X("x:Q", scale=scale))
        )
        marker = (
            alt.Chart(point)
            .mark_point(shape="triangle-down", size=120, filled=True, color=colour_hex, opacity=1)
            .encode(x=alt.X("x:Q", scale=scale), y=alt.value(2))
        )
        texts = [
            alt.Chart(pd.DataFrame({"x": [x_value], "t": [text_value]}))
            .mark_text(
                fontSize=size, fontWeight="bold", align=align, baseline="bottom",
                dy=dy, dx=dx, color=colour_hex,
            )
            .encode(x=alt.X("x:Q", scale=scale), y=alt.value(0), text="t:N")
            for text_value, size, dy in ((name, 12.5, dys[0]), (f"{x_value:.1%}", 11.5, dys[1]))
        ]
        return [line, marker, *texts]

    band = (
        alt.Chart(pd.DataFrame({"x": [min(x_base, x_final)], "x2": [max(x_base, x_final)]}))
        .mark_rect(color=SHAP_ANCHOR_FINAL, opacity=0.06)
        .encode(x=alt.X("x:Q", scale=scale), x2="x2:Q")
    )
    anchors = [
        band,
        *_anchor(x_base, SHAP_ANCHOR_BASE, "Average Win Probability", dashed=True,
                 dys=(-56, -41) if stacked else (-26, -11)),
        *_anchor(x_final, SHAP_ANCHOR_FINAL, "Account Prediction", dashed=False, dys=(-26, -11)),
    ]
    if not stacked:
        anchors.append(
            alt.Chart(
                pd.DataFrame(
                    {"x": [(x_base + x_final) / 2],
                     "t": [f"{(x_final - x_base) * 100:+.1f}pp vs average"]}
                )
            )
            .mark_text(fontSize=11.5, fontWeight="bold", align="center", baseline="bottom",
                       dy=-11, color=SHAP_ANCHOR_FINAL)
            .encode(x=alt.X("x:Q", scale=scale), y=alt.value(0), text="t:N")
        )

    # Each bar spans the running total to the running total plus its contribution.
    bars = (
        alt.Chart(view)
        .mark_bar(height=alt.RelativeBandSize(0.55))
        .encode(
            x=alt.X(
                "start:Q",
                title=axis_title,
                axis=alt.Axis(format=axis_format, grid=True, gridColor="#E6E2F0"),
                scale=scale,
            ),
            x2="end:Q",
            y=y,
            color=colour,
            tooltip=tooltip,
        )
    )
    # Arrowhead at the tip, pointing the way the feature pushed the prediction.
    heads = (
        alt.Chart(view)
        .mark_point(size=150, filled=True, opacity=1)
        .encode(
            x=alt.X("end:Q", scale=scale),
            y=y,
            shape=alt.Shape("arrow:N", scale=None, legend=None),
            color=colour,
            tooltip=tooltip,
        )
    )

    # Three label layers: inside wide bars in white, outside narrow ones in the
    # bar's own colour. Text align is a mark property, not an encoding, so the
    # left- and right-aligned cases have to be separate layers.
    def _text(subset: pd.DataFrame, align: str, fill: str | None) -> alt.Chart:
        mark = dict(fontSize=11, fontWeight="bold", align=align, baseline="middle")
        chart = alt.Chart(subset).mark_text(**mark, color=fill) if fill else alt.Chart(
            subset
        ).mark_text(**mark)
        encoding = dict(
            x=alt.X("label_x:Q", scale=scale),
            y=y,
            text=alt.Text("display_value:Q", format=text_format),
        )
        if fill is None:
            encoding["color"] = colour
        return chart.encode(**encoding)

    layers = [*anchors, bars, heads]
    inside = view[view["inside"]]
    if not inside.empty:
        layers.append(_text(inside, "center", "#FFFFFF"))
    for positive, align in ((True, "left"), (False, "right")):
        subset = view[(~view["inside"]) & ((view["display_value"] >= 0) == positive)]
        if not subset.empty:
            layers.append(_text(subset, align, None))

    # Collapsible (11/5): the categories above carry the summary, so the
    # driver-level waterfall sits in an expander, closed by default (11/12).
    # The price-response curve (11/12, full-visibility persona) sits in the
    # same expander, below the waterfall — see render_price_response_curve().
    with slot.container():
        st.altair_chart(
            alt.layer(*layers)
            .resolve_scale(shape="independent")
            .properties(
                height=44 * len(frame) + 24,
                # Head-room for the anchor labels (two rows; four when stacked).
                padding={"top": 74 if stacked else 44, "left": 5, "right": 5, "bottom": 5},
            ),
            width="stretch",
        )
        # Price-response curve (11/12): same expander, directly below the
        # waterfall. base_row is None only when the caller has no product
        # row to hand it (not expected from render_outcome's own call site,
        # but render_shap_chart is also exercised directly by tests).
        if base_row is not None:
            render_price_response_curve(get_models(), get_metadata(), base_row, overrides or {})


def save_scenario(customer: str, result: dict[str, Any]) -> None:
    """Record a scenario in session state.

    The Saved Scenarios panel that displayed these was removed; the button and
    this store are kept so the feature can be rebuilt without re-deriving what
    a scenario row contains.
    """
    for item in result["product_results"]:
        entry: dict[str, Any] = {
            "Account Name": customer,
            "Product": item["product"],
            "Dead Net Price": round(item["overrides"]["dead_net_price"], 2),
        }
        for config in LEVER_CONFIGS:
            if config["key"] != "dead_net_price":
                entry[config["label"]] = lever_label_for_value(config["key"], item["overrides"].get(config["key"]))
        entry.update(
            {
                "Win": f"{item['win_probability']:.1%}",
                "Threshold": item["threshold"],
                "Predicted": item["predicted_outcome"],
            }
        )
        st.session_state["saved_scenarios"].append(entry)


# --- deep dive --------------------------------------------------------------


# Hover tooltip rather than st.popover: a popover always renders a chevron and
# needs a click, and its own tooltip would only show a label.
WATERFALL_HELP = (
    # div, not span: a <ul> is invalid inside a <span>, so the HTML parser closes
    # the span before the list. The list then escapes .info-tip-body, losing the
    # hidden + absolute styling, and the whole panel renders inline.
    "<div class='info-tip'>"
    "<span class='info-tip-icon'>i</span>"
    "<div class='info-tip-body'>"
    "<ul>"
    "<li><strong>Colour = direction.</strong> Blue bars indicate a contribution "
    "toward Win; red bars indicate a contribution toward Loss. Bar length shows "
    "the magnitude of that contribution.</li>"
    "<li><strong>Order = impact.</strong> Features are sorted top-to-bottom by "
    "their absolute effect on the win-probability score, so the biggest drivers "
    "of this scenario sit at the top.</li>"
    "<li><strong>Y-axis = drivers.</strong> Each label is a driver (feature) "
    "contributing to the prediction for this scenario.</li>"
    "</ul>"
    "</div></div>"
)

# Sits just left of the Log-odds/Probability toggle (see the "shap-view-toggle"
# container in render_customer_deep_dive) — explains what each of the two
# buttons actually shows, since the toggle itself carries no label text
# (label_visibility="collapsed").
SHAP_TOGGLE_HELP = (
    "<div class='info-tip'>"
    "<span class='info-tip-icon'>i</span>"
    "<div class='info-tip-body'>"
    "<ul>"
    "<li><strong>Log-Odds</strong> represents the model's underlying "
    "confidence score. Positive values suggest higher likelihood, while "
    "negative values suggest lower likelihood.</li>"
    "<li><strong>Probability</strong> is the model's estimated chance of "
    "the outcome occurring, expressed as a percentage.</li>"
    "</ul>"
    "</div></div>"
)

# Legend entry for the red dashed line on the dead-net-price chart.
MEDIAN_LEGEND_LABEL = "Product median"

PRODUCT_HEX = {"Viresta": "#7432A8", "Immunexa": "#36A39B", "Noventra": "#F37000"}

PRODUCT_COLOURS = alt.Scale(
    domain=list(PRODUCT_HEX),
    range=list(PRODUCT_HEX.values()),
)


def _active_product_scale(data: pd.DataFrame, series_column: str = "series") -> alt.Scale:
    """Colour scale limited to the brands actually present in ``data``.

    PRODUCT_COLOURS' domain is fixed to all three brands, so Vega-Lite drew
    a legend entry for every brand regardless of the Chart brands filter —
    each brand keeps its usual colour here, the domain is just narrowed to
    what is on the chart, which is what makes the legend follow the filter.
    """
    present = [p for p in PRODUCT_HEX if p in set(data[series_column])]
    return alt.Scale(domain=present, range=[PRODUCT_HEX[p] for p in present])

# Shared by the dose and margin charts so a given party is the same colour in
# both, even though the two charts cover a different set of competitors.
PARTY_COLOURS = alt.Scale(
    domain=["Customer", "Competitor 1", "Competitor 2", "Competitor 3"],
    range=["#7432A8", "#C0392B", "#F37000", "#8D879C"],
)


def deep_dive_frame(customer: str, selected_products: list[str]) -> pd.DataFrame:
    """Season × product aggregate for the selected buying group.

    A buying group can hold several contracts and price groups in a season, so
    volume is summed and the per-dose measures are averaged.
    """
    history = get_history()
    subset = history[
        (history[CUSTOMER_COLUMN] == customer)
        & (history[PRODUCT_COLUMN].isin(selected_products))
    ]
    if subset.empty:
        return subset

    aggregations = {
        "shipments": (SHIPMENTS_COLUMN, "sum"),
        "dead_net_price": ("dead_net_price_per_dose", "mean"),
        "premium_vs_comp1": (PREMIUM_COLUMN, "mean"),
        "market_share": ("ddd_customer_market_share", "mean"),
        "prebook": (TOTAL_PREBOOK_COLUMN, "sum"),
        # Constant per product, so max is just a way to carry it through.
        "product_median_price": ("ref_product_median_price", "max"),
    }
    # Dose series are summed; per-customer they are whole-dose counts.
    for label, column in DDD_DOSE_COLUMNS.items():
        aggregations[f"doses::{label}"] = (column, "sum")
    # Margins are per-dose rates, so they average rather than sum.
    for label, column in MARGIN_COLUMNS.items():
        aggregations[f"margin::{label}"] = (column, "mean")

    frame = (
        subset.groupby(["season", PRODUCT_COLUMN], as_index=False)
        .agg(**aggregations)
        .sort_values(["season", PRODUCT_COLUMN])
    )

    # Shipped-to-prebook ratio, divided after summing rather than taken from the
    # precomputed per-row column. A season with no prebook recorded gives NaN,
    # not a divide-by-zero.
    frame["prebook_ratio"] = frame["shipments"].where(frame["prebook"] > 0) / frame[
        "prebook"
    ].where(frame["prebook"] > 0)
    return frame


def _bar_chart(frame: pd.DataFrame, field: str, title: str, fmt: str) -> alt.Chart:
    data = frame.dropna(subset=[field])
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{field}:Q", title=title, axis=alt.Axis(format=fmt)),
            color=alt.Color(f"{PRODUCT_COLUMN}:N", scale=PRODUCT_COLOURS, legend=alt.Legend(title=None, orient="top")),
            xOffset=f"{PRODUCT_COLUMN}:N",
            tooltip=[
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip(f"{PRODUCT_COLUMN}:N", title="Product"),
                alt.Tooltip(f"{field}:Q", format=fmt, title=title),
            ],
        )
        .properties(height=250)
    )


def _line_chart(frame: pd.DataFrame, field: str, title: str, fmt: str) -> alt.Chart:
    data = frame.dropna(subset=[field])
    base = alt.Chart(data).encode(
        x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{field}:Q", title=title, axis=alt.Axis(format=fmt)),
        color=alt.Color(f"{PRODUCT_COLUMN}:N", scale=PRODUCT_COLOURS, legend=alt.Legend(title=None, orient="top")),
        tooltip=[
            alt.Tooltip("season:N", title="Season"),
            alt.Tooltip(f"{PRODUCT_COLUMN}:N", title="Product"),
            alt.Tooltip(f"{field}:Q", format=fmt, title=title),
        ],
    )
    return (base.mark_line(strokeWidth=2.5) + base.mark_point(size=70, filled=True)).properties(height=250)


def _empty_note(label: str) -> None:
    st.markdown(
        f"<div class='section-subtitle'>No {escape(label)} recorded for this account.</div>",
        unsafe_allow_html=True,
    )


def _series_chart(
    frame: pd.DataFrame,
    title: str,
    fmt: str,
    colours: alt.Scale,
    labels: bool = False,
) -> alt.Chart:
    """Grouped bars over season, one colour per series (product or competitor).

    ``labels`` prints the value above each bar. Used for the shipped-to-prebook
    ratio, where one outlying season sets the axis and the remaining bars are
    only a few pixels tall — the number stays readable even when the bar does
    not.
    """
    base = alt.Chart(frame).encode(
        x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=title, axis=alt.Axis(format=fmt)),
        xOffset="series:N",
        tooltip=[
            alt.Tooltip("season:N", title="Season"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", format=fmt, title=title),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("series:N", scale=colours, legend=alt.Legend(title=None, orient="top")),
    )
    if not labels:
        return bars.properties(height=260)

    text = base.mark_text(
        dy=-7, fontSize=10, fontWeight="bold", color="#1F1F29"
    ).encode(text=alt.Text("value:Q", format=fmt))
    return (bars + text).properties(height=260)


def _long(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    """Reshape one metric column into season / series / value, series = product."""
    out = frame[["season", PRODUCT_COLUMN, field]].dropna(subset=[field]).copy()
    out = out.rename(columns={PRODUCT_COLUMN: "series", field: "value"})
    return out


def _party_long(frame: pd.DataFrame, prefix: str, labels: list[str], how: str) -> pd.DataFrame:
    """Reshape the Customer-vs-competitor metrics into season / series / value.

    Series are collapsed across products because these columns are stated at the
    buying-group level, not per brand. A competitor is dropped entirely when this
    customer has nothing recorded for it, so neither chart carries empty bars —
    the same rule for doses and for margins, since the two extracts differ in
    which competitors they cover.
    """
    rows = []
    for label in labels:
        column = f"{prefix}::{label}"
        if column not in frame:
            continue
        block = frame[["season", column]].dropna(subset=[column])
        if block.empty:
            continue
        per_season = getattr(block.groupby("season", as_index=False)[column], how)()
        if float(per_season[column].abs().sum()) == 0.0 and label != "Customer":
            continue
        for _, entry in per_season.iterrows():
            rows.append(
                {"season": entry["season"], "series": label, "value": float(entry[column])}
            )
    return pd.DataFrame(rows)


def _delta_parts(delta: float | None, formatter) -> tuple[str, str, str]:
    """(css class, arrow, text) for a year-on-year change.

    Colour is purely directional: increase green, decrease red, applied the same
    way to every metric including price.
    """
    if delta is None or pd.isna(delta):
        return "kpi-delta--flat", "", "no prior year"
    value = float(delta)
    if abs(value) < 1e-9:
        return "kpi-delta--flat", "", "no change"
    arrow = "&#9650;" if value > 0 else "&#9660;"
    css = "kpi-delta--up" if value > 0 else "kpi-delta--down"
    return css, arrow, formatter(abs(value))


def _bold_first_sentence_html(text: str) -> str:
    """Escapes ``text`` and bolds its first sentence (up to the first
    ". "), so e.g. a quadrant's "Customer incumbent x Not price-sensitive."
    header reads distinctly from the explanation that follows it — that
    header differs per quadrant, so it can't just be a fixed heading."""
    marker = ". "
    if marker in text:
        first, rest = text.split(marker, 1)
        return f"<strong>{escape(first)}.</strong> {escape(rest)}"
    return f"<strong>{escape(text)}</strong>"


def _hover_tip_html(content_html: str, tooltip_html: str | None, extra_class: str = "") -> str:
    """Wraps ``content_html`` so hovering it shows a styled white tooltip
    box (the app's own .info-tip/.info-tip-body look) instead of the plain
    OS-native title="" tooltip, which reads as an afterthought next to the
    rest of the app's styling. ``tooltip_html`` is already escaped/marked up
    by the caller (see _bold_first_sentence_html), not plain text."""
    if not tooltip_html:
        return content_html
    tip_class = f"info-tip kpi-value-tip {extra_class}".strip()
    return (
        f"<span class='{tip_class}'>{content_html}"
        f"<div class='info-tip-body'>{tooltip_html}</div>"
        f"</span>"
    )


def _delta_inline(delta: float | None, formatter, tooltip: str | None = None) -> str:
    """Compact YoY that sits on the same line as the value."""
    css, arrow, text = _delta_parts(delta, formatter)
    prefix = f"{arrow} " if arrow else ""
    content = f"<span class='kpi-delta-inline {css}'>{prefix}{escape(text)}</span>"
    return _hover_tip_html(content, escape(tooltip) if tooltip else None)


def _delta_html(delta: float | None, formatter) -> str:
    """Block YoY beneath a single headline value."""
    css, arrow, text = _delta_parts(delta, formatter)
    prefix = f"{arrow} " if arrow else ""
    suffix = " vs prior year" if arrow else ""
    return f"<div class='kpi-delta {css}'>{prefix}{escape(text)}{suffix}</div>"


def _product_rows_html(
    product_rows,
    value_of,
    delta_of=None,
    title_of=None,
    bold_title_first_sentence=False,
    title_extra_class="",
) -> str:
    """One line per product: short name left, value and its YoY delta right.

    ``title_of``, when given, wraps the value in a hover tooltip styled like
    the app's other tooltips (e.g. what a Strategic Quadrant label means)
    rather than a plain OS-native title="". ``bold_title_first_sentence``
    bolds that tooltip's first sentence (its per-quadrant header) instead of
    rendering the whole thing as one plain block. ``title_extra_class`` adds
    a CSS class to the tooltip wrapper, e.g. to widen it for a longer body.
    """
    out = ""
    for product, row in product_rows.items():
        delta = "" if delta_of is None else delta_of(row)
        tooltip_text = title_of(row) if title_of is not None else None
        if not tooltip_text:
            tooltip_html = None
        elif bold_title_first_sentence:
            tooltip_html = _bold_first_sentence_html(tooltip_text)
        else:
            tooltip_html = escape(tooltip_text)
        value_html = _hover_tip_html(escape(value_of(row)), tooltip_html, extra_class=title_extra_class)
        out += (
            f"<div class='kpi-row'>"
            f"<span class='kpi-row-name'>{escape(str(product))}</span>"
            f"<span class='kpi-row-right'>"
            f"<span class='kpi-row-value'>{value_html}</span>"
            f"{delta}</span></div>"
        )
    return out


def _kpi_card(column, label: str, body: str, foot: str, css: str, icon: str = "") -> None:
    icon_html = f"<span class='kpi-label-icon'>{icon}</span>" if icon else ""
    column.markdown(
        f"<div class='kpi-card {css}'>"
        f"<div class='kpi-label-row'>{icon_html}<span class='kpi-label'>{escape(label)}</span></div>"
        f"{body}"
        f"<div class='kpi-foot'>{foot}</div></div>",
        unsafe_allow_html=True,
    )


def _num(row: pd.Series, column: str) -> float | None:
    """Numeric value or None, so a missing column blanks a line rather than raising."""
    if column not in row.index:
        return None
    value = row[column]
    return None if value is None or pd.isna(value) else float(value)


def _pct_change_title(current: float | None, prior: float | None) -> str | None:
    """"X% increase/decrease vs prior year" for a KPI card's delta hover, or
    None when there is nothing to compare against (matches the Price / Dose
    card's own price_pct_title, generalised for Shipments and Prebooks)."""
    if current is None or prior is None or prior == 0:
        return None
    pct = (current - prior) / prior * 100
    if abs(pct) < 1e-9:
        return "No change vs prior year"
    direction = "increase" if pct > 0 else "decrease"
    return f"{abs(pct):.1f}% {direction} vs prior year"


# What each of the four quadrant_2x2 labels means, shown as a hover on the
# Strategic Quadrant KPI card's value — the client's own wording.
QUADRANT_DESCRIPTIONS = {
    "Protect Margin": (
        "Customer incumbent × Not price-sensitive. Accounts we already hold whose win "
        "probability barely moves with price. The relationship, not the price, is "
        "carrying the account. Implication: these are the safest accounts for price "
        "discipline or upward movement - deep discounts here give away margin "
        "without buying any additional security."
    ),
    "Targeted Retention": (
        "Customer incumbent × Price-sensitive. Accounts we currently hold but where the "
        "model says a price increase meaningfully raises the risk of losing them. The "
        "win is real but fragile on price. Implication: defend selectively - hold "
        "price steady or concede tactically where volume justifies it, and reinforce "
        "with non-price value; these are the accounts where an ill-timed increase "
        "turns a held account into a loss."
    ),
    "Build Value & Relationship": (
        "Competitor incumbent × Not price-sensitive. Accounts a competitor holds, "
        "where cutting our price alone won't flip the outcome - the barrier is "
        "relationship, clinical preference, or contracting structure, not economics. "
        "Implication: don't lead with discounts; invest in engagement, clinical "
        "evidence, and account development to change the fundamentals first."
    ),
    "Competitive Price Focus": (
        "Competitor incumbent × Price-sensitive. Accounts a competitor holds where "
        "price genuinely moves the win probability - the model's capture "
        "opportunities. Implication: these are the accounts where an aggressive "
        "offer can actually convert; prioritize them for competitive pricing plays, "
        "sized against the volume at stake."
    ),
}


def render_kpi_cards(product_rows: dict[str, pd.Series]) -> None:
    """Seven KPI cards, every one stated per product, across two rows:

        Row 1 (4 cards): Price / Dose, Strategic Quadrant, TAM, Price Sensitivity
        Row 2 (3 cards): Shipments, Prebooks, Market Share

    The former second row (Path of Advocacy, Segmentation) is gone — advocacy
    moved into the Customer Context signal row and segmentation was dropped.
    """
    row1 = st.columns(4, gap="small")
    st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)
    # 4 columns here too (not 3) so row 2's cards stay the same fixed width as
    # row 1's — the 4th slot is just left unfilled, not a card.
    row2 = st.columns(4, gap="small")

    # SEASON is e.g. "2025-2026"; the KPI foot captions used to say "Current
    # year vs prior year", which stops meaning anything once the tool holds
    # more than two seasons of data. Spelling out the actual years keeps them
    # unambiguous as more seasons are added.
    current_year_label = SEASON.split("-")[0]
    prior_year_label = str(int(current_year_label) - 1)
    year_caption = f"{current_year_label} vs {prior_year_label}"

    # --- 1. Price / dose ----------------------------------------------------
    def price_pct_title(row: pd.Series) -> str | None:
        yoy, prior = _num(row, PRICE_YOY_COLUMN), _num(row, "py_dead_price_customer")
        if yoy is None or prior is None or prior == 0:
            return None
        pct = yoy / prior * 100
        if abs(pct) < 1e-9:
            return "No change vs prior year"
        direction = "increase" if pct > 0 else "decrease"
        return f"{abs(pct):.1f}% {direction} vs prior year"

    _kpi_card(
        row1[0],
        "Price / Dose",
        _product_rows_html(
            product_rows,
            lambda r: format_value(r["dead_net_price_per_dose"], "currency"),
            lambda r: _delta_inline(
                _num(r, PRICE_YOY_COLUMN), lambda v: f"${v:,.2f}", tooltip=price_pct_title(r)
            ),
        ),
        f"Net Price | {year_caption}",
        "kpi-card--purple",
        KPI_ICONS["price"],
    )

    # --- 2. Quadrant ---------------------------------------------------------
    # The volume-weighted quadrant comes from the scoring model rather than the
    # placeholder this card used to show. It is assigned per row, and 197 of the
    # 344 customer/price-group pairs span more than one quadrant, so it is
    # stated per product like the other brand-varying cards instead of being
    # collapsed to a single value.
    #
    # Raw text, not escaped here: one of the four values is "Build Value &
    # Relationship", and _product_rows_html() already escapes value_of(row)
    # for every card — escaping here too double-encodes the "&" to "&amp;amp;".
    def quadrant_of(row: pd.Series) -> str:
        value = row.get(QUADRANT_COLUMN)
        return "n/a" if value is None or pd.isna(value) else str(value).strip()

    _kpi_card(
        row1[1],
        "Strategic Quadrant",
        _product_rows_html(
            product_rows,
            quadrant_of,
            title_of=lambda r: QUADRANT_DESCRIPTIONS.get(quadrant_of(r)),
            bold_title_first_sentence=True,
            title_extra_class="kpi-value-tip--quadrant",
        ),
        "",
        "kpi-card--purple",
        KPI_ICONS["quadrant"],
    )

    # --- 3. TAM (Total Addressable Market) -----------------------------------
    # Doses and dollars both live in this one card, on the same line per
    # product — doses as the headline value, dollars as the plain (non-delta)
    # secondary figure next to it. From TAM_XLSX via merge_tam() in
    # src/prepare_real_data.py; ~20% of rows have neither value in the
    # source, which renders as "n/a" rather than a guessed figure.
    def tam_doses_of(row: pd.Series) -> str:
        text = _signal_text(row.get("tam_doses"), "compact")
        return text if text else "n/a"

    def tam_dollars_of(row: pd.Series) -> str:
        text = _signal_text(row.get("tam_dollars"), "compact")
        return "" if not text else f"<span class='kpi-delta-inline kpi-delta--flat'>${escape(text)}</span>"

    _kpi_card(
        row1[2],
        "TAM",
        _product_rows_html(product_rows, tam_doses_of, tam_dollars_of),
        "Doses | Market Value ($)",
        "kpi-card--purple",
        KPI_ICONS["tam"],
    )

    # --- 4. Price Sensitivity -------------------------------------------------
    # Per-brand, unlike the Price sensitivity paragraph in the Narrative
    # Summary, whose band (account_sensitivity_band) is account-level and
    # repeats across every brand for that account.
    #
    # Standard-benchmark rule: banded on how far win probability moves when
    # the dead-net price goes to the product's standard benchmark, against
    # that product's volume-weighted cutoffs (they differ per product, so the
    # hover spells out the pp behind the band and the product's own cuts).
    def sensitivity_band_of(row: pd.Series) -> str:
        value = row.get(SENSITIVITY_BAND_LMH_COLUMN)
        return "n/a" if value is None or pd.isna(value) else str(value).strip()

    def sensitivity_basis_of(row: pd.Series) -> str | None:
        value = _num(row, "sensitivity_benchmark_pp")
        bench = _num(row, "standard_price_benchmark")
        low, high = _num(row, "sensitivity_low_cutoff_pp"), _num(row, "sensitivity_high_cutoff_pp")
        if value is None or bench is None:
            return None
        change = _num(row, "benchmark_price_change_pct") or 0.0
        if row.get("benchmark_direction") == "at benchmark":
            text = f"Already at the ${bench:,.2f} standard benchmark, so no price move (0pp)."
        elif change > 0:
            text = (f"Raising price {change:.1f}% to the ${bench:,.2f} standard benchmark "
                    f"lowers win probability by {value:.1f}pp.")
        else:
            text = (f"Lowering price {abs(change):.1f}% to the ${bench:,.2f} standard benchmark "
                    f"raises win probability by {value:.1f}pp.")
        if low is not None and high is not None:
            text += f" Cutoffs for this brand: Low <{low:.1f}pp | High ≥{high:.1f}pp."
        return text

    _kpi_card(
        row1[3],
        "Price Sensitivity",
        _product_rows_html(product_rows, sensitivity_band_of, title_of=sensitivity_basis_of),
        "Win-prob change at Standard Benchmark",
        "kpi-card--purple",
        KPI_ICONS["sensitivity"],
    )

    # --- 5. Shipments -------------------------------------------------------
    def shipment_delta(row: pd.Series) -> str:
        current, prior = _num(row, CURRENT_VOLUME_COLUMN), _num(row, PRIOR_VOLUME_COLUMN)
        delta = None if current is None or prior is None else current - prior
        return _delta_inline(delta, lambda v: f"{v:,.0f}", tooltip=_pct_change_title(current, prior))

    _kpi_card(
        row2[0],
        "Shipments",
        _product_rows_html(
            product_rows,
            lambda r: "n/a" if _num(r, CURRENT_VOLUME_COLUMN) is None
            else f"{_num(r, CURRENT_VOLUME_COLUMN):,.0f}",
            shipment_delta,
        ),
        f"Doses | {year_caption}",
        "kpi-card--purple",
        KPI_ICONS["shipments"],
    )

    # --- 6. Prebooks --------------------------------------------------------
    def prebook_delta(row: pd.Series) -> str:
        current, prior = _num(row, TOTAL_PREBOOK_COLUMN), _num(row, PRIOR_PREBOOK_COLUMN)
        delta = None if current is None or prior is None else current - prior
        return _delta_inline(delta, lambda v: f"{v:,.0f}", tooltip=_pct_change_title(current, prior))

    _kpi_card(
        row2[1],
        "Prebooks",
        _product_rows_html(
            product_rows,
            lambda r: "n/a" if _num(r, TOTAL_PREBOOK_COLUMN) is None
            else f"{_num(r, TOTAL_PREBOOK_COLUMN):,.0f}",
            prebook_delta,
        ),
        f"Doses | {year_caption}",
        "kpi-card--purple",
        KPI_ICONS["prebooks"],
    )

    # --- 7. Market share ----------------------------------------------------
    _kpi_card(
        row2[2],
        "Market Share",
        _product_rows_html(
            product_rows,
            lambda r: "n/a" if _num(r, CURRENT_SHARE_COLUMN) is None
            else f"{_num(r, CURRENT_SHARE_COLUMN):.1%}",
            lambda r: _delta_inline(
                _num(r, SHARE_CHANGE_COLUMN), lambda v: f"{v * 100:,.1f}pp"
            ),
        ),
        f"% Change | {year_caption}",
        "kpi-card--purple",
        KPI_ICONS["share"],
    )

    # The former second row (Path of Advocacy, Segmentation) is gone — advocacy
    # moved into the Customer Context signal row and segmentation was dropped.
    # Their columns are still written to the parquet, so restoring any of them
    # is a UI-only change.


@st.cache_data(show_spinner=False)
def driver_ranges() -> dict[str, tuple[float, float]]:
    """Population range for each model feature, used to place the track markers.

    The 2nd and 98th percentiles rather than min/max: the prebook ratio has
    extreme outliers, and against a true maximum every ordinary account would
    sit pinned at the left edge.
    """
    data = get_features()
    ranges = {}
    for name in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(data[name], errors="coerce").dropna()
        if values.empty:
            ranges[name] = (0.0, 1.0)
            continue
        low, high = float(values.quantile(0.02)), float(values.quantile(0.98))
        if high <= low:
            low, high = float(values.min()), float(values.max())
        ranges[name] = (low, high) if high > low else (low, low + 1.0)
    return ranges


# Tile kinds that get a mini range bar: numeric values only. Text, yes/no
# and band tiles have no scale to show. A tile opts out with
# "range_bar": False (user, 11/11: dead-net / CY / PY price, cheapest
# competitor price, premium vs cheapest, market share change and prior year
# Customer doses) — per tile, not per column.
RANGE_BAR_KINDS = {"currency", "compact", "num2", "pct1", "pct1signed", "pts1signed", "score100"}


def _has_range_bar(tile: dict[str, Any]) -> bool:
    return tile.get("kind") in RANGE_BAR_KINDS and tile.get("range_bar", True)


@st.cache_data(show_spinner=False)
def signal_ranges() -> dict[str, dict[str, tuple[float, float, int]]]:
    """Per brand, (min, max, n) of every numeric Account signals
    column across the shown season's accounts — the scale for the tiles'
    mini range bars. True min and max (not percentiles), as requested;
    missing values are left out of n."""
    frame = get_features()
    columns = {
        tile["column"]
        for group in SIGNAL_GROUPS
        for tile in group["tiles"]
        if _has_range_bar(tile) and tile.get("column") in frame.columns
    }
    ranges: dict[str, dict[str, tuple[float, float, int]]] = {}
    for product, subset in frame.groupby(PRODUCT_COLUMN):
        ranges[str(product)] = {}
        for name in columns:
            values = pd.to_numeric(subset[name], errors="coerce").dropna()
            if not values.empty:
                ranges[str(product)][name] = (float(values.min()), float(values.max()), int(len(values)))
    return ranges


CLINICAL_BUCKET_DISPLAY = {
    "clinically_inclined": "Clinically Inclined",
    "price_sensitive": "Non-Clinically Inclined",
}


def _signal_text(value: Any, kind: str) -> str:
    """One tile's value, formatted for its kind. Empty values return ''."""
    if kind == "blank":
        return ""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if kind == "text":
        text = str(value).strip()
        return "" if text.lower() in {"no data", "unknown", "nan", "-", ""} else text
    if kind == "titlecase":
        text = str(value).strip()
        return "" if text.lower() in {"unknown", "nan", ""} else text.replace("_", " ").title()
    if kind == "clinical_bucket":
        # Display-only relabel (10/8): the source's "price_sensitive" reads as
        # if it were the price-sensitivity band, so it is shown as the
        # opposite of clinically inclined instead. Column value untouched.
        text = str(value).strip().lower()
        if text in {"unknown", "nan", ""}:
            return ""
        return CLINICAL_BUCKET_DISPLAY.get(text, text.replace("_", " ").title())
    if kind == "negotiation_stance":
        # negotiation_stance_bucket's raw "hard" reads as data-speak, not
        # what the client calls it — display-only relabel, the underlying
        # column value is untouched.
        text = str(value).strip().lower()
        if text in {"unknown", "nan", ""}:
            return ""
        return narrative_technical.NEGOTIATION_STANCE_DISPLAY.get(text, text.replace("_", " ").title())

    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    number = float(number)

    if kind == "score100":
        return f"{number * 100:.0f}"
    if kind == "num2":
        return f"{number:.2f}"
    if kind == "currency":
        return f"${number:,.2f}"
    if kind == "pct1":
        return f"{number:.1f}%"
    if kind == "pct1signed":
        return f"{number:+.1f}%"
    if kind == "pts1signed":
        # Percentage points (e.g. a market-share change), not a percent change.
        return f"{number:+.1f} pts"
    if kind == "pctfrac":
        # Stored as a fraction in the source: 0.15 renders as 15%.
        return f"{number * 100:.0f}%"
    if kind == "pctfracsigned":
        # Same fraction storage as pctfrac, but the sign is meaningful
        # (e.g. share_change can be a loss), so it is shown explicitly.
        return f"{number * 100:+.1f}%"
    if kind == "yesno":
        return "Yes" if number >= 0.5 else "No"
    if kind == "risk":
        # 0 / 0.5 / 1 in the source. Anything else is treated as a band edge.
        return "Low" if number <= 0.25 else ("Moderate" if number < 0.75 else "High")
    if kind == "compact":
        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        if abs(number) >= 1_000:
            return f"{number / 1_000:.0f}K"
        return f"{number:,.0f}"
    if kind == "position":
        # price_vs_std_benchmark_updated = 100*(bench-price)/bench: positive
        # means the offered price sits BELOW the benchmark (cheaper), the
        # opposite sign convention from the prior model's price_dev feature.
        if number > 0:
            return "Below Benchmark"
        return "Above Benchmark" if number < 0 else "At Benchmark"
    return _trim_number(number)


# One short caption per driver, shown under its name in the row header.
DRIVER_SIGNAL_SUBTITLES = {
    "Competitive Pressure": "Intensity of competitive activity",
    "Price vs. Standard Benchmark Offer": "Vs. standard pricing benchmark",
    "Relationship Strength": "Prior customer volume presence",
    "Share of Wallet": "Prior-year wallet-share position",
    "Clinical Focus": "Level of clinical engagement",
    "Field Engagement Health": "Overall engagement quality",
    "SAM Risk Flag": "Account risk from the SAM survey",
    "Year-over-year Price Change": "Change in price vs last year",
    "Provider Margin": "Margin per dose administered",
    "Prebook Conversion Ratio": "Conversion of prebook to doses",
    "Price vs. Cheapest Competitor Price": "Price premium vs cheapest option",
    "Negotiation Receptivity": "Willingness to negotiate",
    "Buying Group": "Buying group the contract sits under",
    "Multi-year Contract": "Contract term commitment",
    # Kept for compatibility with older saved scenarios (row split 11/6 into
    # Buying Group and Multi-year Contract).
    "Contract Structure": "Buying group and contract term",
}

# All 10 model-input driver tiles get the orange treatment plus a hover
# tooltip describing what the tile is derived from and the exact raw
# columns its formula reads (per the modelling team's formula table).
def _driver_tooltip_html(measure: str, key_inputs: list[str]) -> str:
    """Business-facing tooltip body: what the driver measures, then its raw
    "Key inputs include" list, verbatim per the client's 9/11 revision — see
    DRIVER_SIGNAL_TOOLTIPS. Built as raw HTML (not escape()'d by the caller)
    since a <ul> needs real markup, not escaped text — see _signal_head_html
    and _lever_metric_label_html, which insert this unescaped."""
    items = "".join(f"<li>{item}</li>" for item in key_inputs)
    return (
        f"<p style='margin:0;'>{measure}</p>"
        f"<p style='margin:.5rem 0 .25rem 0;'><strong>Key inputs include:</strong></p>"
        f"<ul>{items}</ul>"
    )


# What each driver means and what it is built from, in business language —
# the client's own wording (9/11 revision), not the modelling team's formula
# (that is DRIVER_SIGNAL_FORMULAS below, unchanged by this revision).
DRIVER_SIGNAL_TOOLTIPS = {
    "Competitive Pressure": _driver_tooltip_html(
        "Measures how contested the account appears to be. Higher scores indicate "
        "stronger competitive pressure based on supplier-flip history, market "
        "share trend, and prior-year price movement.",
        [
            "Flip history - how many times the account switched suppliers over the "
            "previous two seasons (the current season is not counted).",
            "Market share change - the change in market share between the previous "
            "two seasons, in percentage points.",
            "Year-over-year price change",
        ],
    ),
    "SAM Risk Flag": _driver_tooltip_html(
        "Indicates whether the account has been marked as high or medium risk "
        "for this brand in the SAM survey.",
        [
            "SAM survey risk flag - product-level where surveyed (shown as "
            "&mdash; where the account was not surveyed).",
        ],
    ),
    "Price vs. Standard Benchmark Offer": _driver_tooltip_html(
        "Measures the account's dead-net price relative to our standard offer "
        "benchmark for the season. Positive values mean the account is priced below the "
        "benchmark; negative values mean it is priced above the benchmark. Prices outside "
        "the season's normal range are replaced with the season's standard offer price.",
        ["Standard benchmark offer", "Dead-net price"],
    ),
    "Relationship Strength": _driver_tooltip_html(
        "Measures the depth of the existing relationship with the customer. "
        "Higher values reflect stronger prior volume presence.",
        ["Prior-year doses"],
    ),
    "Share of Wallet": _driver_tooltip_html(
        "Measures how much of the account's prior-year flu-vaccine purchasing "
        "sat with the customer. Stronger wallet positions (Customer Loyalist, Customer "
        "Leaning) indicate a more committed account.",
        [
            "Prior-year wallet share - possible values, strongest to weakest: "
            "<strong>Customer Loyalist</strong>, <strong>Customer Leaning</strong>, "
            "<strong>Competitive Leaning</strong>, <strong>Competitive "
            "Loyalist</strong> (shown as No Data where not recorded).",
        ],
    ),
    "Clinical Focus": _driver_tooltip_html(
        "Measures whether the account appears more clinically inclined in its "
        "vaccine decision-making. Higher scores reflect stronger clinical "
        "classification and partnership advocacy inputs.",
        [
            "Clinical bucket - clinical inclination based on PSC/APT data. Possible "
            "values: <strong>Clinically Inclined</strong> or <strong>Non-Clinically "
            "Inclined</strong> (shown as &mdash; where unknown).",
            "Partnership advocacy - possible values, strongest to weakest: "
            "<strong>Advocate</strong>, <strong>Operationalize</strong>, "
            "<strong>Prioritize</strong>, <strong>Choose</strong> (shown as "
            "&mdash; where not recorded).",
        ],
    ),
    "Field Engagement Health": _driver_tooltip_html(
        "Measures the recent quality and strength of our engagement with the "
        "account. Higher scores indicate healthier recent engagement.",
        [
            "Call engagement score (it incapsulates the number of call, topics "
            "discussed, title of customer attending the calls etc)",
            "Overall call sentiment (positive value indicates a favourable "
            "engagement)",
        ],
    ),
    "Year-over-year Price Change": _driver_tooltip_html(
        "Measures the change in the account's dead-net price versus the prior "
        "year, in percent. Positive values indicate the current price is above "
        "previous year's price; negative values indicate it is below previous "
        "year's price. It moves with the Dead Net Price lever in simulations.",
        ["Current-year dead-net price", "Prior-year dead-net price"],
    ),
    "Provider Margin": _driver_tooltip_html(
        "Measures the provider's estimated per-dose margin on our "
        "product, a sign of provider economic headroom. Shown for account "
        "context; it is not an input to the current model.",
        ["CMS reimbursement", "Our list price"],
    ),
    "Prebook Conversion Ratio": _driver_tooltip_html(
        "Compares actual doses purchased with the contract's "
        "current-year prebook volume. Higher values indicate stronger "
        "conversion from prebooked commitment to actual doses.",
        ["Shipped doses", "Total contract prebook volume"],
    ),
    "Price vs. Cheapest Competitor Price": _driver_tooltip_html(
        "Measures how much our price is above or below the cheapest "
        "competitor's price for the account. Positive values mean we are "
        "priced above the cheapest competitor; negative values mean we are "
        "priced below it.",
        [
            "Our dead-net price",
            "Cheapest competitor price - Competitor A for Viresta and Noventra; the cheaper of "
            "Competitor B and Competitor C for Immunexa",
        ],
    ),
    "Negotiation Receptivity": _driver_tooltip_html(
        "Measures the account's expected openness to price negotiations "
        "based on the recorded negotiation stance from contracting inputs. "
        "Favorable = 100, Neutral = 50, Unfavorable = 0. Higher scores indicate "
        "the account is more likely to engage constructively in negotiation, "
        "not necessarily that it is more price sensitive.",
        ["Presence in PSC negotiations.", "Accounts negotiation posture from APT data."],
    ),
    "Buying Group": _driver_tooltip_html(
        "The buying group the account's contract sits under — a model input "
        "(read as a category), not a lever.",
        ["Buying group - Vizient, HealthTrust, Captis, Premier or Other"],
    ),
    "Multi-year Contract": _driver_tooltip_html(
        "Whether the account is on a multi-year contract — a model input, not "
        "a lever.",
        ["Multi-year contract - Yes or No"],
    ),
    # Kept for compatibility (row split 11/6 into Buying Group and Multi-year
    # Contract).
    "Contract Structure": _driver_tooltip_html(
        "How the account buys: the buying group its contract sits under, and "
        "whether it is on a multi-year contract. Both are model inputs; neither "
        "is a lever.",
        [
            "Buying group - Vizient, HealthTrust, Captis, Premier or Other",
            "Multi-year contract - Yes or No",
        ],
    ),
}

# The modelling team's own formula for each driver, verbatim, shown behind
# the "i" icon at the right edge of the row — separate from
# DRIVER_SIGNAL_TOOLTIPS, which explains what the driver means, not how it
# is computed.
DRIVER_SIGNAL_FORMULAS = {
    # No formula on the four composite rows (Competitive Pressure,
    # Relationship Strength, Clinical Focus, Field Engagement Health), per the
    # 10/8 request: their components feed the model individually, so there is
    # no combined score to give a formula for — the tooltip's "Key inputs
    # include" list already names them.
    "Price vs. Standard Benchmark Offer": (
        "100 x (standard price benchmark - dead-net price) / standard price benchmark "
        "(positive = priced below benchmark); 2027-28 benchmark $60.63 Viresta / Noventra, "
        "$13.77 Immunexa"
    ),
    # v27: a percent change (earlier models used the $ delta CY - PY).
    "Year-over-year Price Change": "yoy_price_change_pct = 100 x (CY Price - PY Price) / PY Price",
    "Provider Margin": "provider_margin = CMS reimbursement - our list price ($ per dose)",
    "Prebook Conversion Ratio": (
        "doses_to_prebook_ratio = contract-level doses / contract "
        "total prebook"
    ),
    "Price vs. Cheapest Competitor Price": "100 x (dead-net price / cheapest competitor price - 1)",
    "Negotiation Receptivity": "Negotiation stance: favourable 1 / neutral 0.5 / unfavourable 0",
}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def _signal_info_icon_html(title: str) -> str:
    """The "i" icon at a row's right edge shows everything about that driver
    in one place: what it measures, its "Key inputs include" list
    (DRIVER_SIGNAL_TOOLTIPS), and the modelling team's own formula
    (DRIVER_SIGNAL_FORMULAS), verbatim. This used to be split between a
    hover on the driver name (the measure/key-inputs text) and this icon
    (the formula only); it is now all behind this one icon.

    Hover previews the panel as usual; a click additionally pins it open
    (a hidden checkbox — the standard CSS "checkbox hack") so it survives
    the cursor leaving the icon. A full-viewport label tied to the same
    checkbox sits behind the panel and unpins/closes it on any click
    outside — clicking the icon again also closes it, since both labels
    toggle the one checkbox.
    """
    tooltip = DRIVER_SIGNAL_TOOLTIPS.get(title, "")
    formula = DRIVER_SIGNAL_FORMULAS.get(title)
    if not tooltip and not formula:
        return ""
    formula_html = (
        f"<p style='margin:.5rem 0 0 0;'><strong>Formula:</strong> {escape(formula)}</p>"
        if formula
        else ""
    )
    input_id = f"signal-tip-{_slug(title)}"
    return (
        f"<div class='signal-row-formula'>"
        f"<div class='info-tip info-tip--click'>"
        f"<input type='checkbox' class='info-tip-toggle' id='{input_id}'>"
        f"<label for='{input_id}' class='info-tip-backdrop'></label>"
        f"<label for='{input_id}' class='info-tip-icon'>i</label>"
        f"<div class='info-tip-body info-tip-body--formula'>"
        f"<strong>{escape(title)}</strong>"
        f"<div style='margin-top:.35rem;'>{tooltip}</div>"
        f"{formula_html}"
        f"</div>"
        f"</div>"
        f"</div>"
    )


def _signal_direction(tile: dict[str, Any], row: pd.Series) -> str | None:
    """Which way this tile's value leans for Customer: "positive", "negative" or
    None (no clear direction, or missing). Drives the tile colour (user
    request 11/3): the values read like the SHAP chart — blue helps the win,
    red hurts it — and undirected tiles keep the default teal. The rules
    follow the model's own encodings and monotone constraints; a tile without
    a "direction" key (benchmark, PY price, competitor price, buying group,
    multi-year, engagement score) has no defensible lean, so it stays teal."""
    rule = tile.get("direction")
    if rule is None:
        return None

    def _num(column: str | None) -> float | None:
        value = pd.to_numeric(row.get(column), errors="coerce")
        return None if pd.isna(value) else float(value)

    number = _num(tile.get("column"))
    raw = row.get(tile.get("column"))
    text = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip().lower()

    if rule == "bad_if_yes":
        return None if number is None else ("negative" if number >= 0.5 else "positive")
    if rule == "zero_good":
        return None if number is None else ("negative" if number > 0 else "positive")
    if rule == "nonzero_good":
        return None if number is None else ("positive" if number > 0 else "negative")
    if rule in ("higher_better", "lower_better"):
        # A value that rounds to 0 at the tile's displayed precision (1
        # decimal for the signed kinds, 2 for num2) shows as "+0.0"/"-0.0"
        # and must not pick a side.
        epsilon = 0.05 if tile.get("kind") in ("pts1signed", "pct1signed") else 0.005
        if number is None or abs(number) < epsilon:
            return None
        better = number > 0 if rule == "higher_better" else number < 0
        return "positive" if better else "negative"
    if rule in ("below_benchmark", "below_py"):
        # Same half-cent tolerance as build_feature_row(): prices carry up to
        # 4 decimals but display 2, so "equal as shown" must not pick a side.
        other = _num("standard_price_benchmark" if rule == "below_benchmark" else "py_dead_price_customer")
        if number is None or other is None or abs(number - other) < 0.005:
            return None
        return "positive" if number < other else "negative"
    if rule == "wallet_share":
        if text.startswith("customer"):
            return "positive"
        if text.startswith("competitive"):
            return "negative"
        return None
    if rule == "advocacy":
        # Encodings: Advocate 1 / Operationalize .75 / Prioritize .5 /
        # Choose .25 / No Data 0 — the midpoint stays neutral.
        if text in ("advocate", "operationalize"):
            return "positive"
        if text == "choose":
            return "negative"
        return None
    if rule == "clinical":
        if text == "clinically_inclined":
            return "positive"
        if text == "price_sensitive":
            return "negative"
        return None
    if rule == "negotiation":
        if text == "favorable":
            return "positive"
        if text == "hard":  # displayed as "Unfavourable"
            return "negative"
        return None
    return None


def _range_end_label(value: float, kind: str) -> str:
    """A range bar's end label: the tile's own format, shortened where the
    full one is too wide for the bar's ends."""
    if kind == "pct1signed":
        return f"{value:+.0f}%"
    if kind == "pts1signed":
        return f"{value:+.0f} pts"
    if kind == "pct1":
        return f"{value:.0f}%"
    return _signal_text(value, kind)


def _range_bar_html(value: Any, kind: str, stats: tuple[float, float, int] | None, product: str) -> str:
    """The mini range bar under a numeric tile: the brand's min-to-max track,
    labelled "(min)" / "(max)" at its ends, and the account's dot (no dot
    when the value is missing). Nothing when every account has the same
    value — there is no range."""
    if stats is None:
        return ""
    low, high, n = stats
    if high - low < 1e-9:
        return ""

    def position(x: float) -> float:
        return max(0.0, min(100.0, (x - low) / (high - low) * 100.0))

    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    dot = "" if pd.isna(number) else f"<span class='range-dot' style='left:{position(float(number)):.1f}%;'></span>"
    hover = f"{product}, {SEASON} accounts (n={n}): min {_signal_text(low, kind)} · max {_signal_text(high, kind)}"
    return (
        f"<div class='range-bar' title='{escape(hover)}'>"
        f"<div class='range-track'>{dot}</div>"
        f"<div class='range-ends'><span>{escape(_range_end_label(low, kind))} (min)</span>"
        f"<span>{escape(_range_end_label(high, kind))} (max)</span></div>"
        f"</div>"
    )


def _signal_tile_html(
    tile: dict[str, Any], row: pd.Series, ranges: dict[str, tuple[float, float, int]] | None = None
) -> str:
    """One tile for the selected brand — one of the driver's source columns.
    The tooltip explaining the driver itself lives on the row's "i" icon.

    "risk_with_raw" tiles show a banded sub-component (Low/Moderate/High)
    with the raw column behind that band in parentheses, e.g.
    "Moderate (Flips: 1)" — the band reads at a glance, the raw value is the
    evidence for it.

    ``ranges`` (the brand's entry from signal_ranges()) adds a mini range bar
    under numeric values.
    """
    if tile["kind"] == "risk_with_raw":
        band = _signal_text(row.get(tile["column"]), "risk")
        raw = _signal_text(row.get(tile["raw_column"]), tile["raw_kind"])
        rendered = f"{band} ({tile['raw_label']}: {raw})" if band and raw else band
    else:
        column = tile.get("column")
        rendered = "" if column is None else _signal_text(row.get(column), tile["kind"])
    css = "signal-value" if rendered else "signal-value signal-value--empty"
    # Direction colouring reverted (user, 11/5): every value shows in the
    # default teal again. _signal_direction(), the tiles' "direction" keys and
    # the --positive/--negative CSS classes are kept (unused) in case it
    # returns; restore by re-appending the class here.
    # if rendered:
    #     direction = _signal_direction(tile, row)
    #     if direction:
    #         css += f" signal-value--{direction}"
    body = escape(rendered) if rendered else "&mdash;"
    bar = ""
    if ranges is not None and _has_range_bar(tile):
        column = tile.get("column")
        bar = _range_bar_html(row.get(column), tile["kind"], ranges.get(column), str(row.get(PRODUCT_COLUMN, "")))
    return (
        f"<div class='signal-tile'><div class='{css}'>{body}</div>{bar}"
        f"<div class='signal-label'>{escape(tile['label'])}</div></div>"
    )


def _signal_head_html(title: str) -> str:
    """Row header: driver name + a short caption (the per-row icon badge was
    removed, user 11/5 — the groups' "icon" keys stay as data).

    No hover tooltip here any more — the description that used to live on
    this hover (DRIVER_SIGNAL_TOOLTIPS) has moved into the "i" icon at the
    row's right edge (_signal_info_icon_html), alongside that driver's
    formula, so a driver's full explanation lives behind one hover target.
    """
    subtitle = DRIVER_SIGNAL_SUBTITLES.get(title, "")
    return (
        f"<div class='signal-head'>"
        f"<div class='signal-head-text-wrap'>"
        f"<span class='signal-head-text'>{escape(title)}</span>"
        f"<span class='signal-head-subtitle'>{escape(subtitle)}</span>"
        f"</div></div>"
    )


# The drivers export (user request 11/11): for each model driver, the column
# and Account signals format its value is shown with — the readable source
# column rather than the encoded model input (e.g. "Customer Leaning", not
# 0.75; prior-year doses, not their log). "benchmark_gap" and "count" are
# export-only formats (see _driver_export_value).
DRIVER_EXPORT_SOURCES = {
    "price_vs_std_benchmark_updated": ("price_vs_std_benchmark_updated", "benchmark_gap"),
    "premium_gap_vs_min_competitor_pct": ("premium_gap_vs_min_competitor_pct", "pct1signed"),
    "yoy_price_change_pct": ("yoy_price_change_pct", "pct1signed"),
    "negotiation_receptivity_score": ("negotiation_stance_bucket", "negotiation_stance"),
    "buying_group_category_cat": ("buying_group_5", "text"),
    "bob_Prior Season Shr of Wallet_enc": ("bob_Prior Season Shr of Wallet (L1BG)", "text"),
    "flips_prior_2_seasons": ("flips_prior_2_seasons", "compact"),
    "yoy_ms_change_pct_prior": ("yoy_ms_change_pct_prior", "pts1signed"),
    "multi_year_flag": ("multi_year_flag", "yesno"),
    "cde_engagement_length_log": ("cde_engagement_length_log", "num2"),
    "cde_net_sentiment_score": ("cde_net_sentiment_score", "num2"),
    "sam_risk_flag_product": ("sam_risk_flag_product", "yesno"),
    "log_prior_customer_doses": ("bob_L1BG Prior Season Customer Doses", "count"),
    "apt_clinical_bucket_enc": ("clinical_bucket", "clinical_bucket"),
    "bob_Prior Season HS Partnership Advocacy_enc": ("bob_Prior Season HS Partnership Advocacy (L1BG+Brand)", "text"),
}


def _driver_export_value(feature: str, row: pd.Series) -> str:
    column, kind = DRIVER_EXPORT_SOURCES[feature]
    value = row.get(column)
    if kind in ("benchmark_gap", "count"):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            return "No data"
        if kind == "count":
            return f"{float(number):,.0f}"
        # 100 x (benchmark - price) / benchmark: positive = cheaper than the
        # benchmark. Within 0.05 shows as at the benchmark, as it rounds to 0.0.
        if abs(number) < 0.05:
            return "At benchmark"
        return f"{abs(float(number)):.1f}% {'below' if number > 0 else 'above'}"
    return _signal_text(value, kind) or "No data"


def driver_export_groups(product_rows: dict[str, pd.Series]) -> list[tuple[str, list[list[str]]]]:
    """[(category, [[driver, value per product, ...], ...]), ...] for the
    drivers PDF: DRIVER_CATEGORIES' drivers and order, products as columns."""
    return [
        (
            category,
            [
                [FEATURE_LABELS.get(feature, feature)]
                + [_driver_export_value(feature, row) for row in product_rows.values()]
                for feature in features
            ],
        )
        for category, features in DRIVER_CATEGORIES
    ]


@st.cache_data(show_spinner=False)
def drivers_pdf_bytes(customer: str, products: tuple[str, ...], groups: tuple) -> bytes:
    """Cached so the Deep Dive's reruns (any widget on the page) don't rebuild
    the PDF; the key is the content itself."""
    return build_drivers_pdf(
        customer, SEASON, list(products), [(title, [list(r) for r in rows]) for title, rows in groups]
    )


def render_drivers_download(customer: str, product_rows: dict[str, pd.Series]) -> None:
    """The drivers export's download button, styled like the Simulation tab's
    report button (icon only)."""
    groups = driver_export_groups(product_rows)
    data = drivers_pdf_bytes(
        customer,
        tuple(str(product) for product in product_rows),
        tuple((title, tuple(tuple(r) for r in rows)) for title, rows in groups),
    )
    st.download_button(
        "",
        icon=":material/download:",
        data=data,
        file_name=report_filename(customer, suffix="drivers"),
        mime="application/pdf",
        width="stretch",
        help="Download this account's drivers (PDF)",
        key="drivers_download",
    )


def render_account_signals(product_rows: dict[str, pd.Series], customer: str | None = None) -> None:
    """One row per model-input driver (see SIGNAL_GROUPS), the row header is
    the driver itself (with a hover tooltip explaining it) and its tiles are
    the driver's own raw source columns — replacing the earlier Customer
    Context / Negotiation Context / etc. grouping.

    One brand at a time. These signals differ by brand — dead-net price alone
    can vary threefold across an account's products — so showing all three in a
    tile either misleads by collapsing them or wrecks the table alignment by
    stacking three lines in some tiles and one in others.
    """
    # S1 frame (user decision 11/5, v1 only): the whole section — header row,
    # category tab strip, rows — renders inside one bordered card (see the
    # .st-key-signals_panel CSS), so it no longer floats between neighbours.
    # v2 keeps its open list + detail layout in a plain container.
    panel = (
        st.container(key="signals_panel")
        if DRIVER_CATEGORY_STYLE == "ranked"
        else st.container()
    )
    with panel:
        # heading is narrow — just the icon and "Account signals" text, no
        # subtitle anymore — and picker, though its column is wide, is
        # left-aligned within it (see .st-key-signal_product CSS), so the
        # HD/STD/FLK toggle lands right after the heading plus the column gap,
        # rather than at the far right edge where it was easy to miss. The
        # drivers export's download button sits at the right end (11/11).
        # The button's size is set in CSS (.st-key-drivers_download).
        heading, picker, download = st.columns([1.6, 2.9, 0.5], gap="small", vertical_alignment="center")
        with heading:
            st.markdown(
                "<div class='signal-section-title'>"
                f"<span class='signal-section-icon'>{SIGNAL_ICONS['signal']}</span>"
                "<span class='signal-section-copy'>"
                "<span class='subsection-title'>Account signals</span>"
                # Subtitle commented out per request — keep only the heading next
                # to the icon.
                # "<span class='signal-section-subtitle'>"
                # "Key indicators to assess account opportunity and risk."
                # "</span>"
                "</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        available = list(product_rows)
        with picker:
            chosen = st.segmented_control(
                "Brand",
                available,
                format_func=str,  # the brand name (Viresta / Immunexa / Noventra), not the model key
                default=available[0],
                key="signal_product",
                label_visibility="collapsed",
            )
        if chosen not in product_rows:
            chosen = available[0]
        with download:
            if customer is None:
                customer = str(next(iter(product_rows.values())).get(CUSTOMER_COLUMN, ""))
            render_drivers_download(customer, product_rows)
        row = product_rows[chosen]
        brand_ranges = signal_ranges().get(str(chosen), {})

        # No Driver Value column in v8: the composite driver scores it showed no
        # longer exist, so each row is its signal name and its source columns.
        def rows_html(groups: list[dict[str, Any]]) -> str:
            html = (
                "<div class='signal-table-header'>"
                "<div class='signal-table-header-signal'>Signal</div>"
                "<div class='signal-table-header-features'><span>Corresponding Features</span></div>"
                "<div class='signal-table-header-formula'></div>"
                "</div>"
            )
            for group in groups:
                feature_html = "".join(_signal_tile_html(tile, row, brand_ranges) for tile in group["tiles"])
                html += (
                    f"<div class='signal-row'>"
                    f"{_signal_head_html(group['title'])}"
                    f"<div class='signal-tiles'>{feature_html}</div>"
                    f"{_signal_info_icon_html(group['title'])}"
                    f"</div>"
                )
            # Only when a bar was actually drawn: an eligible tile with no
            # spread (e.g. the standard benchmark) draws none.
            if "class='range-bar'" in html:
                html += (
                    "<div class='signal-range-note'>Bars show where this account sits among all "
                    f"{escape(SEASON)} {escape(str(chosen))} accounts, from the lowest (min) to the highest "
                    "(max) value. Hover a bar for the exact values.</div>"
                )
            return html

        # Category view (user decision 11/5): one category of rows at a time
        # instead of the whole list. The rows themselves are unchanged.
        groups_by_title = {group["title"]: group for group in SIGNAL_GROUPS}
        category_names = [name for name, _ in SIGNAL_CATEGORIES]
        category_titles = dict(SIGNAL_CATEGORIES)

        def category_feature_count(name: str) -> int:
            return sum(
                len(groups_by_title[title]["tiles"])
                for title in category_titles[name]
                if title in groups_by_title
            )

        def groups_for(category: str | None) -> list[dict[str, Any]]:
            # "All" (and a deselected control, which returns None) shows every
            # row together — the pre-category view.
            if category not in category_titles:
                return list(SIGNAL_GROUPS)
            return [groups_by_title[t] for t in category_titles[category] if t in groups_by_title]

        if DRIVER_CATEGORY_STYLE == "ranked":
            # v1 / option N + S1: an attached tab strip (styled via the
            # .st-key-signal_category CSS) with an "All" tab first.
            st.markdown("<div class='signals-panel-divider'></div>", unsafe_allow_html=True)
            category = st.segmented_control(
                "Signal category",
                ["All"] + category_names,
                default="All",
                key="signal_category",
                label_visibility="collapsed",
            )
            st.markdown("<div class='signals-panel-divider'></div>", unsafe_allow_html=True)
            st.markdown(rows_html(groups_for(category)), unsafe_allow_html=True)
        else:
            # v2 / option O: a category list on the left, the chosen category's
            # rows on the right.
            left, right = st.columns([1.15, 3.85], gap="medium")
            with left:
                category = st.radio(
                    "Signal category",
                    category_names,
                    format_func=lambda name: f"{name} ({category_feature_count(name)})",
                    key="signal_category",
                    label_visibility="collapsed",
                )
            with right:
                st.markdown(rows_html(groups_for(category)), unsafe_allow_html=True)


def render_driver_rows(product_rows: dict[str, pd.Series]) -> None:
    """One row per model driver: name on the left, then a red-to-green range
    track carrying a marker and the values per product, then the derivation."""
    st.markdown(
        "<div class='small-label'>Model drivers — where this account sits</div>",
        unsafe_allow_html=True,
    )

    # One brand at a time, chosen from whatever the global filter selected, so
    # this can only ever narrow that set.
    available = list(product_rows)
    picker, _ = st.columns([1.2, 3.8], gap="small")
    with picker:
        chosen = st.selectbox(
            "Driver product",
            available,
            key="driver_product",
            label_visibility="collapsed",
        )
    if chosen not in product_rows:
        chosen = available[0]
    shown_rows = {chosen: product_rows[chosen]}

    ranges = driver_ranges()
    rows = ""
    # Numeric features only: the segmentation flag is a category and has no
    # position on a low-to-high track.
    for name in NUMERIC_FEATURE_COLUMNS:
        low, high = ranges[name]

        markers, values = "", ""
        for product, row in shown_rows.items():
            raw_value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            key = str(product)
            if pd.isna(raw_value):
                values += f"<span>{escape(key)} n/a</span>"
                continue
            pct = min(max((float(raw_value) - low) / (high - low), 0.0), 1.0) * 100
            markers += (
                f"<span class='driver-marker' style='left:{pct:.1f}%;' "
                f"title='{escape(key)} — {escape(_trim_number(raw_value))}'>"
                f"<svg width='22' height='24' viewBox='0 0 22 24'>"
                f"<polygon points='11,1.5 20.5,22.5 1.5,22.5' fill='#59524E' "
                f"stroke='#FFFFFF' stroke-width='2.5' stroke-linejoin='round'/>"
                f"</svg></span>"
            )
            values += f"<span>{escape(key)} {escape(_trim_number(raw_value))}</span>"

        rows += (
            f"<div class='driver-row'>"
            f"<div class='driver-name'>{escape(FEATURE_LABELS[name])}</div>"
            f"<div class='driver-track'>{markers}"
            f"<span class='driver-inline'>{values}</span></div>"
            f"</div>"
        )

    st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:var(--text-secondary); font-size:.72rem; margin-top:.3rem;'>"
        "Each track spans the 2nd–98th percentile of that feature across all "
        f"{len(get_features()):,} contract lines, shaded light (low) to dark (high); "
        "the marker is this account's value. Shade indicates magnitude only, not "
        "whether the value is favourable."
        "</div>",
        unsafe_allow_html=True,
    )


def render_customer_deep_dive(
    customer: str, selected_products: list[str], product_rows: dict[str, pd.Series]
) -> None:
    st.markdown("<div class='section-title'>Customer Deep Dive</div>", unsafe_allow_html=True)
    render_kpi_cards(product_rows)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    render_account_signals(product_rows, customer)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    frame = deep_dive_frame(customer, selected_products)
    if frame.empty:
        _empty_note("history")
        return

    seasons = [s for s in HISTORY_SEASONS if s in set(frame["season"])]

    # A filter local to these two charts, separate from the account-level
    # Products filter at the top of the page: narrowing to one brand makes
    # the two charts comparable in scale (HD and STD sit on very different
    # price/volume ranges). All brands selected is today's combined view —
    # there is no separate "All" option, selecting everything means the same
    # thing.
    chart_products = [p for p in PRODUCT_VALUES if p in set(frame[PRODUCT_COLUMN])]
    picked = safe_multiselect(
        st, "Chart brands", chart_products, chart_products, "deep_dive_chart_products", width=420
    )
    chart_frame = frame[frame[PRODUCT_COLUMN].isin(picked)]

    _render_deep_dive_charts(chart_frame)


def _price_median_marks(frame: pd.DataFrame) -> alt.Chart:
    """Per-bar median: a red dash across each bar plus its value above the bar.

    ref_product_median_price is one constant per product, so the dash sits over
    every bar of that brand rather than spanning the whole plot — which made the
    three brands' medians impossible to tell apart.
    """
    medians = (
        frame[["season", PRODUCT_COLUMN, "dead_net_price", "product_median_price"]]
        .dropna(subset=["product_median_price", "dead_net_price"])
        .rename(
            columns={
                PRODUCT_COLUMN: "series",
                "product_median_price": "median",
                "dead_net_price": "bar_top",
            }
        )
    )
    # Label above whichever is higher, so it never lands inside a bar.
    medians["label_y"] = medians[["median", "bar_top"]].max(axis=1)
    medians["legend"] = MEDIAN_LEGEND_LABEL

    shared = dict(x=alt.X("season:N"), xOffset="series:N")
    tooltip = [
        alt.Tooltip("series:N", title="Product"),
        alt.Tooltip("median:Q", format="$.2f", title="Product median"),
    ]
    # A zero-height bar: it inherits the bar's own sub-band width, so the line
    # spans the bar exactly, and only its stroke is drawn — which is what lets
    # it be dashed. A rule would span the whole plot; a tick is a fixed pixel
    # width, neither of which tracks the bar.
    #
    # Stroke is encoded rather than hard-coded so the line earns a legend entry.
    dash = (
        alt.Chart(medians)
        .mark_bar(fill=None, strokeDash=[4, 3], strokeWidth=2, opacity=1)
        .encode(
            y=alt.Y("median:Q"),
            y2=alt.Y2("median:Q"),
            stroke=alt.Stroke(
                "legend:N",
                scale=alt.Scale(domain=[MEDIAN_LEGEND_LABEL], range=["#DC2626"]),
                # The mark is a bar, so the legend defaults to a filled square —
                # which read as a solid red box next to a dashed red line on the
                # plot. symbolType/symbolDash draw the swatch as the same dash
                # the chart actually uses.
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    symbolType="stroke",
                    symbolDash=[4, 3],
                    symbolStrokeWidth=2,
                    symbolSize=260,
                ),
            ),
            tooltip=tooltip,
            **shared,
        )
    )
    # No white halo: in Vega the text stroke paints over the fill, which hollowed
    # out the glyphs and made the red look washed out. The label sits above the
    # bar on the plot background, so it does not need one.
    label = (
        alt.Chart(medians)
        .mark_text(dy=-8, fontSize=11, fontWeight="bold", color="#DC2626")
        .encode(y=alt.Y("label_y:Q"), text=alt.Text("median:Q", format="$.2f"), **shared)
    )
    return dash + label


def _render_deep_dive_charts(frame: pd.DataFrame) -> None:
    """The two deep-dive charts: dead net price, total shipments.

    Margins by Season is dropped (per request); the two remaining charts
    split the row 50/50. _series_chart fixes the plot height at 260px, so
    both stay the same height.
    """
    row = st.columns(2, gap="medium")

    with row[0]:
        st.markdown(
            "<div class='subsection-title'>Dead Net Price by Season by Brand</div>",
            unsafe_allow_html=True,
        )
        data = _long(frame, "dead_net_price")
        if data.empty:
            _empty_note("dead net price")
        else:
            chart = _series_chart(data, "$ per dose", "$.2f", _active_product_scale(data))
            st.altair_chart(chart + _price_median_marks(frame), width="stretch")

    with row[1]:
        st.markdown(
            "<div class='subsection-title'>Total Shipments by Season by Brand</div>",
            unsafe_allow_html=True,
        )
        data = _long(frame, "shipments")
        if data.empty:
            _empty_note("shipments")
        else:
            st.altair_chart(_series_chart(data, "Doses", ",.0f", _active_product_scale(data)), width="stretch")

    # st.markdown(
    #     "<div style='color:var(--text-secondary); font-size:.78rem; margin-top:.4rem;'>"
    #     "The red dash on each bar of the price chart is that product's median price "
    #     "across the book (ref_product_median_price), printed above the bar — the "
    #     "reference the price driver measures against."
    #     "</div>",
    #     unsafe_allow_html=True,
    # )




# --- main -------------------------------------------------------------------


def _zs_tenant_id() -> str:
    """The ZS tenant GUID a signed-in user's `tid` claim must match.

    From secrets.toml's [zs] section, written at container start by
    src/bootstrap_secrets.py — not an environment variable, so no part of the
    Entra configuration travels via the environment. ZS_TENANT_ID is still
    honoured as a fallback for local sign-in testing without a [zs] section.

    Read at call time rather than captured in a module-level constant: the file
    is written after this module is imported, so a constant would capture the
    empty value. Locally (APP_DISABLE_AUTH=1), _require_zs_login() returns
    before the check that uses this, so unset is fine there.
    """
    try:
        # .get() rather than [...] — st.secrets raises on a missing key, and a
        # missing/incomplete file is a normal state here (see the caller,
        # which reports it as unconfigured sign-in rather than crashing).
        section = st.secrets.get("zs", {})
        tenant_id = str(section.get("tenant_id", "") or "").strip()
        if tenant_id:
            return tenant_id
    except Exception:  # noqa: BLE001 - no secrets file at all, etc.
        pass
    return os.environ.get("ZS_TENANT_ID", "").strip()


_LOGIN_CSS = """
<style>
/* Scoped to the sign-in screen: injected only when the login renders, so none
   of it can reach the application surfaces.

   Art direction. Content/media readiness is L0-L1 — real copy, but no logo
   asset, no imagery and no product object on this screen. That sets Productive
   App UI, a low expression budget and a 22px type ceiling: no display type, no
   decorative hero. The wordmark is a restrained text setting rather than the
   36px pseudo-logo it was, because a CSS approximation of a trademark should
   not be the largest thing a client sees. Stance is quiet precision — the
   character comes from the tinted field, a hairline rule, an inset credential
   row and designed states rather than from colour or scale. */

/* The field. A flat white page left the card with no edge to sit on; a tinted
   ground reads as institutional rather than blank. */
body:has(.st-key-login_card) [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 420px at 50% -8%, rgba(236,114,0,.07) 0%, rgba(236,114,0,0) 70%),
        #F7F5FB;
}
body:has(.st-key-login_card) [data-testid="stHeader"] {background: transparent;}

.st-key-login_card {
    max-width: 400px; margin: 4.2rem auto 0 auto;
    background: #FFFFFF;
    border: 1px solid #F1E1D9;
    border-radius: 16px;
    padding: 2.1rem 2.1rem 1.9rem 2.1rem;
    box-shadow: 0 1px 2px rgba(31,31,41,.04), 0 18px 44px rgba(46,17,0,.10);
    animation: login-rise 240ms cubic-bezier(.22,.61,.36,1) both;
}
@keyframes login-rise {
    from {opacity: 0; transform: translateY(8px);}
    to   {opacity: 1; transform: none;}
}

/* Left-aligned: centred text in a portrait card reads as a marketing splash.
   A credential form is a left-to-right reading task. */
.st-key-login_card p,
.st-key-login_card [data-testid="stMarkdown"],
.st-key-login_card [data-testid="stMarkdownContainer"] {text-align: left !important;}

.login-brand {
    display: block; font-size: 1.36rem; font-weight: 800; color: #12121A;
    letter-spacing: -.05em; line-height: 1;
}
.login-brand-dot {
    display: inline-block; width: .3rem; height: .3rem; border-radius: 50%;
    background: #EC7200; margin-left: .12rem; vertical-align: .62rem;
}
/* Signature move: a hairline authority rule, not a gradient bar. */
.login-rule {width: 34px; height: 2px; background: #EC7200; margin: .95rem 0 1.15rem 0;}

.login-title {
    font-size: 1.02rem; font-weight: 800; color: var(--text-primary);
    margin: 0 0 .32rem 0; line-height: 1.3;
}
.login-sub {font-size: .8rem; color: var(--text-secondary); margin: 0; line-height: 1.5;}
.login-foot {
    max-width: 400px; margin: .8rem auto 0 auto; text-align: center;
    font-size: .68rem; color: #8E8581; line-height: 1.5;
}

/* With the eyebrow and provider row removed the card holds only four elements,
   so the gap before the action does the work those rows used to: too tight and
   the button crowds the copy, too loose and the card reads half-empty. */
.st-key-login_button {margin-top: 1.6rem;}
.st-key-login_button button {
    border-radius: 10px !important; font-weight: 760 !important;
    min-height: 46px !important;
    transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease !important;
}
.st-key-login_button button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(46,17,0,.26) !important;
    filter: saturate(1.06);
}
.st-key-login_button button:active {transform: translateY(0); box-shadow: none !important;}

@media (prefers-reduced-motion: reduce) {
    .st-key-login_card {animation: none;}
    .st-key-login_button button,
    .st-key-login_button button:hover {transition: none !important; transform: none !important;}
}
</style>
"""


def _render_login_screen() -> None:
    """The sign-in screen.

    Deliberately restrained. There is no Customer logo asset in the repo, so the
    wordmark is a text setting held at the card's type ceiling rather than a
    36px CSS pseudo-logo — an approximated trademark should not be the largest
    element on a screen shown to the client. Swap in the official SVG when one
    is supplied and this becomes an <img>.

    The signature interaction is the pending state: st.login() navigates the
    browser away, and previously the button gave no sign it had been pressed.
    """
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    # A keyed container rather than a markdown <div>: Streamlit widgets cannot be
    # nested inside HTML emitted by st.markdown, so the card has to be a real
    # container for the button to sit inside it. The key gives it a stable
    # st-key-login_card class to style.
    with st.container(key="login_card"):
        st.markdown(
            """
            <div class="login-brand">demo<span class="login-brand-dot"></span></div>
            <div class="login-rule"></div>
            <p class="login-title">Customer Response Prediction Engine</p>
            <p class="login-sub">Win/Loss simulation and scenario decisioning.</p>
            """,
            unsafe_allow_html=True,
        )
        pending = st.empty()
        if st.button(
            "Sign in with Microsoft",
            type="primary",
            key="login_button",
            width="stretch",
        ):
            # Rendered before the redirect is issued, so the press is
            # acknowledged in the moment before the browser navigates away.
            pending.markdown(
                "<div class='login-sub' style='margin-top:.6rem;'>"
                "Redirecting to Microsoft&hellip;</div>",
                unsafe_allow_html=True,
            )
            st.login()

    st.markdown(
        f"<div class='login-foot'>Demo build &middot; "
        f"internal use only<br/>Version {escape(APP_VERSION)} &middot; "
        f"updated {escape(build_date())}</div>",
        unsafe_allow_html=True,
    )


def _user_roles() -> list[str]:
    """The signed-in user's Entra app roles (the ID token's ``roles`` claim).

    Locally (APP_DISABLE_AUTH=1, never in production) there is no sign-in, so
    APP_TEST_ROLES can stand in for the claim, comma-separated, e.g.
    APP_TEST_ROLES=DirectionalView — to try each role's tabs on a laptop.
    """
    if os.environ.get("APP_DISABLE_AUTH") == "1" and os.environ.get("APP_ENV") != "production":
        return roles_from_claim([r for r in os.environ.get("APP_TEST_ROLES", "").split(",") if r.strip()])
    try:
        return roles_from_claim(st.user.get("roles"))
    except Exception:  # noqa: BLE001 - no auth config / no user: treat as no role
        return []


def user_view_access() -> dict[str, Any]:
    """Which Simulation views this user may open (src/access.py), logged once per session."""
    roles = _user_roles()
    access = resolve_views(roles)
    if st.session_state.get("_access_logged") != roles:
        st.session_state["_access_logged"] = roles
        # One line per session to the container log (CloudWatch): which roles
        # the token carried and what they opened. No other user details.
        accounts = "all" if access["accounts"] is None else len(access["accounts"])
        print(
            f"[access] roles={roles} views={access['views']} accounts={accounts} "
            f"unrecognised={access['unrecognised']} fallback_all_views={access['fallback']}",
            file=sys.stderr,
            flush=True,
        )
    return access


def restrict_to_allowed_accounts(data: pd.DataFrame, allowed: list[str] | None) -> pd.DataFrame:
    """The app data cut down to the listed health systems (None = all of them).

    Matched on IQVIA LOC ID (ACCOUNT_ID_COLUMN; lists in ROLE_ACCOUNTS in
    src/access.py). A listed ID missing from the data is logged — it is how an
    ID changed by a client refresh gets noticed — and a user left with no
    accounts is stopped with a message rather than shown an empty app.
    """
    if allowed is None:
        return data
    ids = data[ACCOUNT_ID_COLUMN].astype(str).str.strip()
    present = set(ids.unique())
    missing = [i for i in allowed if i not in present]
    if missing and st.session_state.get("_access_missing_logged") != missing:
        st.session_state["_access_missing_logged"] = missing
        print(f"[access] allowed IQVIA LOC IDs not found in the data: {missing}", file=sys.stderr, flush=True)
    restricted = data[ids.isin(allowed)]
    if restricted.empty:
        st.error("No health systems are assigned to your access role. Please contact the CDE team.")
        st.stop()
    return restricted


def _require_zs_login() -> bool:
    """Gate the app on a Microsoft Entra sign-in from the ZS tenant.

    Streamlit's native OIDC flow is authorization-code with PKCE (S256), forced by
    client_kwargs in .streamlit/secrets.toml because Entra does not advertise
    code_challenge_methods_supported. Sign-in is restricted to ZS by the
    tenant-scoped server_metadata_url; the tid check below is defence in depth
    against that URL being pointed at /common.

    APP_DISABLE_AUTH=1 bypasses this for local runs and for the headless render
    test, which would otherwise stall on the sign-in screen. Also requires
    APP_ENV != "production" so a runtime env-var override (misconfiguration,
    orchestrator default, supply-chain tampering) cannot bypass auth in a
    deployed container: APP_ENV=production is baked into the image at build
    time by the Dockerfile, not set by whatever injects APP_DISABLE_AUTH.
    """
    if os.environ.get("APP_DISABLE_AUTH") == "1" and os.environ.get("APP_ENV") != "production":
        return True
    # Demo build: synthetic data, no sign-in unless APP_REQUIRE_AUTH=1.
    if os.environ.get("APP_REQUIRE_AUTH") != "1":
        return True

    # Streamlit only exposes st.user.is_logged_in once it has loaded an [auth]
    # config, i.e. once .streamlit/secrets.toml exists. Missing means
    # src/bootstrap_secrets.py never ran, or failed and start.sh continued
    # anyway (it needs `set -e`). Say so, rather than letting the attribute
    # access raise a bare AttributeError that reads like an app bug — and fail
    # CLOSED: no secrets means no auth, so nobody gets in.
    if not hasattr(st.user, "is_logged_in"):
        st.error(
            "Sign-in is not configured: no `[auth]` section was loaded, so "
            "`.streamlit/secrets.toml` is missing. The container's start.sh must run "
            "`python -m src.bootstrap_secrets` (which fetches it from AWS Secrets "
            "Manager) *before* streamlit, with `set -e` so a failure there stops "
            "startup instead of launching without credentials. Check the container "
            "logs for `[bootstrap_secrets]` lines."
        )
        return False

    if not st.user.is_logged_in:
        _render_login_screen()
        return False

    tenant_id = _zs_tenant_id()
    if not tenant_id:
        # Fail closed, but say why: comparing against "" would reject every
        # account with the misleading "restricted to ZS accounts" message below.
        st.error(
            "Sign-in succeeded but the ZS tenant ID is not configured, so the account "
            "could not be verified. The secret's `ZS_TENANT_ID` value should have been "
            "written to secrets.toml's `[zs]` section — check the container logs for "
            "`[bootstrap_secrets]` lines."
        )
        st.button("Sign out", on_click=st.logout)
        return False

    if st.user.get("tid") != tenant_id:
        st.error("This application is restricted to ZS accounts.")
        # The one place sign-out is kept. This user is signed in but rejected,
        # and the session cookie outlives the page — without it they cannot get
        # back to the sign-in screen to try the right account.
        st.button("Sign out", on_click=st.logout)
        return False

    # No signed-in caption and no sign-out control by request. That also means
    # no sidebar at all, since nothing else rendered into it.
    return True


def render_simulation_view(
    view: str,
    models: dict[str, Any],
    metadata: dict[str, Any],
    customer: str,
    contract_data: pd.DataFrame,
    selected_products: list[str],
    product_rows: dict[str, pd.Series],
) -> None:
    """One Simulation screen (levers + outcome) with its own state; see SIMULATION_VIEWS."""
    set_active_view(view)
    st.session_state.setdefault(view_key("latest_result"), None)
    st.session_state.setdefault(view_key("latest_result_at"), None)

    # Hopping to another health system drops the SHAP/Narrative product pick,
    # so the panel opens on Viresta (see render_outcome_details() for the
    # fallback order) rather than carrying the last account's choice over.
    # Keyed on the customer alone, not context_signature below: that signature
    # also moves when the product multiselect changes, and a pick that is
    # still on offer should survive adding or removing a *different* product.
    # Both keys have to go: whenever the widget's own state survives, it wins
    # over the index the shadow key feeds, so clearing only the shadow would
    # leave the previous account's pick on screen. Safe here because this runs
    # before render_outcome() instantiates the widget — Streamlit only objects
    # to writing a widget's state after it has been created in the same run.
    if st.session_state.get(view_key("_customer_for_shap")) != customer:
        st.session_state[view_key("_customer_for_shap")] = customer
        st.session_state.pop(view_key("shap_product_choice"), None)
        st.session_state.pop(view_key("shap_product"), None)

    # Per view: a view opened after an account change still sees the change
    # the first time it is drawn, and resets its own levers and result.
    signature = context_signature(customer, contract_data, selected_products)
    context_changed = st.session_state.get(view_key("context_signature")) != signature
    if context_changed:
        st.session_state[view_key("context_signature")] = signature
        st.session_state[view_key("latest_result")] = None
        st.session_state[view_key("latest_result_at")] = None
        reset_product_values(product_rows)
    else:
        initialize_product_values(product_rows)

    if st.session_state.get(view_key("latest_result")) is None:
        # Nothing simulated yet for this account/product selection — show the
        # account's real, unbanded feature values (prob0) rather than an empty
        # panel, so switching accounts always has something to show before the
        # user touches a lever. Computed before the lever panel below so its
        # download button reflects this result too.
        st.session_state[view_key("latest_result")] = score_selected_products(
            models, metadata, product_rows, fold_models=get_fold_models(), baseline=True
        )
        st.session_state[view_key("latest_result_at")] = None

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        with st.container(key="lever-sticky-panel"):
            with st.container(border=True):
                run_clicked, _, _ = render_pricing_levers(
                    selected_products, product_rows, metadata, customer
                )

    if run_clicked:
        st.session_state[view_key("latest_result")] = score_selected_products(
            models, metadata, product_rows, fold_models=get_fold_models()
        )
        st.session_state[view_key("latest_result_at")] = datetime.now().strftime("%b %d, %Y")
        # The download button above this point was rendered from the
        # previous result. Rerun so it picks up the one just scored.
        st.rerun()

    with right:
        with st.container(border=True):
            render_outcome(st.session_state.get(view_key("latest_result")), customer, product_rows)


def main() -> None:
    st.set_page_config(page_title="Customer Decision Engine", layout="wide")
    inject_css()

    if not _require_zs_login():
        return

    try:
        metadata = get_metadata()
        data = get_features()
        models = get_models()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python scripts/make_dummy_data.py", language="powershell")
        st.stop()

    render_header()

    landing_variant = active_variant()
    if landing_variant:
        page_switch_effects()
    if landing_variant and st.session_state.setdefault("show_landing", True):
        render_landing(landing_variant, metadata)
        # The landing is already on screen; fill the tool's caches while it is
        # read, so the first switch into the tool does not pay for them.
        get_fold_models()
        product_medians()
        raw_column_medians()
        driver_ranges()
        signal_ranges()
        return

    # A run cut short by a newer one leaves its queue behind; start clean.
    st.session_state[_DEFERRED_KEY] = []
    st.session_state.setdefault("saved_scenarios", [])
    # Before any widget: keeps the closed tabs' lever selections (lazy tabs).
    keep_lever_state()

    # Role-based access (11/17, src/access.py): which views and which health
    # systems this user may open. The account filter is applied here, once,
    # before the Health System picker is built — so the picker, Simulation,
    # Customer Deep Dive and both PDFs only ever see the allowed accounts.
    access = user_view_access()
    data = restrict_to_allowed_accounts(data, access["accounts"])

    shown_views = [(view, label) for view, label in SIMULATION_VIEWS if view in access["views"]]
    tab_labels = [label for _, label in shown_views] + (
        ["Customer Deep Dive"] if SHOW_CUSTOMER_DEEP_DIVE else []
    )

    # The open tab's own account list (user, 11/17: "in any view, Full
    # Simulation should only show these accounts"). The Health System picker
    # sits above the tabs and is shared by them, so it is narrowed to the open
    # tab's list here; the tab's label is already in session_state at the top
    # of the run that switching tabs triggers. Moving into Full Simulation with
    # an unlisted account selected opens the first listed one (safe_selectbox).
    open_label = st.session_state.get("main_tab")
    if open_label not in tab_labels:
        open_label = tab_labels[0]  # first run: st.tabs opens the first tab
    open_view = next((view for view, label in shown_views if label == open_label), None)
    if open_view is not None:
        data = restrict_to_allowed_accounts(data, view_accounts(open_view))

    customer, contract_data, selected_products = choose_context(data)

    product_rows = get_product_rows(contract_data, selected_products)
    if not product_rows:
        st.warning("No contract lines match this selection.")
        st.stop()

    if landing_variant:
        # Sits on the tab row's right end rather than taking a row of its own.
        st.markdown(
            "<style>"
            "[data-testid='stLayoutWrapper']:has(> .st-key-landing-return) {"
            " height:0 !important; min-height:0 !important; overflow:visible !important;"
            " margin-bottom:-1rem; position:relative; z-index:2;}"
            # The block spans the whole tab row; only the button may take
            # clicks, or the block swallows clicks on the tabs underneath.
            ".st-key-landing-return {align-items:flex-end; pointer-events:none;}"
            ".st-key-landing-return div[data-testid='stButton'] {pointer-events:auto;}"
            ".st-key-landing-return div[data-testid='stButton'] > button,"
            " .st-key-landing-return div[data-testid='stButton'] > button:hover {"
            " border:0 !important; outline:0 !important; background:transparent !important;"
            " box-shadow:none !important; color:var(--purple) !important; font-weight:700;}"
            "</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="landing-return"):
            st.button(
                "How to use this tool", type="tertiary", icon=":material/help:",
                on_click=go_to, args=(True,),
            )

    # Three independent simulation views as tabs for now (user, 11/17), then
    # the deep dive. on_change="rerun" makes the tabs lazy: only the open
    # simulation view runs, so three copies cost no more than one, and their
    # identical widget/container keys never meet in the same run.
    # Role-based (11/17, src/access.py): a user tagged with an app role sees
    # only that role's view(s) plus Customer Deep Dive; a user with no role
    # sees every view while roles are being tested. Views a user may not open
    # are never built, so nothing of theirs reaches the browser.
    tabs = st.tabs(tab_labels, key="main_tab", on_change="rerun")

    for (view, _label), tab in zip(shown_views, tabs):
        if tab.open:
            with tab:
                render_simulation_view(
                    view, models, metadata, customer, contract_data, selected_products, product_rows
                )
    set_active_view("full")

    # Rendered on every run as before, so its pickers keep their state.
    if SHOW_CUSTOMER_DEEP_DIVE:
        with tabs[-1]:
            render_customer_deep_dive(customer, selected_products, product_rows)

    run_deferred()

    # st.caption(
    #     "Real Health Systems contracting data for season "
    #     f"{metadata['season']}. Internal use only."
    # )


if __name__ == "__main__":
    main()
