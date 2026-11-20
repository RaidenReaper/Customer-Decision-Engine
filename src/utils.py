"""Schema, paths and feature derivations for the real Health Systems dataset.

The app keeps its own vocabulary and maps to the source workbook's column names
here, so a future schema change touches this module only.

Every derivation in this file was verified against the source workbook to
machine precision by ``src/prepare_real_data.py``; see the docstrings.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
DATA_PATH = DATA_DIR / "hs_features.parquet"
HISTORY_PATH = DATA_DIR / "hs_history.parquet"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Application version, shown in the header. Bump this by hand with each release
# — it identifies the build, not the model (model_metadata carries its own
# model_version).
APP_VERSION = "1.0.0"

# Written by the Dockerfile at image build time. Reading a file rather than
# taking datetime.now() at render time is deliberate: the header must state when
# the running build was deployed, not when someone happened to open the page.
BUILD_DATE_PATH = PROJECT_ROOT / "BUILD_DATE"

# The season the app shows and scores. v38 (11.5) is trained on 2024-25 +
# 2025-26 + 2026-27 and this is its scored 2027-28 season — upcoming, so it
# has prices (the revised 26/27 offers carried forward) but no actual
# outcomes yet; the sensitivity and account outputs describe it.
SEASON = "2027-2028"

# Seasons present in the extract, ordered. Only used by the deep-dive charts.
# The charts drop empty series rather than plotting zeros, so a future
# multi-season extract needs only this list extended.
HISTORY_SEASONS = ["2025-2026", "2026-2027"]

# --- target -----------------------------------------------------------------

# Binary. Index 1 of predict_proba is the Win probability.
OUTCOME_LABELS = ["Loss", "Win"]
WIN_INDEX = 1
# v2 (8/20 model package) renamed the source's decided-outcome column from
# outcome_clean to outcome_combined; src/prepare_real_data.py renames it back
# to this app-internal name so nothing downstream has to change.
TARGET_COLUMN = "outcome_clean"
ACTUAL_OUTCOME_COLUMN = "is_win_actual"

# --- model features ---------------------------------------------------------

# Order must match XGBClassifier.feature_names_in_ on all three v8 models
# (sam_xgb_{HD,STD,Noventra}.joblib, v8 "individual drivers" package) —
# build_metadata() in src/prepare_real_data.py fails if any bundle disagrees.
#
# v27 (10.22 notebook, trained 2024-25 + 2025-26) — 16 individual drivers:
#   * price vs the row's OWN season benchmark (2026-27: $60.63 HD / Noventra,
#     $13.77 STD), on dead_net_final (out-of-range prices replaced by the
#     season standard offer);
#   * premium vs the CHEAPEST competitor (Competitor A for HD / Noventra, cheaper of
#     Competitor B / Competitor C for STD);
#   * YoY price change in % (no longer $) and MOVES with price in scenarios;
#   * product-level SAM risk, flips over the prior 2 seasons, Customer share
#     change over prior seasons (pp) — all known before the season starts;
#   * new: buying group (category), multi-year flag, season (category; the
#     shown season is scored as 2025-26), log(1 + prior-season Customer doses).
FEATURE_COLUMNS = [
    "price_vs_std_benchmark_updated",
    "bob_Prior Season Shr of Wallet_enc",
    "apt_clinical_bucket_enc",
    "bob_Prior Season HS Partnership Advocacy_enc",
    "sam_risk_flag_product",
    "flips_prior_2_seasons",
    "yoy_ms_change_pct_prior",
    "cde_engagement_length_log",
    "cde_net_sentiment_score",
    "negotiation_receptivity_score",
    "premium_gap_vs_min_competitor_pct",
    "yoy_price_change_pct",
    "buying_group_category_cat",
    "multi_year_flag",
    "season_cat",
    "log_prior_customer_doses",
]

# Features passed to XGBoost as native categories, with their trained
# categories in order (CATEGORICAL_FEATURES, label -> pandas Categorical).
# None in v27: its two categories are instead float CODES the model itself
# declares categorical (feature_types "c"), listed in CODED_CATEGORICAL_FEATURES
# — they stay floats in every frame, and FEATURE_TYPES tells a hand-built
# DMatrix to read them as categories.
CATEGORICAL_FEATURES: dict[str, tuple[str, ...]] = {}
CODED_CATEGORICAL_FEATURES = ("buying_group_category_cat", "season_cat")
FEATURE_TYPES = ["c" if name in CODED_CATEGORICAL_FEATURES else "q" for name in FEATURE_COLUMNS]
NUMERIC_FEATURE_COLUMNS = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_FEATURES]

# Buying-group codes behind buying_group_category_cat (notebook's BG_CODES).
BUYING_GROUP_CODES = {"Vizient": 0, "HealthTrust": 1, "Captis": 2, "Premier": 3, "Other": 4}
# The season driver's code for the season the app shows: the notebook scores
# an unseen season as its latest training season — v38 trains through 2026-27,
# so 2027-28 is scored as 2026-27 (= 2.0; notebook cell 11:
# seg['season_cat'] = SEASON_CODES['2026-2027']).
SEASON_SCORING_CODE = 2.0

# Raw segmentation flag column the app displays (not a v8 model input).
SEGMENTATION_FEATURE = "bob_Prior Season HS Segmentation Flag"

FEATURE_LABELS = {
    # Each component is now its own driver, so the SHAP chart, the Top
    # Drivers pills and the driver-track visual name them individually —
    # labels match the Account Signals tiles and levers that show them.
    "price_vs_std_benchmark_updated": "Price vs. Standard Benchmark Offer",
    "bob_Prior Season Shr of Wallet_enc": "Share of Wallet",
    "apt_clinical_bucket_enc": "Clinical Focus",
    "bob_Prior Season HS Partnership Advocacy_enc": "Partnership Advocacy",
    "sam_risk_flag_product": "Risk Flag",
    "flips_prior_2_seasons": "Supplier Change History",
    "yoy_ms_change_pct_prior": "Year-over-year Market Share Change",
    "cde_engagement_length_log": "Engagement Score",
    "cde_net_sentiment_score": "Net Call Sentiment",
    "negotiation_receptivity_score": "Negotiation Stance",
    "premium_gap_vs_min_competitor_pct": "Price vs. Cheapest Competitor Price",
    "yoy_price_change_pct": "Year-over-year Price Change",
    "buying_group_category_cat": "Buying Group",
    "multi_year_flag": "Multi-year Contract",
    "log_prior_customer_doses": "Prior Year Doses",
    # The same code for every row the app shows (scored as 2025-26): a small
    # bar of its own on the SHAP chart, never a named driver elsewhere
    # (NARRATIVE_EXCLUDED_FEATURES).
    "season_cat": "Season",
    # Earlier models' inputs, kept so old saved scenarios still label.
    "customer_doses_to_prebook_ratio_corrected": "Prebook Conversion Ratio",
    "yoy_price_missing": "Year-over-year Price Missing",
}

# --- products ---------------------------------------------------------------

PRODUCT_COLUMN = "product_standardised"
# source value -> model key
# Demo build: hypothetical brand names (Viresta = high-dose, Immunexa =
# standard-dose, Noventra = recombinant). The model keys are unchanged.
PRODUCT_KEYS = {"Viresta": "HD", "Immunexa": "STD", "Noventra": "FLK"}
PRODUCT_VALUES = list(PRODUCT_KEYS)

# --- context / display columns ----------------------------------------------

CUSTOMER_COLUMN = "bob_L1-Buying Group"
# The health system's IQVIA LOC ID (IDN...), carried into the app data 11/17 for
# role-based account access (src/access.py). Unique per health system, unlike
# the name: two IDNs share "Example Health".
ACCOUNT_ID_COLUMN = "IQVIA LOC ID"
BUYING_GROUP_COLUMN = "bob_buying_group_name"
# "street, city, state" from DDD, by IDN — see merge_address() in
# prepare_real_data.py. "—" where DDD has no address for the account's IDN.
ADDRESS_COLUMN = "bob_address"
CONTRACT_ID_COLUMN = "contract_id"
CONTRACT_TITLE_COLUMN = "rmus_contract_title"
TIER_COLUMN = "tier_label"
INCUMBENT_COLUMN = "bob_current_incumbent_manufacturer"
PRIOR_INCUMBENT_COLUMN = "bob_prior_year_incumbent_manufacturer"

# Columns the derivations read from the selected row.
PRICE_BASE_COLUMN = "dead_net_price_per_dose"
# Per-product median dead-net price, computed by src/prepare_real_data.py.
# Display-only now (the price-lever input's fallback default and the dead-net
# price chart's reference line) — v2's price-linked features benchmark against
# STANDARD_PRICE_BENCHMARK_COLUMN instead, not this.
REFERENCE_PRICE_COLUMN = "ref_product_median_price"
STANDARD_PRICE_COLUMN = "standard_contract_price_per_dose"
# The row's own season standard benchmark (2026-27: $60.63 Viresta / Noventra,
# $13.77 Immunexa), set by src/prepare_real_data.py — the fixed reference the
# price lever's linked features recompute against.
STANDARD_PRICE_BENCHMARK_COLUMN = "standard_price_benchmark"
MULTIYEAR_COLUMN = "has_multiyear"
PREBOOK_FLAG_COLUMN = "has_prebook"
TOTAL_VOLUME_COLUMN = "bob_totalfinalsales"
TOTAL_PREBOOK_COLUMN = "total_prebook_cy"
# Contract-level BOB sales — the corrected prebook ratio's numerator.
CONTRACT_SALES_COLUMN = "contract_level_total_final_sales"

# --- Customer Deep Dive KPI cards -------------------------------------------

PRIOR_PREBOOK_COLUMN = "total_prebook_py"

# Business line. Only Health System is modelled; Retail is planned.
SEGMENT_OPTIONS = ["Health System", "Retail"]
HEALTH_SYSTEM = SEGMENT_OPTIONS[0]

# The five non-price Contracting Levers. Each key is a v8 model feature, so a
# lever choice is fed to the model directly — no composite in between.
#
# "values" maps each dropdown option to the number the model was trained on
# (the notebook's own encodings; NaN = the model's native "missing"). The
# options are the real categories an Account Manager would recognise, so
# correcting an inferred input (e.g. this account IS clinically inclined)
# is one pick, not a translation into a score.
#
# Engagement Score is the one continuous input. Its "edges" are the 1/3 and
# 2/3 quantiles of the 429 rows (of 680, 2026-27) that have a score, and each
# band feeds the median of the accounts in it, so "High" is a value real
# accounts have, not an endpoint. Accounts with no engagement data open on
# "No Data" and keep feeding the model NaN until a band is picked.
NAN = float("nan")
LEVER_SPECS: dict[str, dict[str, Any]] = {
    "bob_Prior Season Shr of Wallet_enc": {
        "values": {
            "Customer Loyalist": 1.0,
            "Customer Leaning": 0.75,
            "Competitive Leaning": 0.5,
            "Competitive Loyalist": 0.25,
            "No Data": 0.0,
        },
        # v35: a few 2026-27 rows (newly priced accounts with no prior-season
        # BOB link) carry a true NaN rather than the "No Data" encoding (0).
        # They open on "No Data" too; an untouched lever feeds the row's own
        # value (collect_overrides), so the model still sees their NaN.
        "missing_label": "No Data",
    },
    "bob_Prior Season HS Partnership Advocacy_enc": {
        # "No Data" also covers the source's "-" placeholder (both encode 0).
        "values": {
            "Advocate": 1.0,
            "Operationalize": 0.75,
            "Prioritize": 0.5,
            "Choose": 0.25,
            "No Data": 0.0,
        },
        # Same v35 true-NaN case as Share of Wallet above.
        "missing_label": "No Data",
    },
    "cde_engagement_length_log": {
        # v27 / 2026-27: 1/3 and 2/3 quantiles of the 429 (of 680) rows with a
        # score, and each band's median.
        "values": {"Low": 6.1759, "Medium": 7.6212, "High": 9.1228, "No Data": NAN},
        "edges": (6.9856, 8.3643),
    },
    "negotiation_receptivity_score": {
        "values": {"Favourable": 1.0, "Neutral": 0.5, "Unfavourable": 0.0, "Unknown": NAN},
    },
    "apt_clinical_bucket_enc": {
        # Source label "price_sensitive" shown as "Non-Clinically Inclined"
        # (10/8), matching the Account Signals tile — it is the opposite of
        # clinically inclined, not the price-sensitivity band.
        "values": {"Clinically Inclined": 1.0, "Non-Clinically Inclined": 0.0, "Unknown": NAN},
    },
}
LEVER_FEATURES = tuple(LEVER_SPECS)


def lever_options(feature: str) -> list[str]:
    return list(LEVER_SPECS[feature]["values"])


def lever_label_for_value(feature: str, value: Any) -> str:
    """The dropdown option an account's own value opens on.

    Missing maps to the option whose value is NaN, else to the spec's
    "missing_label" (the BOB levers: their "No Data" option encodes 0, and a
    true NaN — a v35 row with no prior-season BOB link — opens there too; the
    untouched lever still feeds the row's own NaN). A banded lever
    (Engagement Score) picks its band by the edges; a categorical one picks
    the option with exactly this encoded value.
    """
    spec = LEVER_SPECS[feature]
    missing = value is None or pd.isna(value)
    if missing:
        for label, encoded in spec["values"].items():
            if pd.isna(encoded):
                return label
        if spec.get("missing_label"):
            return spec["missing_label"]
        raise ValueError(f"{feature} has no option for a missing value")
    number = float(value)
    if "edges" in spec:
        low, high = spec["edges"]
        return "Low" if number < low else ("Medium" if number < high else "High")
    for label, encoded in spec["values"].items():
        if not pd.isna(encoded) and abs(float(encoded) - number) < 1e-9:
            return label
    raise ValueError(f"{feature} value {value!r} matches no lever option")


def lever_value_for_label(feature: str, label: str) -> float:
    """The number a lever option feeds the model (NaN for a missing option)."""
    return float(LEVER_SPECS[feature]["values"][label])

# Excluded from the on-screen SHAP waterfall and the printed PDF's driver
# chart alike (folded into the base value rather than dropped) — both surfaces
# must show the same set of drivers, or the numbers won't reconcile.
#
# Empty for v27. season_cat was the candidate (one code on every row the app
# shows), but its SHAP still varies by account through interactions (up to
# 0.05 log-odds on Viresta / Noventra), so folding it here would make the
# waterfall's "Average Win Probability" drift by account again (the 10/8
# issue). It was shown as its own small "Season" bar until 11/6; it is now
# REDISTRIBUTED into the other drivers instead (REDISTRIBUTED_FEATURES below
# — the user's request that season leave the UI entirely), which keeps the
# base constant. It stays out of the Top Drivers pills and the narratives
# (NARRATIVE_EXCLUDED_FEATURES) as a guard.
#
# Because whatever lands on these features disappears into base_value rather
# than into a bar, redistribute_shap() holds them out of its split — see its
# docstring. Keep the two in step: anything added here must not be a
# redistribution recipient.
SHAP_CHART_EXCLUDED_FEATURES: set[str] = set()
# Drivers that stay on the SHAP chart but are never named as a top driver or
# in a narrative — not something an Account Manager can act on or explain.
NARRATIVE_EXCLUDED_FEATURES: set[str] = {"season_cat"} | SHAP_CHART_EXCLUDED_FEATURES


def shap_chart_contributions(contributions: pd.DataFrame | None) -> pd.DataFrame | None:
    """Fold ``SHAP_CHART_EXCLUDED_FEATURES`` into the base value.

    Shared by the app's on-screen waterfall and the PDF report so both render
    the same set of drivers and the same f(x) = base_value + sum(shap) identity.
    """
    if contributions is None or contributions.empty:
        return contributions
    mask = contributions["feature"].isin(SHAP_CHART_EXCLUDED_FEATURES)
    if not mask.any():
        return contributions
    folded_base = float(contributions.attrs.get("base_value", 0.0)) + float(
        contributions.loc[mask, "shap"].sum()
    )
    filtered = contributions.loc[~mask].reset_index(drop=True)
    filtered.attrs["base_value"] = folded_base
    return filtered


# A hidden feature whose SHAP value is added onto a visible driver's bar
# rather than folded into the base value. yoy_price_missing ("no prior-year
# price on file") is the same signal as Year-over-year Price Change, and the
# 10/7 Viresta model splits on it (non-zero SHAP on every HD account, up to
# -0.16 log-odds), so folding it into the base made the waterfall's "Average
# Win Probability" — the model's E[f(X)], a per-product constant — drift by
# account (57.8%-62.2% vs a true 61.6%). Merged here instead, the base stays
# the model's own constant and the identity sum(shap) + base == f(x) holds.
SHAP_MERGED_FEATURES: dict[str, str] = {"yoy_price_missing": "yoy_price_change_pct"}


def merge_shap_features(contributions: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add each SHAP_MERGED_FEATURES feature's SHAP value onto its host
    driver and drop its own row. Applied once, before
    redistribute_shap(), where item["shap"] is first set."""
    if contributions is None or contributions.empty:
        return contributions
    merged = contributions.copy()
    for hidden, host in SHAP_MERGED_FEATURES.items():
        hidden_mask, host_mask = merged["feature"] == hidden, merged["feature"] == host
        if not hidden_mask.any() or not host_mask.any():
            continue
        merged.loc[host_mask, "shap"] += float(merged.loc[hidden_mask, "shap"].sum())
        merged = merged.loc[~hidden_mask]
    merged["direction"] = ["Increases win" if v >= 0 else "Decreases win" for v in merged["shap"]]
    merged["magnitude"] = merged["shap"].abs()
    merged = merged.sort_values("magnitude", ascending=False).reset_index(drop=True)
    merged.attrs["base_value"] = float(contributions.attrs.get("base_value", 0.0))
    return merged


# Business-requested exclusions: these drivers never appear as their own
# bar/driver anywhere SHAP values are read (chart, PDF, Narrative Summary),
# but unlike SHAP_CHART_EXCLUDED_FEATURES above, their value isn't folded
# invisibly into the base value — it's redistributed into the remaining
# drivers (see redistribute_shap()) so it still visibly reshapes the rest of
# the story instead of disappearing.
#   * Prebook Conversion Ratio (9/15 request; no longer even a model input
#     since 10/8, so in practice a no-op kept as a guard);
#   * Season (11/6 request: season leaves the UI entirely — same 60/40 rule.
#     Redistribution rather than folding keeps the waterfall's "Average Win
#     Probability" a per-product constant, the 10/8 issue).
REDISTRIBUTED_FEATURES = ("customer_doses_to_prebook_ratio_corrected", "season_cat")
REDISTRIBUTE_SAME_SIDE_SHARE = 0.6
REDISTRIBUTE_OPPOSITE_SIDE_SHARE = 0.4


def redistribute_shap(contributions: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove each REDISTRIBUTED_FEATURES driver and split its SHAP value
    into the remaining drivers: 60% divided equally across the drivers on
    its own side (positive if it was itself positive, negative if negative),
    40% divided equally across the drivers on the opposite side. Features
    are processed in REDISTRIBUTED_FEATURES order, each against the
    already-adjusted frame.

    Applied once, at the point compute_shap_contributions()'s output is
    first stored on a scenario result — so every consumer (Narrative
    Summary, the on-screen waterfall, the driver-categories accordion, the
    PDF report) sees the same redistributed values, not just one of them.

    Preserves the additivity identity (contributions['shap'].sum() +
    base_value == the model's margin): if a side has no remaining drivers to
    receive its share (e.g. every other driver is positive), that share does
    not fall back to the base value — it joins the other side instead, so
    the full removed_value ends up split equally across whichever side does
    have drivers, at a freshly recomputed average (not the untouched 60/40
    figures added on top of each other).

    SHAP_CHART_EXCLUDED_FEATURES are held out of the split (9/16). They stay
    in the frame carrying their own untouched SHAP value, but receive none of
    removed_value — because shap_chart_contributions() folds them into
    base_value, so anything handed to them would ride invisibly into the
    waterfall's "Average Win Probability" figure instead of appearing in a
    bar. That figure is the model's E[f(X)], a per-product constant; while
    yoy_price_missing's own TreeSHAP value was a flat 0.0 on every account
    the fold added nothing, but a share of the prebook value moved it per
    account (up to 26pp on Noventra) and made a population average look like
    an account-level number.
    """
    for feature in REDISTRIBUTED_FEATURES:
        contributions = _redistribute_one(contributions, feature)
    return contributions


def _redistribute_one(contributions: pd.DataFrame | None, feature: str) -> pd.DataFrame | None:
    """One REDISTRIBUTED_FEATURES driver's removal — see redistribute_shap()."""
    if contributions is None or contributions.empty:
        return contributions
    mask = contributions["feature"] == feature
    if not mask.any():
        return contributions

    removed_value = float(contributions.loc[mask, "shap"].iloc[0])
    remaining = contributions.loc[~mask].copy()
    base_value = float(contributions.attrs.get("base_value", 0.0))

    eligible_mask = ~remaining["feature"].isin(SHAP_CHART_EXCLUDED_FEATURES)
    if not eligible_mask.any():
        # No driver the chart would actually show is left to redistribute
        # into — the only case where the value has nowhere to go but the
        # base value.
        remaining.attrs["base_value"] = base_value + removed_value
        return remaining

    side_mask = remaining["shap"] >= 0 if removed_value >= 0 else remaining["shap"] < 0
    same_side_mask = eligible_mask & side_mask
    opposite_side_mask = eligible_mask & ~side_mask
    same_side_n = int(same_side_mask.sum())
    opposite_side_n = int(opposite_side_mask.sum())

    if same_side_n and opposite_side_n:
        remaining.loc[same_side_mask, "shap"] += (removed_value * REDISTRIBUTE_SAME_SIDE_SHARE) / same_side_n
        remaining.loc[opposite_side_mask, "shap"] += (
            removed_value * REDISTRIBUTE_OPPOSITE_SIDE_SHARE
        ) / opposite_side_n
    elif same_side_n:
        # Opposite side is empty — its 40% doesn't fall back to the base
        # value; the whole removed_value splits across the same side instead.
        remaining.loc[same_side_mask, "shap"] += removed_value / same_side_n
    else:
        # Same side is empty — mirrored: the whole removed_value splits
        # across the opposite side instead.
        remaining.loc[opposite_side_mask, "shap"] += removed_value / opposite_side_n

    remaining["direction"] = ["Increases win" if v >= 0 else "Decreases win" for v in remaining["shap"]]
    remaining["magnitude"] = remaining["shap"].abs()
    remaining = remaining.sort_values("magnitude", ascending=False).reset_index(drop=True)
    remaining.attrs["base_value"] = base_value
    return remaining


# 2x2 quadrant assignment: prior-year incumbency (Customer vs not) crossed with
# price sensitivity (Sensitive vs Non-sensitive). One of "Build Value &
# Relationship", "Targeted Retention", "Protect Margin" or "Competitive Price
# Focus". The model package ships no scored file, so src/prepare_real_data.py's
# score_population() computes it (and the two columns below) the notebook's
# way and checks it against the notebook's own output row for row.
QUADRANT_COLUMN = "quadrant_2x2"
# Sensitive / Non-sensitive: SENSITIVITY_BAND_LMH_COLUMN collapsed —
# Sensitive when the band is Medium or High. The 2x2 quadrant is built on it,
# so the quadrant and the Price Sensitivity card always agree.
SENSITIVITY_COLUMN = "sensitivity_flag"
# Per-brand (not account-level) Low/Medium/High price-sensitivity band, the
# standard-benchmark rule: S = |win-probability change when the dead-net price
# moves to the flat standard benchmark| (sensitivity_benchmark_pp), Low below
# the product's volume-weighted median of S (or S == 0, already at the
# benchmark), High at or above its volume-weighted P70
# (sensitivity_low_cutoff_pp / sensitivity_high_cutoff_pp carry the two cuts).
SENSITIVITY_BAND_LMH_COLUMN = "sensitivity_band_lmh"
# Account-level (one value per Health System, broadcast across its 2-3
# products) Low/Medium/High price-sensitivity band and confidence in that
# band, from a supplementary extract narrower than the main population (see
# merge_confidence() in src/prepare_real_data.py) — missing for rows outside
# it, not guessed. The narrative's price-sensitivity sentence gates on
# ACCOUNT_SENSITIVITY_BAND_COLUMN, not the per-product SENSITIVITY_COLUMN.
ACCOUNT_SENSITIVITY_BAND_COLUMN = "account_sensitivity_band"
ACCOUNT_CONFIDENCE_LABEL_COLUMN = "account_confidence_label"

CURRENT_VOLUME_COLUMN = "bob_totalfinalsales"
# Prior-season volume at buying-group level. Deliberately not py_shipped, whose
# median is 33,040 against this column's 1,020 — a different grain entirely.
PRIOR_VOLUME_COLUMN = "bob_L1BG Prior Season Customer Doses"

CURRENT_SHARE_COLUMN = "ddd_customer_market_share"
PRIOR_SHARE_COLUMN = "bob_L1BG Prior Season Customer MS"
# Verified exact: share_change == current share - prior share.
SHARE_CHANGE_COLUMN = "share_change"

PRICE_YOY_COLUMN = "yoy_dead_price_change_customer"
# Verified to 5.33e-15: equals dead_net_price_per_dose - py_dead_price_customer.

ADVOCACY_COLUMN = "bob_Prior Season HS Partnership Advocacy (L1BG+Brand)"
SEGMENTATION_COLUMN = "bob_Prior Season HS Segmentation Flag (L1BG)"
# Both are constant per buying group (0 and 7 of 583 groups vary), so each
# renders as one value rather than per product.

# Values that mean "not recorded" in the two categorical KPI fields.
EMPTY_CATEGORICAL_VALUES = {"No Data", "-", "", "nan"}

# --- Customer Deep Dive charts ----------------------------------------------

SHIPMENTS_COLUMN = "bob_totalfinalsales"
PREMIUM_COLUMN = "premium_%_vs_competitor1"

# Provider margin series. There is no competitor 3 margin column in the source
# extract, and competitor 2 is only ~36% populated, so a series is dropped per
# customer when it has no values.
MARGIN_COLUMNS = {
    "Customer": "customer_provider_margin",
    "Competitor 1": "competitor1_provider_margin",
    "Competitor 2": "competitor2_provider_margin",
}

# DDD dose series. Competitor 2 and 3 are dropped per customer when they are
# all zero, so the chart does not carry empty bars.
DDD_DOSE_COLUMNS = {
    "Customer": "ddd_customer_doses",
    "Competitor 1": "ddd_competitor1_doses",
    "Competitor 2": "ddd_competitor2_doses",
    "Competitor 3": "ddd_competitor3_doses",
}

# The four account-posture buckets, shown as KPI values rather than a chart:
# across 289 buying groups, 0 vary by season and only 1 varies at all, so a
# season-by-season view carried no information. Only ~22-38% populated.
BUCKET_DIMENSIONS = {
    "Negotiation Stance": "negotiation_stance_bucket",
    "Competitive Landscape": "competitive_landscape_bucket",
    "Competitive Strategy": "competitor_strategy_bucket",
    "Clinical Bucket": "clinical_bucket",
}

# Fixed colour per category value, so a favourable reading is green and an
# adverse one red whichever of the four dimensions it belongs to.
BUCKET_VALUE_COLOURS = {
    # negotiation stance
    "favorable": "#1B7F3B",
    "neutral": "#4285F4",
    "hard": "#C0392B",
    # competitive landscape
    "low": "#1B7F3B",
    "medium": "#EC7200",
    "high": "#C0392B",
    # competitor strategy
    "defensive": "#1B7F3B",
    "balanced": "#4285F4",
    "aggressive": "#C0392B",
    # clinical bucket
    "clinically_inclined": "#EC7200",
    "price_sensitive": "#EC7200",
    # shared
    "unknown": "#BEB5B0",
}

TIER_ORDER = [
    "VPN / VAG",
    "Option 3",
    "Option 2",
    "Option 1",
    "Partner Plus / Captis",
    "Committed / Enhanced Partner Plus",
]

COMMITMENT_WEIGHTS = {"multiyear": 0.4, "prebook": 0.6}

# Scenario the app opens on.
#
# The app falls back to the first alphabetical option if this row is ever absent.
DEFAULT_SCENARIO = {
    # Example Health, per 9/11 request. In v38's scored 2027-28 season it has
    # all three products. The app opens on whichever of "products" the
    # account has.
    #
    # Under v38 it opens on Viresta 34.1% Loss, Immunexa 12.2% Loss, Noventra
    # 72.0% Win (notebook account_sensitivity_all.xlsx).
    # Demo build: a synthetic account (see scripts/make_dummy_data.py).
    "customer": "Northwind Health",
    "price_group": "PG-1001",
    "products": ["Viresta", "Immunexa", "Noventra"],
}


# --- io ---------------------------------------------------------------------


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run python scripts/make_dummy_data.py first."
        )
    return read_json(path)


def load_features(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run python scripts/make_dummy_data.py first."
        )
    return pd.read_parquet(path)


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """All-season slice used by the Customer Deep Dive charts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run python scripts/make_dummy_data.py first."
        )
    return pd.read_parquet(path)


def today_iso() -> str:
    return date.today().isoformat()


def build_date() -> str:
    """The date this build was deployed, as shown in the header.

    Resolution order:
      1. BUILD_DATE, written by the Dockerfile when the image is built. This is
         the value that matters in production — it changes only when a new image
         is built and pushed, which is exactly "last updated" for the deployment.
      2. APP_BUILD_DATE, for hosts that would rather inject it.
      3. The newest data/model artifact's mtime, so a source checkout shows
         something truthful instead of today's date.
    """
    if BUILD_DATE_PATH.exists():
        stamped = BUILD_DATE_PATH.read_text(encoding="utf-8").strip()
        if stamped:
            return stamped

    injected = os.environ.get("APP_BUILD_DATE", "").strip()
    if injected:
        return injected

    artifacts = [path for path in (DATA_PATH, MODEL_METADATA_PATH) if path.exists()]
    if not artifacts:
        return "unknown"
    newest = max(path.stat().st_mtime for path in artifacts)
    return datetime.fromtimestamp(newest, tz=timezone.utc).date().isoformat()


def product_key(product_value: str) -> str:
    try:
        return PRODUCT_KEYS[product_value]
    except KeyError as exc:
        raise KeyError(
            f"Unknown {PRODUCT_COLUMN} value {product_value!r}; "
            f"expected one of {PRODUCT_VALUES}"
        ) from exc


def threshold_for(product_value: str, metadata: dict[str, Any]) -> float:
    """Each product has its own decision threshold, chosen by maximum
    F0.5 (precision-weighted), v2 model: HD 0.77, STD 0.53, FLK 0.57. Using 0.5
    for all three would misclassify.

    The key is "classification_threshold" — the previous model's metadata called
    it "threshold", so a stale metadata file fails loudly here rather than
    silently scoring against a default.
    """
    product = metadata["products"][product_key(product_value)]
    return float(product["classification_threshold"])


def lever_ranges_for(product_value: str, metadata: dict[str, Any]) -> dict[str, float]:
    """Per-product slider bounds.

    These moved: the previous metadata carried a top-level "lever_ranges" block
    keyed by product, the current one nests it inside each product entry.
    """
    product = metadata["products"][product_key(product_value)]
    return product["lever_ranges"]


# --- feature derivations ----------------------------------------------------


def derive_price_features(
    dead_net_price: float,
    baseline_dead_net_price: float,
    standard_price_benchmark: float,
    baseline_premium_gap_pct: float,
    prior_season_price: float = float("nan"),
) -> dict[str, float]:
    """The dead net price lever drives three of the model features — the
    v27 bundles' "price_linked" list.

    Ported from the modelling team's notebook (scenario_X), which expresses
    the scenario as a percentage change `pct` off the row's own baseline
    price. Here the lever gives an absolute new price directly, so
    ``m = dead_net_price / baseline_dead_net_price`` plays the role their
    ``1 + pct/100`` does — the two are algebraically the same scenario:

        price_vs_std_benchmark_updated     = 100 * (bench - dead_net_price) / bench
        premium_gap_vs_min_competitor_pct  = 100 * (m * (1 + baseline_premium/100) - 1)
        yoy_price_change_pct               = 100 * (dead_net_price - prior_price) / prior_price

    ``standard_price_benchmark`` is the row's own season benchmark (2026-27:
    $60.63 Viresta / Noventra, $13.77 Immunexa). YoY is NOT frozen in v27: it is
    recomputed against the prior-season price (known before contracting), and
    stays NaN where there is no prior-season price. src/prepare_real_data.py
    checks that at m == 1 all three reproduce the extract's own columns.
    """
    price = float(dead_net_price)
    baseline = float(baseline_dead_net_price)
    bench = float(standard_price_benchmark)
    premium = float(baseline_premium_gap_pct)
    prior = float(prior_season_price) if prior_season_price is not None else float("nan")

    m = price / baseline if baseline else float("nan")
    price_offered = 100 * (bench - price) / bench if bench else float("nan")
    premium_gap = 100 * (m * (1 + premium / 100) - 1) if not pd.isna(m) else float("nan")
    yoy = 100 * (price - prior) / prior if (not pd.isna(prior) and prior != 0) else float("nan")
    return {
        "price_vs_std_benchmark_updated": price_offered,
        "premium_gap_vs_min_competitor_pct": premium_gap,
        "yoy_price_change_pct": yoy,
    }


def derive_prebook_ratio(total_volume: float, total_prebook: float) -> float:
    """customer_doses_to_prebook_ratio_corrected =
    contract_level_total_final_sales / total_prebook_cy.

    Both sides are contract-level for the same season, contract and brand
    (the numerator is BOB sales summed over every customer on the contract),
    so the ratio compares like with like — the "corrected" in the name.
    ``bob_totalfinalsales`` (one customer's sales) is *not* the numerator.

    Returns NaN when the contract has no current-year prebook, matching the
    source column — the model routes NaN down its learned default branch.
    """
    if total_prebook is None or pd.isna(total_prebook) or float(total_prebook) == 0:
        return float("nan")
    if total_volume is None or pd.isna(total_volume):
        return float("nan")
    return float(total_volume) / float(total_prebook)


def _passthrough(base_row: pd.Series, overrides: dict[str, Any], column: str) -> float:
    """A feature the app reads straight from the row unless a lever moved it.

    Returns NaN rather than raising when the extract has no value. The model
    bundle has no imputer step — a bare XGBClassifier trained with
    missing=nan, so NaN passes straight through to predict_proba and is
    routed by XGBoost's own learned default split direction, not filled with
    a training median beforehand.
    """
    value = overrides.get(column, base_row.get(column))
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _category(base_row: pd.Series, feature: str) -> str | float:
    """A categorical feature as one of its trained category labels, or NaN
    when missing or unseen (an unseen label must not be scored as if it were
    a known category)."""
    value = base_row.get(feature)
    if value is None or pd.isna(value):
        return float("nan")
    label = str(value).strip()
    return label if label in CATEGORICAL_FEATURES[feature] else float("nan")


PRICE_LINKED_FEATURES = (
    "price_vs_std_benchmark_updated",
    "premium_gap_vs_min_competitor_pct",
    "yoy_price_change_pct",
)
# The prior-season dead-net price the YoY channel is measured against.
PRIOR_PRICE_COLUMN = "py_dead_price_customer"


def build_feature_row(
    base_row: pd.Series,
    overrides: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the model's feature vector for a scenario (v27 model).

    ``base_row`` is the selected contract/product row from the parquet;
    ``overrides`` carries the lever values the user has set, keyed by model
    feature name (the five non-price levers) or "dead_net_price". Anything
    not overridden falls back to the row's actual value.

    The three price-linked features (benchmark position, premium gap, YoY %)
    are recomputed only when the dead-net price lever is present; otherwise
    the extract's own values pass through unchanged.

    Key order matches FEATURE_COLUMNS, which the models enforce. Every value
    is a float (the two coded categories included) except a
    CATEGORICAL_FEATURES feature's, a category label (or NaN).
    """
    # The price box shows 2 decimals, but v27 prices can carry more (the
    # season offer that replaces out-of-range prices is $57.352 HD /
    # $13.7696 STD). Re-entering the shown value moves the price by a
    # fraction of a cent, which can cross a tree split (Example Health STD:
    # 28.4% -> 26.5%), so a price within half a cent of the row's own counts
    # as unchanged.
    if "dead_net_price" in overrides and pd.notna(base_row.get(PRICE_BASE_COLUMN)):
        if abs(float(overrides["dead_net_price"]) - float(base_row[PRICE_BASE_COLUMN])) < 0.005:
            overrides = {key: value for key, value in overrides.items() if key != "dead_net_price"}
    if "dead_net_price" in overrides:
        price_features = derive_price_features(
            overrides["dead_net_price"],
            base_row[PRICE_BASE_COLUMN],
            base_row[STANDARD_PRICE_BENCHMARK_COLUMN],
            base_row["premium_gap_vs_min_competitor_pct"],
            base_row.get(PRIOR_PRICE_COLUMN, float("nan")),
        )
    else:
        price_features = {name: _passthrough(base_row, overrides, name) for name in PRICE_LINKED_FEATURES}

    if "total_volume" in overrides or "total_prebook" in overrides:
        prebook_ratio = derive_prebook_ratio(
            overrides.get("total_volume", base_row[CONTRACT_SALES_COLUMN]),
            overrides.get("total_prebook", base_row[TOTAL_PREBOOK_COLUMN]),
        )
    else:
        prebook_ratio = _passthrough(base_row, overrides, "customer_doses_to_prebook_ratio_corrected")

    features: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        if name in PRICE_LINKED_FEATURES:
            features[name] = price_features[name]
        elif name == "customer_doses_to_prebook_ratio_corrected":
            features[name] = prebook_ratio
        elif name in CATEGORICAL_FEATURES:
            features[name] = _category(base_row, name)
        else:
            features[name] = _passthrough(base_row, overrides, name)
    return features
