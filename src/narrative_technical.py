"""Deterministic, rule-based "technical" scenario narrative — four labeled
sections per product: Predicted outcome, Account context, Price sensitivity,
Implication. Every driver clause is backed by its %-vs-median figure and,
where available, the raw source column and value behind it. No model or LLM
call, so it stays auditable and reproducible: same scenario in, same text
out.

This is one of two interchangeable narrative styles — see
``src/narrative_business.py`` for the plain-language, no-numbers version.
Which one renders in the app is a single constant, ``NARRATIVE_STYLE`` in
app.py; see the note at the bottom of this module for how to switch it.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from src.utils import EMPTY_CATEGORICAL_VALUES, NARRATIVE_EXCLUDED_FEATURES, SEGMENTATION_FEATURE

# negotiation_stance_bucket's raw values, relabelled for display — "hard" is
# a data-team term, not what the client calls a negotiation stance.
NEGOTIATION_STANCE_DISPLAY = {
    "favorable": "Favorable",
    "neutral": "Neutral",
    "hard": "Unfavourable",
}


def _join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _top_shap_features(
    shap: pd.DataFrame | None, positive: bool, limit: int = 2
) -> list[tuple[str, float]]:
    """Up to `limit` drivers of one sign, ranked by |shap| — the single
    selection rule shared by the narrative's driver clauses and the Top
    Drivers cards, so they never disagree about which features are the top
    drivers. NARRATIVE_EXCLUDED_FEATURES (non-business features with no
    narrative phrase, e.g. the season code) are left out."""
    if shap is None or shap.empty:
        return []
    shap = shap[~shap["feature"].isin(NARRATIVE_EXCLUDED_FEATURES)]
    subset = shap[shap["shap"] >= 0] if positive else shap[shap["shap"] < 0]
    top = subset.reindex(subset["shap"].abs().sort_values(ascending=False).index).head(limit)
    return list(top[["feature", "shap"]].itertuples(index=False, name=None))


# Deterministic, rule-based narrative copy — no model or LLM call. Each
# entry names a driver plainly for both sides; wording keyed to direction of
# effect, not magnitude (the %-vs-median number carries the magnitude).
# Every model feature except the two _driver_phrase() phrases from their own
# value (price_vs_std_benchmark_updated, the segmentation flag) needs an
# entry — a top driver without one raises.
# All entries are neutral — no directional claim ("weak", "elevated",
# "limited", ...) — because the %-vs-median figure shown right next to the
# phrase already carries the direction, and a fixed word can contradict it
# (SHAP's constraining/supporting side does not always match the raw
# value's own position vs. the median — a fixed "weaker engagement" once
# sat next to a positive %-vs-median). Same stem on both sides; only the
# verb ("supported by" / "constrained by") distinguishes them.
DRIVER_PHRASES = {
    # price_vs_std_benchmark_updated is handled separately in _driver_phrase()
    # — its wording depends on the raw value's sign, not only the SHAP sign.
    "bob_Prior Year Customer Doses": {
        "supporting": "prior-year volume",
        "constraining": "prior-year volume",
    },
    "bob_Prior Season Shr of Wallet_enc": {"supporting": "share of wallet", "constraining": "share of wallet"},
    "bob_Prior Season HS Partnership Advocacy_enc": {
        "supporting": "partnership advocacy",
        "constraining": "partnership advocacy",
    },
    "apt_clinical_bucket_enc": {"supporting": "clinical focus", "constraining": "clinical focus"},
    "sam_feat_high_or_medium_risk_flag": {"supporting": "account risk flag", "constraining": "account risk flag"},
    "cde_engagement_length_log": {"supporting": "account engagement", "constraining": "account engagement"},
    "cde_net_sentiment_score": {"supporting": "call sentiment", "constraining": "call sentiment"},
    "negotiation_receptivity_score": {"supporting": "negotiation stance", "constraining": "negotiation stance"},
    "no_of_flips_cumulative": {"supporting": "supplier change history", "constraining": "supplier change history"},
    "yoy_ms_change_enc": {"supporting": "market share trend", "constraining": "market share trend"},
    "customer_doses_to_prebook_ratio_corrected": {
        "supporting": "prebook commitment",
        "constraining": "prebook commitment",
    },
    "customer_provider_margin": {"supporting": "provider margin", "constraining": "provider margin"},
    "premium_gap_vs_min_competitor_pct": {
        "supporting": "competitive price positioning",
        "constraining": "competitive price positioning",
    },
    # v27 drivers
    "log_prior_customer_doses": {"supporting": "prior-year volume", "constraining": "prior-year volume"},
    "sam_risk_flag_product": {"supporting": "account risk flag", "constraining": "account risk flag"},
    "flips_prior_2_seasons": {"supporting": "supplier change history", "constraining": "supplier change history"},
    "yoy_ms_change_pct_prior": {"supporting": "market share trend", "constraining": "market share trend"},
    "buying_group_category_cat": {"supporting": "buying-group affiliation", "constraining": "buying-group affiliation"},
    "multi_year_flag": {"supporting": "contract term", "constraining": "contract term"},
    "season_cat": {"supporting": "season effect", "constraining": "season effect"},
    # Full comparative clauses — _driver_phrase() uses these two as-is.
    "yoy_price_change_pct": {
        "supporting": "similar price compared to previous year",
        "constraining": "price increase compared to previous year",
    },
}


def _driver_phrase(item: dict[str, Any], feature: str, positive: bool) -> str:
    """The narrative phrase for one driver. price_vs_std_benchmark_updated
    branches three ways on the raw value's own sign (>0 priced below the
    standard offer, ==0 priced at it, <0 priced above it) crossed with which
    side of the outcome it is on. The segmentation flag is a category with no
    direction, so it is named as such. Every other feature just picks its
    supporting or constraining phrase from DRIVER_PHRASES — the %-vs-median
    figure carries the magnitude, not a prepended word."""
    if feature == SEGMENTATION_FEATURE:
        return "HS segmentation"
    if feature == "price_vs_std_benchmark_updated":
        raw_value = item.get("features", {}).get(feature)
        if raw_value is None or pd.isna(raw_value) or raw_value == 0:
            return "pricing at the standard offer" if positive else "pricing in line with the standard offer"
        if raw_value > 0:
            return "pricing above the standard offer" if positive else "higher price than standard offer"
        return "lower price than standard offer" if positive else "pricing below the standard offer"
    return DRIVER_PHRASES[feature]["supporting" if positive else "constraining"]


# Features expressed as percentages vs. plain 0-1 scores or ratios — %-vs-median
# (pct_vs_median()) does not apply to these, since they are already a relative
# measure; their own value is shown directly instead.
_PCT_FEATURES = {
    "price_vs_std_benchmark_updated",
    "premium_gap_vs_min_competitor_pct",
    "yoy_price_change_pct",
}


def _format_driver_value(feature: str, value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if feature == SEGMENTATION_FEATURE:
        # A category label, not a number.
        return str(value).title()
    if feature in _PCT_FEATURES:
        return f"{value:+.1f}%"
    if feature == "customer_doses_to_prebook_ratio_corrected":
        return f"{value:.2f}x"
    return f"{value:.2f}"


def _pct_vs_median(value: float | None, median: float | None) -> float | None:
    """100 x (value - median) / |median| — omitted (None) when either value
    is missing or the median sits too close to zero for a percent difference
    to be meaningful (spec's near-zero-median guard)."""
    if value is None or median is None or pd.isna(value) or pd.isna(median):
        return None
    if abs(median) < 0.05:
        return None
    return 100 * (value - median) / abs(median)


def _format_pct_change(pct: float) -> str:
    """"100% above product median" / "33% below product median" — spelled
    out, same convention the price-position sentence already uses, rather
    than a bare signed percentage."""
    return f"{abs(pct):.0f}% {'above' if pct >= 0 else 'below'} product median"


def _driver_detail(item: dict[str, Any], feature: str, median: float | None) -> str:
    """The parenthetical backing a driver phrase: %-vs-median for score-type
    features (the _enc encodings, prior doses, flips, ...), the feature's own
    value for features already expressed as a percentage or ratio
    (pct_vs_median doesn't apply to those), and the label itself for the
    categorical segmentation flag."""
    value = item.get("features", {}).get(feature)
    if feature not in _PCT_FEATURES and feature != SEGMENTATION_FEATURE:
        pct = _pct_vs_median(value, median)
        if pct is not None:
            return _format_pct_change(pct)
    formatted = _format_driver_value(feature, value)
    return formatted if formatted is not None else "value not available"


# Raw columns behind each composite driver score (per the modelling team's
# formula table) that _raw_detail() can cite by name. Features not listed
# have no raw breakdown available (single-source, or nothing beyond the
# composite) — their driver detail stays the composite %-vs-median alone.
#
# Each entry's "kind" picks the inclusion rule:
#   numeric_above_median — value (item[item_key]) qualifies when it's above
#     that raw column's population median (raw_column_medians()), the
#     agreed threshold for numeric raw columns.
#   categorical_present  — qualifies whenever populated and not a
#     No Data / "-" / blank placeholder (EMPTY_CATEGORICAL_VALUES).
#   categorical_in       — qualifies only when the value is one of a fixed
#     allowed set (case-insensitive) — e.g. clinical_bucket only counts as
#     evidence when it is literally "clinically_inclined", not "unknown" or
#     "price_sensitive".
#   flag_equals          — qualifies only when the value equals a fixed
#     number (sam_feat_high_or_medium_risk_flag: only when == 1).
#   share_change_pct     — special-cased: %% change from
#     PRIOR_SHARE_COLUMN to CURRENT_SHARE_COLUMN, shown with an up/down
#     direction word; omitted when the two are ~equal (no real change).
#
# The model's drivers are these raw columns themselves, so the
# encoded ones cite the label behind their code (e.g. "Customer Loyalist"
# behind wallet_enc 1.0) and market-share change cites the actual % move.
RAW_DETAIL_BY_FEATURE = {
    "bob_Prior Season Shr of Wallet_enc": [
        {
            "kind": "categorical_present",
            "item_key": "prior_wallet_share",
            "label": "Wallet share",
            "hide_label": True,
        },
    ],
    "bob_Prior Season HS Partnership Advocacy_enc": [
        {
            "kind": "categorical_present",
            "item_key": "advocacy",
            "label": "Partnership advocacy",
            "hide_label": True,
        },
    ],
    "yoy_ms_change_enc": [
        {"kind": "share_change_pct", "label": "YoY share change"},
    ],
    "yoy_ms_change_pct_prior": [
        {"kind": "share_change_pct", "label": "YoY share change"},
    ],
    "apt_clinical_bucket_enc": [
        {
            "kind": "categorical_in",
            "item_key": "clinical_bucket",
            "values": {"clinically_inclined", "price_sensitive"},
            "label": "Clinical bucket",
            "hide_label": True,
        },
    ],
    "negotiation_receptivity_score": [
        {
            "kind": "categorical_in",
            "item_key": "negotiation_stance",
            "values": {"hard", "neutral", "favorable"},
            "label": "Negotiation stance",
        },
    ],
    "premium_gap_vs_min_competitor_pct": [
        {
            "kind": "numeric_above_median",
            "item_key": "premium_vs_competitor1",
            "raw_column": "premium_%_vs_competitor1",
            "label": "Premium vs competitor 1",
        },
    ],
    "price_vs_std_benchmark_updated": [
        {
            "kind": "numeric_above_median",
            "item_key": "dead_net_price",
            "raw_column": "dead_net_price_per_dose",
            "label": "Dead-net price",
        },
        {
            "kind": "numeric_above_median",
            "item_key": "standard_price_benchmark",
            "raw_column": "standard_price_benchmark",
            "label": "Standard price benchmark",
        },
    ],
}


def _raw_detail(item: dict[str, Any], feature: str, raw_medians_for_product: dict[str, float]) -> str | None:
    """Which raw column(s) behind ``feature`` back up its driver phrase, per
    each entry's own inclusion rule (see RAW_DETAIL_BY_FEATURE). Any number
    of a feature's raw columns can qualify; returns None when none do (or
    the feature has no raw breakdown at all)."""
    entries = RAW_DETAIL_BY_FEATURE.get(feature)
    if not entries:
        return None

    bits = []
    for entry in entries:
        kind = entry["kind"]
        label = entry["label"]

        if kind == "numeric_above_median":
            value = item.get(entry["item_key"])
            median = raw_medians_for_product.get(entry["raw_column"])
            if value is None or median is None or pd.isna(value) or pd.isna(median):
                continue
            if value > median:
                bits.append(f"{label}: {value:,.2f}")

        elif kind == "categorical_present":
            value = item.get(entry["item_key"])
            if value and value not in EMPTY_CATEGORICAL_VALUES:
                bits.append(str(value) if entry.get("hide_label") else f"{label}: {value}")

        elif kind == "categorical_in":
            value = item.get(entry["item_key"])
            if value and str(value).strip().lower() in entry["values"]:
                # negotiation_stance_bucket's raw "hard" reads as data-speak,
                # not what the client calls it — display-only relabel, the
                # underlying column value is untouched.
                display_value = NEGOTIATION_STANCE_DISPLAY.get(str(value).strip().lower(), value)
                if entry.get("hide_label"):
                    bits.append(str(display_value).replace("_", " ").capitalize())
                else:
                    bits.append(f"{label}: {display_value}")

        elif kind == "flag_equals":
            value = item.get(entry["item_key"])
            if value is not None and not pd.isna(value) and float(value) == entry["value"]:
                bits.append(f"{label}: Elevated")

        elif kind == "share_change_pct":
            current = item.get("current_customer_share")
            prior = item.get("prior_customer_share")
            if (
                current is None or prior is None or pd.isna(current) or pd.isna(prior)
                or abs(float(prior)) < 1e-9
            ):
                continue
            pct = 100 * (float(current) - float(prior)) / abs(float(prior))
            if abs(pct) < 0.5:  # ~flat — not genuinely "up" or "down"
                continue
            bits.append(f"{label}: {abs(pct):.0f}% {'more' if pct > 0 else 'less'}")

    return "; ".join(bits) if bits else None


# Strength qualifier for the win/loss lead, per the governed prediction-margin
# bands (own thresholds, not score_scenario()'s confidence band): >=20pp
# Strong, 10-20pp Moderate, 0-10pp Marginal, mirrored on the loss side.
def _prediction_strength(margin_pp: float) -> str:
    magnitude = abs(margin_pp)
    if magnitude >= 20:
        return "Strong"
    if magnitude >= 10:
        return "Moderate"
    return "Marginal"


def _predicted_outcome_sentence(
    item: dict[str, Any], medians: dict[str, dict[str, float]], raw_medians: dict[str, dict[str, float]]
) -> str:
    """"Predicted outcome" — prediction strength, P(Win), the top 2 drivers on
    the side matching the call (each backed by %-vs-median and, when it
    clears its own population median, its raw source column too — same
    _raw_detail() rule Account Context uses), then prior incumbent (only
    named when it is not Customer) and share-change direction.
    """
    product = str(item["product"])
    margin_pp = 100 * (item["win_probability"] - item["threshold"])
    is_win = margin_pp >= 0
    strength = _prediction_strength(margin_pp)
    pct = f"{item['win_probability']:.0%}"

    lead = f"{strength} predicted {'Win' if is_win else 'Loss'} — {pct} P(Win)"

    drivers = _top_shap_features(item.get("shap"), positive=is_win, limit=2)
    product_medians_ = medians.get(product, {})
    raw_medians_ = raw_medians.get(product, {})
    if drivers:
        bits = []
        for feature, _ in drivers:
            detail = _driver_detail(item, feature, product_medians_.get(feature))
            raw = _raw_detail(item, feature, raw_medians_)
            if raw:
                detail = f"{detail}; {raw}"
            bits.append(f"{_driver_phrase(item, feature, is_win)} ({detail})")
        verb = "supported" if is_win else "constrained"
        lead += f", primarily {verb} by {_join_list(bits)}"

    context = []
    incumbent = item.get("prior_incumbent")
    if incumbent and incumbent not in EMPTY_CATEGORICAL_VALUES and incumbent != "Customer":
        context.append(f"{incumbent} as prior incumbent")
    share_change = item.get("share_change")
    if share_change is not None and not pd.isna(share_change):
        context.append("declining market share" if share_change < 0 else "growing market share")
    if context:
        lead += f", with {_join_list(context)}"

    return f"<p><strong>Predicted outcome:</strong> {escape(lead)}.</p>"


def _top_drivers_bits(
    item: dict[str, Any], medians: dict[str, dict[str, float]], raw_medians: dict[str, dict[str, float]]
) -> list[str]:
    """The two best (positive-SHAP) and two worst (negative-SHAP) drivers,
    always both sides regardless of the predicted outcome, as Account
    Context bits. Each is backed by its composite %-vs-median and, when the
    account's own value for one of its raw source columns is above that
    column's population median (the threshold), the raw column and its
    value too — e.g. "account engagement (56%+; Engagement length: 7.82)".

    The Predicted Outcome sentence already names the top 2 drivers on the
    side matching the call (supporting for a Win, constraining for a Loss)
    — repeating those exact 2 here would just restate the same sentence, so
    this section skips them and shows ranks 3-4 on that side instead. The
    opposite side (not shown anywhere else) still gets its true top 2.
    """
    shap = item.get("shap")
    is_win = item["predicted_outcome"] == "Win"
    matching_side = _top_shap_features(shap, positive=is_win, limit=4)[2:4]
    opposite_side = _top_shap_features(shap, positive=not is_win, limit=2)
    best, worst = (matching_side, opposite_side) if is_win else (opposite_side, matching_side)
    if not best and not worst:
        return []

    product_medians_ = medians.get(str(item["product"]), {})
    raw_medians_ = raw_medians.get(str(item["product"]), {})

    def describe(feature: str, positive: bool) -> str:
        phrase = _driver_phrase(item, feature, positive)
        detail = _driver_detail(item, feature, product_medians_.get(feature))
        raw = _raw_detail(item, feature, raw_medians_)
        if raw:
            detail = f"{detail}; {raw}"
        return f"{phrase} ({detail})"

    bits = []
    if best:
        bits.append(f"supported by {_join_list([describe(feature, True) for feature, _ in best])}")
    if worst:
        bits.append(f"constrained by {_join_list([describe(feature, False) for feature, _ in worst])}")
    return bits


def _account_context_sentence(
    item: dict[str, Any], medians: dict[str, dict[str, float]], raw_medians: dict[str, dict[str, float]]
) -> str | None:
    """"Account context" — exactly the two best and two worst drivers (via
    SHAP, both sides always, regardless of predicted outcome), each backed
    by its composite %-vs-median and, when the account's own value clears
    that column's median, its raw source column and value too. No other
    account attributes are shown here — engagement/clinical/negotiation are
    only surfaced when SHAP actually ranks them as one of the 4 drivers.
    """
    bits = _top_drivers_bits(item, medians, raw_medians)
    if not bits:
        return None
    return f"<p><strong>Account context:</strong> {escape('; '.join(bits))}.</p>"


def _sensitivity_basis_phrase(item: dict[str, Any]) -> str:
    """What the account's band is graded on: the driving product's
    win-probability change when its price moves to the standard benchmark
    (account_sensitivity_basis is that product's benchmark_direction).
    Empty when unknown."""
    basis = item.get("account_sensitivity_basis")
    product = item.get("account_driving_product")
    if basis is None or pd.isna(basis) or product is None or pd.isna(product):
        return ""
    if basis == "at benchmark":
        return f"driven by {product}, already at its standard benchmark price"
    move = "raised" if basis == "price up to benchmark" else "lowered"
    return f"driven by {product}, graded on its price being {move} to the standard benchmark"


def _price_sensitivity_sentence(item: dict[str, Any]) -> str:
    """"Price sensitivity" — the account-level Low/Medium/High band
    (ACCOUNT_SENSITIVITY_BAND_COLUMN) with its own confidence label
    (ACCOUNT_CONFIDENCE_LABEL_COLUMN, same source), the governed symmetric
    +/-10% response (symmetric_price_sensitivity() — the primary
    business-facing sensitivity measure, not a smallest-tested-step search),
    and the account's price position vs. the product median, the standard
    offer, and the lowest competitor.
    """
    band = item.get("account_sensitivity_band")
    label = item.get("account_confidence_label")
    confidence_clause = f" ({label} confidence)" if label and not pd.isna(label) else ""
    lead = f"{band}{confidence_clause}" if band else "Not classified"
    basis = _sensitivity_basis_phrase(item)
    if band and basis:
        lead += f", {basis}"

    sens = item.get("symmetric_sensitivity")
    if sens is not None:
        lead += (
            f" — a 10% price increase changes modeled P(Win) by "
            f"{sens['upward_price_risk_pp']:+.1f} pp, while a 10% decrease provides "
            f"{sens['downward_price_opportunity_pp']:+.1f} pp"
        )

    facts = []
    median_pct = item.get("price_vs_median_pct")
    if median_pct is not None and not pd.isna(median_pct):
        facts.append(f"{abs(median_pct):.0f}% {'above' if median_pct >= 0 else 'below'} the product median")
    offered = item.get("features", {}).get("price_vs_std_benchmark_updated")
    if offered is not None and not pd.isna(offered):
        facts.append(f"{abs(offered):.0f}% {'below' if offered >= 0 else 'above'} the standard offer")
    premium = item.get("features", {}).get("premium_gap_vs_min_competitor_pct")
    if premium is not None and not pd.isna(premium):
        facts.append(f"{abs(premium):.0f}% {'above' if premium >= 0 else 'below'} the lowest competitor")
    if facts:
        lead += f". Current price is {_join_list(facts)}"

    return f"<p><strong>Price sensitivity:</strong> {escape(lead)}.</p>"


# Response archetype (spec-style), from the symmetric +/-10% response, at a
# 2pp materiality threshold — the same shape as the 2x2 quadrant's own cut.
RESPONSE_ARCHETYPE_TEXT = {
    "two_sided": "Price is an actionable lever in both directions",
    "upward_exposed": (
        "Higher pricing weakens the position, but deeper discounting alone is "
        "unlikely to materially improve the modeled outcome"
    ),
    "discount_opportunity": "Lower pricing provides meaningful modeled upside despite limited exposure to a price increase",
    "resilient": "Modeled win probability remains relatively stable around the current price",
}

RESPONSE_ARCHETYPE_THRESHOLD_PP = 2.0

# Non-price action mapped to each driver, for the "beyond price" clause —
# only surfaced when that driver is among the account's top constraints.
# Each individual driver maps to the business area it used to roll up into.
NON_PRICE_ACTION = {
    "bob_Prior Year Customer Doses": "relationship",
    "bob_Prior Season Shr of Wallet_enc": "relationship",
    "apt_clinical_bucket_enc": "clinical engagement",
    "bob_Prior Season HS Partnership Advocacy_enc": "clinical engagement",
    "sam_feat_high_or_medium_risk_flag": "competitive positioning",
    "no_of_flips_cumulative": "competitive positioning",
    "yoy_ms_change_enc": "competitive positioning",
    "cde_engagement_length_log": "engagement",
    "cde_net_sentiment_score": "engagement",
    "negotiation_receptivity_score": "negotiation approach",
    "customer_provider_margin": "provider economics",
    "customer_doses_to_prebook_ratio_corrected": "prebook execution",
    # v27 drivers
    "log_prior_customer_doses": "relationship",
    "sam_risk_flag_product": "competitive positioning",
    "flips_prior_2_seasons": "competitive positioning",
    "yoy_ms_change_pct_prior": "competitive positioning",
    "buying_group_category_cat": "contracting structure",
    "multi_year_flag": "contracting structure",
}


def _response_archetype(upward_pp: float, downward_pp: float) -> str:
    up = upward_pp >= RESPONSE_ARCHETYPE_THRESHOLD_PP
    down = downward_pp >= RESPONSE_ARCHETYPE_THRESHOLD_PP
    if up and down:
        return "two_sided"
    if up:
        return "upward_exposed"
    if down:
        return "discount_opportunity"
    return "resilient"


def _implication_sentence(item: dict[str, Any]) -> str | None:
    """"Implication" — the response archetype's canned commercial read, plus
    (when the account is predicted to lose) which non-price areas are worth
    strengthening, drawn from the same top-2 constraining drivers as the
    Predicted outcome sentence."""
    sens = item.get("symmetric_sensitivity")
    if sens is None:
        return None
    archetype = _response_archetype(sens["upward_price_risk_pp"], sens["downward_price_opportunity_pp"])
    text = RESPONSE_ARCHETYPE_TEXT[archetype]

    if item["predicted_outcome"] == "Loss":
        worst = _top_shap_features(item.get("shap"), positive=False, limit=2)
        actions = [NON_PRICE_ACTION[feature] for feature, _ in worst if feature in NON_PRICE_ACTION]
        if actions:
            text += (
                f"; beyond price, improving {_join_list(actions)} is likely more important to "
                f"strengthening the modeled outcome"
            )

    return f"<p><strong>Implication:</strong> {escape(text)}.</p>"


def _build_item_narrative(
    item: dict[str, Any], medians: dict[str, dict[str, float]], raw_medians: dict[str, dict[str, float]]
) -> str:
    """Per product, four labeled sections: Predicted outcome, Account
    context, Price sensitivity, Implication."""
    sections = [_predicted_outcome_sentence(item, medians, raw_medians)]
    context = _account_context_sentence(item, medians, raw_medians)
    if context:
        sections.append(context)
    sections.append(_price_sensitivity_sentence(item))
    implication = _implication_sentence(item)
    if implication:
        sections.append(implication)
    return "".join(sections)


def build_narrative(
    result: dict[str, Any],
    customer: str,
    product_filter: str | None,
    medians: dict[str, dict[str, float]],
    raw_medians: dict[str, dict[str, float]],
) -> str:
    """Deterministic, rule-based scenario summary — four labeled sections per
    product (Predicted outcome, Account context, Price sensitivity,
    Implication). Same scenario in, same text out — no model or LLM call, so
    it stays auditable and works with no AI access.

    ``product_filter`` narrows the story to that one product — it shares the
    same picker as the SHAP curve, so the two stay in sync.
    """
    items = result["product_results"]
    if product_filter is not None:
        items = [item for item in items if str(item["product"]) == product_filter]
    customer_html = escape(str(customer)) if customer else "This account"

    if items:
        return "".join(_build_item_narrative(item, medians, raw_medians) for item in items)
    return f"No prediction available for {customer_html}."
