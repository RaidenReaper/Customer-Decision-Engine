"""Deterministic, rule-based "business-friendly" scenario narrative — four
labeled sections per product: Expected outcome, Account signals, Price
sensitivity, Recommended action. Same underlying model output as the
technical narrative (SHAP drivers, symmetric price sensitivity, account
sensitivity band) but no feature values or %-vs-median figures — plain
language a non-technical reader can act on. No model or LLM call, so it
stays auditable and reproducible: same scenario in, same text out.

This is one of two interchangeable narrative styles — see
``src/narrative_technical.py`` for the fuller version with feature values
and raw-column backing. Which one renders in the app is a single constant,
``NARRATIVE_STYLE`` in app.py; see the note at the bottom of this module for
how to switch it.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from src.utils import EMPTY_CATEGORICAL_VALUES, SEGMENTATION_FEATURE

# Reuses the exact same driver-selection rule as the technical narrative
# (ranked by |shap|) so the two styles always agree on which features are
# the top drivers for a given account.
from src.narrative_technical import (
    RESPONSE_ARCHETYPE_THRESHOLD_PP,
    _prediction_strength,
    _response_archetype,
    _sensitivity_basis_phrase,
    _top_shap_features,
)

# Non-price action mapped to each driver, for the "beyond price" clause on
# predicted losses. Own copy (not shared with the technical narrative's
# NON_PRICE_ACTION) so its wording can diverge — e.g. "prebook conversion"
# here reads better in a business sentence than the technical module's
# "prebook execution".
#
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
    "customer_provider_margin": "provider margin",
    "customer_doses_to_prebook_ratio_corrected": "prebook conversion",
    # v27 drivers
    "log_prior_customer_doses": "relationship",
    "sam_risk_flag_product": "competitive positioning",
    "flips_prior_2_seasons": "competitive positioning",
    "yoy_ms_change_pct_prior": "competitive positioning",
    "buying_group_category_cat": "contracting structure",
    "multi_year_flag": "contracting structure",
}


def _join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# Plain-language phrase for each driver, already signed toward "good" vs.
# "bad" — unlike the technical narrative's neutral DRIVER_PHRASES (which
# leans on the %-vs-median figure to carry direction), this version shows no
# numbers, so the phrase itself has to carry the direction.
#
# Every model feature except price_vs_std_benchmark_updated and the
# segmentation flag (both phrased from their own value in _driver_phrase())
# must have an entry: a top driver without one raises.
BUSINESS_DRIVER_PHRASES = {
    "bob_Prior Year Customer Doses": {
        "supporting": "an established prior-year volume",
        "constraining": "limited prior-year volume",
    },
    "bob_Prior Season Shr of Wallet_enc": {
        "supporting": "a strong customer share of wallet",
        "constraining": "a weaker customer share of wallet",
    },
    "bob_Prior Season HS Partnership Advocacy_enc": {
        "supporting": "strong partnership advocacy",
        "constraining": "limited partnership advocacy",
    },
    "apt_clinical_bucket_enc": {
        "supporting": "a clinically focused account",
        "constraining": "a price-focused account",
    },
    "sam_feat_high_or_medium_risk_flag": {
        "supporting": "a low account risk flag",
        "constraining": "an elevated account risk flag",
    },
    "cde_engagement_length_log": {
        "supporting": "healthy recent engagement",
        "constraining": "weaker account engagement",
    },
    "cde_net_sentiment_score": {
        "supporting": "positive call sentiment",
        "constraining": "weaker call sentiment",
    },
    "negotiation_receptivity_score": {
        "supporting": "a favorable negotiation stance",
        "constraining": "a difficult negotiation stance",
    },
    "no_of_flips_cumulative": {
        "supporting": "a stable supplier history",
        "constraining": "a history of switching suppliers",
    },
    "yoy_ms_change_enc": {
        "supporting": "a favorable market share trend",
        "constraining": "a weakening market share trend",
    },
    "customer_doses_to_prebook_ratio_corrected": {
        "supporting": "strong prebook conversion",
        "constraining": "weaker prebook conversion",
    },
    "customer_provider_margin": {
        "supporting": "healthy provider margin",
        "constraining": "thin provider margin",
    },
    "premium_gap_vs_min_competitor_pct": {
        "supporting": "favorable competitive price positioning",
        "constraining": "a competitive price disadvantage",
    },
    "yoy_price_change_pct": {
        "supporting": "stable year-over-year pricing",
        "constraining": "a year-over-year price increase",
    },
    # v27 drivers
    "log_prior_customer_doses": {
        "supporting": "an established prior-year volume",
        "constraining": "limited prior-year volume",
    },
    "sam_risk_flag_product": {
        "supporting": "a low account risk flag",
        "constraining": "an elevated account risk flag",
    },
    "flips_prior_2_seasons": {
        "supporting": "a stable supplier history",
        "constraining": "a history of switching suppliers",
    },
    "yoy_ms_change_pct_prior": {
        "supporting": "a favorable market share trend",
        "constraining": "a weakening market share trend",
    },
    "buying_group_category_cat": {
        "supporting": "the account's buying-group affiliation",
        "constraining": "the account's buying-group affiliation",
    },
    "multi_year_flag": {
        "supporting": "the account's contract term",
        "constraining": "the account's contract term",
    },
    "season_cat": {
        "supporting": "the season effect",
        "constraining": "the season effect",
    },
}


def _driver_phrase(item: dict[str, Any], feature: str, positive: bool) -> str:
    """Plain-language phrase for one driver. price_vs_std_benchmark_updated
    branches on the raw value's own sign, same rule the technical narrative
    uses, just worded for a non-technical reader. The segmentation flag is a
    category with no good/bad direction, so it is named by its own label and
    the sentence it lands in carries the direction."""
    if feature == SEGMENTATION_FEATURE:
        label = item.get("features", {}).get(feature)
        if label is None or pd.isna(label) or str(label) in EMPTY_CATEGORICAL_VALUES:
            return "its HS segmentation"
        return f"its {str(label).title()} HS segmentation"
    if feature == "price_vs_std_benchmark_updated":
        raw_value = item.get("features", {}).get(feature)
        if raw_value is None or pd.isna(raw_value) or raw_value == 0:
            return "pricing at the standard offer"
        if raw_value > 0:
            return "a price below the standard offer" if positive else "pricing above the standard offer"
        return "a price above the standard offer" if positive else "pricing below the standard offer"
    return BUSINESS_DRIVER_PHRASES[feature]["supporting" if positive else "constraining"]


def _expected_outcome_sentence(item: dict[str, Any], show_probability: bool = True) -> str:
    """"Expected outcome" — strength + call in plain words, the P(Win)
    number, then the top 2 drivers on the side matching the call (named
    plainly, no %-vs-median), then incumbent/share-change context.
    ``show_probability=False`` leaves the P(Win) sentence out."""
    margin_pp = 100 * (item["win_probability"] - item["threshold"])
    is_win = margin_pp >= 0
    strength = _prediction_strength(margin_pp)
    pct = f"{item['win_probability']:.0%}"

    headline = {
        ("Strong", True): "Strong win likely",
        ("Moderate", True): "Likely win",
        ("Marginal", True): "Marginal win — close call",
        ("Strong", False): "Strong loss likely",
        ("Moderate", False): "Likely loss",
        ("Marginal", False): "Marginal loss — close call",
    }[(strength, is_win)]

    sentences = [f"{headline}."]
    if show_probability:
        sentences.append(f"The model estimates a {pct} chance of winning for this account.")

    drivers = _top_shap_features(item.get("shap"), positive=is_win, limit=2)
    if drivers:
        names = [_driver_phrase(item, feature, is_win) for feature, _ in drivers]
        label = "strongest positive signals" if is_win else "strongest headwinds"
        sentences.append(f"The {label} {'is' if len(names) == 1 else 'are'} {_join_list(names)}.")

    context = []
    incumbent = item.get("prior_incumbent")
    if incumbent and incumbent not in EMPTY_CATEGORICAL_VALUES and incumbent != "Customer":
        context.append(f"{incumbent} remains the prior incumbent")
    share_change = item.get("share_change")
    if share_change is not None and not pd.isna(share_change):
        context.append("Market share is declining" if share_change < 0 else "Market share is also growing")
    if context:
        joined = _join_list(context)
        sentences.append(f"{joined[0].upper()}{joined[1:]}.")

    return f"<p><strong>Expected outcome:</strong> {escape(' '.join(sentences))}</p>"


def _account_signals_sentence(item: dict[str, Any]) -> str | None:
    """"Account signals" — the same 2-best/2-worst driver selection as the
    technical narrative's Account context (ranks 3-4 on the matching side,
    to avoid repeating Expected outcome; true top 2 on the opposite side),
    named plainly with no numbers."""
    shap = item.get("shap")
    is_win = item["predicted_outcome"] == "Win"
    matching_side = _top_shap_features(shap, positive=is_win, limit=4)[2:4]
    opposite_side = _top_shap_features(shap, positive=not is_win, limit=2)
    best, worst = (matching_side, opposite_side) if is_win else (opposite_side, matching_side)
    if not best and not worst:
        return None

    sentences = []
    if best:
        names = _join_list([_driver_phrase(item, feature, True) for feature, _ in best])
        sentences.append(f"This account has several supportive signals, including {names}.")
    if worst:
        names = _join_list([_driver_phrase(item, feature, False) for feature, _ in worst])
        sentences.append(f"Watchouts include {names}.")

    return f"<p><strong>Account signals:</strong> {escape(' '.join(sentences))}</p>"


def _price_sensitivity_sentence(item: dict[str, Any]) -> str:
    """"Price sensitivity" — the account-level band and confidence label in
    plain wording, the governed symmetric +/-10% response in
    percentage-point terms, and price position vs. the standard offer and
    the lowest competitor (the two genuine commercial reference points —
    the product-median comparison is a modeling construct, so it's left to
    the technical narrative)."""
    band = item.get("account_sensitivity_band")
    label = item.get("account_confidence_label")
    if band:
        lead = f"{band} price sensitivity"
        if label and not pd.isna(label):
            lead += f", {str(label).lower()} confidence"
        basis = _sensitivity_basis_phrase(item)
        if basis:
            lead += f" ({basis})"
    else:
        lead = "Price sensitivity not classified"
    sentences = [f"{lead}."]

    sens = item.get("symmetric_sensitivity")
    if sens is not None:
        up = sens["upward_price_risk_pp"]
        down = sens["downward_price_opportunity_pp"]
        sentences.append(
            f"A 10% price increase is expected to lower win likelihood by {abs(up):.1f} percentage points."
        )
        sentences.append(
            f"A 10% price decrease is expected to improve win likelihood by {abs(down):.1f} percentage points."
        )

    facts = []
    offered = item.get("features", {}).get("price_vs_std_benchmark_updated")
    if offered is not None and not pd.isna(offered):
        if abs(offered) < 2:
            facts.append("near the standard offer")
        else:
            facts.append(f"about {abs(offered):.0f}% {'below' if offered >= 0 else 'above'} the standard offer")
    premium = item.get("features", {}).get("premium_gap_vs_min_competitor_pct")
    if premium is not None and not pd.isna(premium):
        if abs(premium) < 2:
            facts.append("near the lowest competitor's price")
        else:
            facts.append(f"about {abs(premium):.0f}% {'above' if premium >= 0 else 'below'} the lowest competitor")
    if facts:
        sentences.append(f"Current price is {_join_list(facts)}.")

    return f"<p><strong>Price sensitivity:</strong> {escape(' '.join(sentences))}</p>"


# Recommendation-toned read of the response archetype (same +/-10%
# thresholds as the technical narrative's RESPONSE_ARCHETYPE_TEXT).
BUSINESS_RESPONSE_ACTION_TEXT = {
    "two_sided": "Treat price as an active lever in both directions — moves here should meaningfully affect the outcome",
    "upward_exposed": (
        "Avoid raising price; deeper discounting alone is unlikely to meaningfully change the outcome, "
        "so focus on non-price value instead"
    ),
    "discount_opportunity": "A price reduction offers meaningful upside here with limited downside risk at current levels",
    "resilient": "Price has limited influence on this account; focus on non-price factors to shift the outcome",
}


def _recommended_action_sentence(item: dict[str, Any]) -> str | None:
    """"Recommended action" — the response archetype's recommendation-toned
    read, a confidence caveat when the account's sensitivity confidence is
    Low, an upside-protection clause when there's upward price exposure,
    and (when predicted to lose) which non-price areas to strengthen."""
    sens = item.get("symmetric_sensitivity")
    if sens is None:
        return None
    up = sens["upward_price_risk_pp"]
    down = sens["downward_price_opportunity_pp"]
    archetype = _response_archetype(up, down)
    text = BUSINESS_RESPONSE_ACTION_TEXT[archetype]

    label = item.get("account_confidence_label")
    if label and not pd.isna(label) and str(label).lower() == "low":
        text += ", but validate with account context because confidence is low"
    sentences = [f"{text}."]

    is_win = item["predicted_outcome"] == "Win"
    if up >= RESPONSE_ARCHETYPE_THRESHOLD_PP:
        goal = "protecting the win" if is_win else "improving the position"
        sentences.append(
            f"If {goal} is the priority, avoid upward price movement unless supported by stronger "
            f"non-price value messaging."
        )

    if not is_win:
        worst = _top_shap_features(item.get("shap"), positive=False, limit=2)
        actions = [NON_PRICE_ACTION[feature] for feature, _ in worst if feature in NON_PRICE_ACTION]
        if actions:
            sentences.append(f"Beyond price, strengthening {_join_list(actions)} is likely the bigger lever.")

    return f"<p><strong>Recommended action:</strong> {escape(' '.join(sentences))}</p>"


def _build_item_narrative(item: dict[str, Any], show_probability: bool = True) -> str:
    """Per product, four labeled sections: Expected outcome, Account
    signals, Price sensitivity, Recommended action."""
    sections = [_expected_outcome_sentence(item, show_probability)]
    signals = _account_signals_sentence(item)
    if signals:
        sections.append(signals)
    sections.append(_price_sensitivity_sentence(item))
    action = _recommended_action_sentence(item)
    if action:
        sections.append(action)
    return "".join(sections)


def build_narrative(
    result: dict[str, Any],
    customer: str,
    product_filter: str | None,
    show_probability: bool = True,
) -> str:
    """Deterministic, rule-based scenario summary — four labeled sections per
    product (Expected outcome, Account signals, Price sensitivity,
    Recommended action). Same scenario in, same text out — no model or LLM
    call, so it stays auditable and works with no AI access.

    ``product_filter`` narrows the story to that one product — it shares the
    same picker as the SHAP curve, so the two stay in sync. Unlike the
    technical narrative, this version needs no medians — it never shows a
    %-vs-median figure.
    """
    items = result["product_results"]
    if product_filter is not None:
        items = [item for item in items if str(item["product"]) == product_filter]
    customer_html = escape(str(customer)) if customer else "This account"

    if items:
        return "".join(_build_item_narrative(item, show_probability) for item in items)
    return f"No prediction available for {customer_html}."
