"""Win/loss prediction confidence, v5 (11/13).

    confidence = 0.30 x D + 0.30 x H + 0.30 x M + 0.10 x S

    D  Data completeness & quality  - are the account's inputs recorded, current values?
    H  Historical similarity        - do this account's values (and the scenario's price)
                                      sit inside the ranges the model was trained on?
    M  Decision margin              - how far the win probability sits from the product's
                                      own Win/Loss threshold (stepped, in points)
    S  Prediction stability         - do the 5 out-of-fold model versions agree?

Bands: High >= 85, Moderate 60-84, Low < 60.

Source: Win_Loss_Confidence_Scoring_Implementation_Guide.md (v1.0, 13 Nov 2026).
Changes from the guide (user decisions, 11/16): M is a stepped score on the
distance from the threshold in points (<5: 0, 5-10: 25, 10-15: 50, 15-20: 75,
>=20: 100) instead of section 6.1's scaled formula; High starts at 85, not 80.
Replaces v4 (0.25 stability + 0.25 fixed product reliability + 0.5 margin, with a
price-neighbour gate): the fixed reliability term is dropped and H takes over the
neighbour gate's job of flagging prices the model has not seen.

The score is a decision-support index, NOT the probability that the prediction is
right. Every weight, band and cut-off below is a provisional policy choice from the
guide and should be revisited after backtesting.

Everything here is pure (no Streamlit, no I/O) so it can be unit-tested directly.
src/scoring.py wires it to the models; src/prepare_real_data.py builds the
provenance columns and the historical reference tables this module reads.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    from src.utils import FEATURE_LABELS
except ModuleNotFoundError:  # running as a script from inside src/
    from utils import FEATURE_LABELS

METHOD_VERSION = "confidence-v5.1"  # 5.1 (11/16): stepped margin, High >= 85

# --- weights ------------------------------------------------------------------

FACTOR_WEIGHTS = {"D": 0.30, "H": 0.30, "M": 0.30, "S": 0.10}
FACTOR_NAMES = {
    "D": "Data completeness",
    "H": "Historical similarity",
    "M": "Decision margin",
    "S": "Model agreement",
}

# The four business groups of the Simulation screen (same membership as
# DRIVER_CATEGORIES in app.py — tests/test_confidence.py keeps them in step).
# season_cat is not a business input and is in no group.
GROUPS: dict[str, dict[str, Any]] = {
    "relationship": {
        "label": "Relationship with Customer",
        "weight": 0.30,
        "features": [
            "cde_engagement_length_log",
            "cde_net_sentiment_score",
            "sam_risk_flag_product",
            "log_prior_customer_doses",
        ],
    },
    "loyalty": {
        "label": "Customer Loyalty",
        "weight": 0.25,
        "features": [
            "bob_Prior Season Shr of Wallet_enc",
            "flips_prior_2_seasons",
            "yoy_ms_change_pct_prior",
            "multi_year_flag",
        ],
    },
    "price": {
        "label": "Price Sensitivity",
        "weight": 0.30,
        "features": [
            "price_vs_std_benchmark_updated",
            "premium_gap_vs_min_competitor_pct",
            "yoy_price_change_pct",
            "negotiation_receptivity_score",
            "buying_group_category_cat",
        ],
    },
    "clinical": {
        "label": "Clinical Inclination",
        "weight": 0.15,
        "features": [
            "apt_clinical_bucket_enc",
            "bob_Prior Season HS Partnership Advocacy_enc",
        ],
    },
}
FEATURE_GROUP = {feature: group for group, spec in GROUPS.items() for feature in spec["features"]}
CONFIDENCE_FEATURES = list(FEATURE_GROUP)

# Price-linked features: recomputed by the price lever (build_feature_row).
PRICE_LINKED = (
    "price_vs_std_benchmark_updated",
    "premium_gap_vs_min_competitor_pct",
    "yoy_price_change_pct",
)

# --- provenance (D) -----------------------------------------------------------

OBSERVED = "observed"            # a recorded value for the period the model expects
ENRICHED = "enriched"            # a recorded value from a verified second source
MANUAL = "manual"                # set by the user on a Contracting Lever
CARRY_FORWARD = "carry_forward"  # last season's value copied into the scored season
IMPUTED = "imputed"              # an estimate, not a recording
MISSING = "missing"              # no value
NOT_APPLICABLE = "not_applicable"  # absent by design (e.g. no prior season for a new account)
SCENARIO = "scenario_input"      # computed from the user's simulated price

# Parquet column holding each row's code for one input: "prov__<feature>".
PROVENANCE_PREFIX = "prov__"

# (completeness, source reliability) per code; Q = 0.6 x C + 0.4 x R.
PROVENANCE_SCORES = {
    OBSERVED: (100.0, 100.0),
    ENRICHED: (100.0, 90.0),
    MANUAL: (100.0, 85.0),
    CARRY_FORWARD: (0.0, 60.0),
    IMPUTED: (0.0, 40.0),
    MISSING: (0.0, 0.0),
}
COMPLETENESS_WEIGHT = 0.60
RELIABILITY_WEIGHT = 0.40
# Not part of D at all (neither numerator nor denominator).
EXCLUDED_FROM_D = {NOT_APPLICABLE, SCENARIO}
# Only these are compared with history in H; carried, imputed and missing
# values are "applicable but not assessable" and lower H's coverage instead.
ASSESSABLE_FOR_H = {OBSERVED, ENRICHED, MANUAL, SCENARIO}
KNOWN_CODES = set(PROVENANCE_SCORES) | EXCLUDED_FROM_D
# Which issue the explanation names first.
ISSUE_SEVERITY = {MISSING: 0, IMPUTED: 1, CARRY_FORWARD: 2, MANUAL: 3, ENRICHED: 4}

# --- historical similarity (H) ------------------------------------------------

# Compared by category frequency, not percentiles (few distinct values).
DISCRETE_FEATURES = {
    "bob_Prior Season Shr of Wallet_enc",
    "bob_Prior Season HS Partnership Advocacy_enc",
    "apt_clinical_bucket_enc",
    "sam_risk_flag_product",
    "flips_prior_2_seasons",
    "negotiation_receptivity_score",
    "buying_group_category_cat",
    "multi_year_flag",
}
QUANTILES = (0.01, 0.05, 0.10, 0.90, 0.95, 0.99)
# A continuous feature with at least this share of exact zeros in training is
# zero-inflated: an exact 0 is scored by how common 0 is, any other value by
# percentiles of the non-zero values (guide 5.3 "degenerate cases").
ZERO_INFLATED_SHARE = 0.05
# A "continuous" feature with this many distinct values or fewer is scored as discrete.
MAX_DISCRETE_UNIQUE = 10
MIN_REFERENCE_ROWS = 20

COVERAGE_FULL = 0.70
COVERAGE_PARTIAL = 0.40
CAP_PARTIAL = 60.0
CAP_LOW = 30.0

# --- bands, explanation -------------------------------------------------------

# High raised from 80 to 85 (user decision 11/16, with the stepped margin).
BAND_HIGH = 85.0
BAND_MODERATE = 60.0

# --- decision margin (M) steps (user decision 11/16) ---------------------------
# (upper bound in points from the threshold, exclusive; score below it).
MARGIN_STEPS = ((5.0, 0.0), (10.0, 25.0), (15.0, 50.0), (20.0, 75.0))
MARGIN_FULL = 100.0  # 20 points or more
NEAR_THRESHOLD_PP = 5.0
CAVEAT = (
    "Confidence shows how well-supported this prediction is by the account's data "
    "and by the model. It is not the chance that the prediction is right."
)


# =============================================================================
# D — data completeness & quality
# =============================================================================


def field_quality(code: str) -> float | None:
    """Q for one input, 0-100; None when the input is excluded from D."""
    if code in EXCLUDED_FROM_D:
        return None
    if code not in PROVENANCE_SCORES:
        raise ValueError(f"Unknown provenance code: {code!r}")
    completeness, reliability = PROVENANCE_SCORES[code]
    return COMPLETENESS_WEIGHT * completeness + RELIABILITY_WEIGHT * reliability


def _weighted_group_mean(group_scores: Mapping[str, float]) -> float:
    if not group_scores:
        raise ValueError("No applicable feature groups")
    denominator = sum(GROUPS[g]["weight"] for g in group_scores)
    return sum(GROUPS[g]["weight"] * v for g, v in group_scores.items()) / denominator


def data_quality(provenance: Mapping[str, str]) -> dict[str, Any]:
    """D from per-input provenance codes.

    Missing inputs stay in their group's denominator; only not-applicable and
    scenario inputs are left out, and a group with no applicable input drops
    out of the weighting (never a group merely because its data is missing).
    """
    grouped: dict[str, list[float]] = {}
    fields = []
    for feature in CONFIDENCE_FEATURES:
        code = provenance.get(feature, MISSING)
        score = field_quality(code)
        fields.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "group": FEATURE_GROUP[feature],
                "code": code,
                "quality": score,
            }
        )
        if score is not None:
            grouped.setdefault(FEATURE_GROUP[feature], []).append(score)
    group_scores = {g: sum(v) / len(v) for g, v in grouped.items()}
    issues = sorted(
        (f for f in fields if f["quality"] is not None and f["quality"] < 100),
        key=lambda f: (ISSUE_SEVERITY.get(f["code"], 9), -GROUPS[f["group"]]["weight"]),
    )
    counts: dict[str, int] = {}
    for f in fields:
        counts[f["code"]] = counts.get(f["code"], 0) + 1
    return {
        "score": _weighted_group_mean(group_scores),
        "group_scores": group_scores,
        "fields": fields,
        "issues": issues,
        "counts": counts,
    }


# =============================================================================
# H — historical similarity
# =============================================================================


def category_key(value: Any) -> str:
    """Stable string key for a discrete value (0.75 -> '0.75', 1.0 -> '1')."""
    return f"{round(float(value), 4):g}"


def _frequencies(values: pd.Series) -> dict[str, float]:
    keys = values.map(category_key)
    return {str(k): round(float(v), 6) for k, v in keys.value_counts(normalize=True).items()}


def build_feature_reference(values: Iterable[Any], discrete: bool) -> dict[str, Any] | None:
    """Training-only reference for one feature of one product (guide 5.2).

    Missing values are left out — they are not a "value seen in history".
    """
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(series) < MIN_REFERENCE_ROWS:
        return None
    n = int(len(series))
    if discrete or series.nunique() <= MAX_DISCRETE_UNIQUE:
        return {"type": "discrete", "n": n, "freq": _frequencies(series)}
    zero_share = float((series == 0).mean())
    reference: dict[str, Any] = {"type": "numeric", "n": n, "zero_share": round(zero_share, 6)}
    base = series
    if zero_share >= ZERO_INFLATED_SHARE:
        base = series[series != 0]
        reference["zero_inflated"] = True
        if len(base) < MIN_REFERENCE_ROWS:
            return {"type": "discrete", "n": n, "freq": _frequencies(series)}
    reference["quantiles"] = [round(float(q), 6) for q in base.quantile(list(QUANTILES))]
    return reference


def build_historical_reference(
    training: pd.DataFrame,
    product_column: str,
    products: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """{product: {feature: reference}} over the TRAINING rows only."""
    result: dict[str, dict[str, Any]] = {}
    for product in products:
        pop = training[training[product_column] == product]
        result[product] = {}
        for feature in CONFIDENCE_FEATURES:
            reference = build_feature_reference(pop[feature], feature in DISCRETE_FEATURES)
            if reference is not None:
                result[product][feature] = reference
    return result


def frequency_familiarity(share: float) -> float:
    if share >= 0.05:
        return 100.0
    if share >= 0.01:
        return 75.0
    if share > 0:
        return 40.0
    return 0.0


def percentile_familiarity(value: float, quantiles: list[float]) -> float:
    p1, p5, p10, p90, p95, p99 = quantiles
    if p10 <= value <= p90:
        return 100.0
    if p5 <= value <= p95:
        return 75.0
    if p1 <= value <= p99:
        return 40.0
    return 10.0


def feature_familiarity(value: float, reference: Mapping[str, Any]) -> float:
    """0-100: how common this value is among training rows of the product."""
    if reference["type"] == "discrete":
        return frequency_familiarity(reference["freq"].get(category_key(value), 0.0))
    if reference.get("zero_inflated") and value == 0:
        return frequency_familiarity(reference["zero_share"])
    return percentile_familiarity(float(value), reference["quantiles"])


def coverage_cap(raw: float, coverage: float) -> tuple[float, bool]:
    """Guide 5.5: too few comparable inputs cannot earn a high H."""
    if coverage <= 0:
        return 0.0, True
    if coverage < COVERAGE_PARTIAL:
        return min(raw, CAP_LOW), raw > CAP_LOW
    if coverage < COVERAGE_FULL:
        return min(raw, CAP_PARTIAL), raw > CAP_PARTIAL
    return raw, False


def historical_similarity(
    features: Mapping[str, Any],
    provenance: Mapping[str, str],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """H for one scenario. Uses the scenario's own feature values, so a moved
    price lever is compared with history automatically."""
    applicable: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    per_feature = []
    for feature in CONFIDENCE_FEATURES:
        code = provenance.get(feature, MISSING)
        if code == NOT_APPLICABLE:
            continue
        group = FEATURE_GROUP[feature]
        applicable[group] = applicable.get(group, 0) + 1
        value = features.get(feature)
        ref = reference.get(feature)
        if code not in ASSESSABLE_FOR_H or ref is None or value is None or pd.isna(value):
            continue
        score = feature_familiarity(float(value), ref)
        scores.setdefault(group, []).append(score)
        per_feature.append(
            {"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "group": group, "familiarity": score}
        )

    if not applicable:
        return {"score": 0.0, "raw_score": 0.0, "coverage": 0.0, "cap_applied": True,
                "group_scores": {}, "features": [], "unusual": []}
    group_scores = {g: sum(v) / len(v) for g, v in scores.items()}
    weight_sum = sum(GROUPS[g]["weight"] for g in applicable)
    coverage = sum(
        GROUPS[g]["weight"] * len(scores.get(g, [])) / count for g, count in applicable.items()
    ) / weight_sum
    raw = _weighted_group_mean(group_scores) if group_scores else 0.0
    score, capped = coverage_cap(raw, coverage)
    unusual = sorted((f for f in per_feature if f["familiarity"] <= 40), key=lambda f: f["familiarity"])
    return {
        "score": score,
        "raw_score": raw,
        "coverage": coverage,
        "cap_applied": capped,
        "group_scores": group_scores,
        "features": per_feature,
        "unusual": unusual,
    }


# =============================================================================
# M — decision margin, S — stability
# =============================================================================


def decision_margin(win_probability: float, threshold: float) -> float:
    """Stepped score on the distance from the product's threshold, in
    percentage points, either side (user decision 11/16, replacing the guide's
    6.1 formula, which only reached 100 at 0% / 100% and so scored a clear
    22-point buffer at 39): see MARGIN_STEPS. Each edge belongs to the higher
    step (exactly 5 points scores 25, exactly 20 scores 100)."""
    if not (0 < threshold < 1):
        raise ValueError("threshold must be strictly between 0 and 1")
    p = min(max(float(win_probability), 0.0), 1.0)
    # Rounded so 0.61 - 0.56 lands on 5 points despite float error.
    gap_pp = round(100.0 * abs(p - threshold), 9)
    for upper, score in MARGIN_STEPS:
        if gap_pp < upper:
            return score
    return MARGIN_FULL


def stability(fold_probabilities: Iterable[float], threshold: float) -> dict[str, Any]:
    """S from the spread (max - min, in points) of the 5 out-of-fold models'
    win probabilities, capped at 50 when they disagree on Win vs Loss."""
    probabilities = [float(p) for p in fold_probabilities]
    if not probabilities:
        raise ValueError("No fold-model probabilities")
    spread_pp = 100.0 * (max(probabilities) - min(probabilities))
    if spread_pp < 2:
        score = 100.0
    elif spread_pp <= 5:
        score = 80.0
    elif spread_pp <= 10:
        score = 50.0
    else:
        score = 20.0
    disagreement = len({p >= threshold for p in probabilities}) > 1
    if disagreement:
        score = min(score, 50.0)
    return {
        "score": score,
        "spread_pp": spread_pp,
        "disagreement": disagreement,
        "fold_probabilities": probabilities,
        "n_models": len(probabilities),
    }


# =============================================================================
# combine + explain
# =============================================================================


def band_for(score: float) -> str:
    if score >= BAND_HIGH:
        return "High"
    if score >= BAND_MODERATE:
        return "Moderate"
    return "Low"


def combine(components: Mapping[str, float]) -> dict[str, Any]:
    for key, value in components.items():
        if math.isnan(value) or not (0 <= value <= 100):
            raise ValueError(f"Component {key} out of 0-100: {value}")
    score = sum(FACTOR_WEIGHTS[k] * components[k] for k in FACTOR_WEIGHTS)
    return {
        "score": score,
        "band": band_for(score),
        "weighted_shortfall": {k: FACTOR_WEIGHTS[k] * (100.0 - components[k]) for k in FACTOR_WEIGHTS},
    }


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _named_fields(issues: list[dict[str, Any]], limit: int = 2) -> str:
    names = [f["label"] for f in issues[:limit]]
    extra = len(issues) - len(names)
    if extra > 0:
        return f"{', '.join(names)} and {extra} other input{'s' if extra > 1 else ''}"
    return _join(names)


def _data_limitation(quality: dict[str, Any], lever_features: set[str]) -> tuple[str, str, str | None] | None:
    """(reason code, sentence, action) for the most severe data issue."""
    issues = quality["issues"]
    if not issues:
        return None
    code = issues[0]["code"]
    same = [f for f in issues if f["code"] == code]
    first = same[0]
    if code == MISSING:
        verb = "is" if len(same) == 1 else "are"
        sentence = f"{_named_fields(same)} {verb} not available for this account."
        if first["feature"] in lever_features:
            action = f"If you know the account's {first['label']}, set it under Contracting Levers and run the simulation again."
        else:
            action = f"Ask the data team whether {first['label']} can be sourced for this account."
        return "SOURCE_MISSING", sentence, action
    if code == CARRY_FORWARD:
        verb = "uses" if len(same) == 1 else "use"
        sentence = f"{_named_fields(same)} {verb} last season's values, as this season's are not available yet."
        action = f"Confirm {first['label']} with the account team before a final pricing decision."
        return "SOURCE_CARRIED", sentence, action
    if code == IMPUTED:
        verb = "is" if len(same) == 1 else "are"
        sentence = f"{_named_fields(same)} {verb} estimated rather than recorded."
        return "SOURCE_IMPUTED", sentence, f"Validate {first['label']} before a final pricing decision."
    return None


# --- business summary (user's examples, 11/16) ---------------------------------
#
# One short paragraph in plain business language: what information supports
# the prediction (data, D), whether the account looks like historical
# customers (H), what limits it (data gaps, unfamiliar conditions, model
# versions that differ, a call close to the threshold), and a closing that
# matches the band — "strong support for the predicted win/loss" (High),
# "reasonable support ... some uncertainty remains" / "limiting the evidence"
# (Moderate), "reducing confidence in the prediction" (Low). Only facts the
# scores actually carry are named.

DATA_PHRASES = {
    "relationship": "relationship with customer",
    "loyalty": "customer loyalty",
    "price": "pricing",
    "clinical": "clinical inclination",
}
HISTORY_PHRASES = {
    "relationship": "relationship characteristics",
    "loyalty": "customer loyalty characteristics",
    "price": "pricing conditions",
    "clinical": "clinical characteristics",
}
PURCHASE_FEATURE = "log_prior_customer_doses"
STRONG_GROUP_QUALITY = 99.9   # every applicable input in the group recorded
TYPICAL_GROUP_SIMILARITY = 80.0
HIGH_SUMMARY_MARGIN = 75.0     # margin score at which a High summary names it (>= 15pp)
HIGH_SUMMARY_STABILITY = 80.0  # spread under 5pp across the model versions


def _verb(names: list[str], singular: str, plural: str) -> str:
    return singular if len(names) == 1 else plural


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def business_summary(
    band: str,
    outcome: str,
    gap_pp: float,
    quality: Mapping[str, Any],
    similarity: Mapping[str, Any],
    stable: Mapping[str, Any],
    components: Mapping[str, float],
) -> str:
    order = list(GROUPS)
    # (priority, text): the biggest weighted shortfall leads; a Win/Loss
    # disagreement always comes first (guide 9.3).
    negatives: list[tuple[float, str]] = []

    def shortfall(key: str) -> float:
        return FACTOR_WEIGHTS[key] * (100.0 - components[key])

    # --- data (D) ---------------------------------------------------------------
    issues = [f for f in quality["issues"] if f["code"] != MANUAL]
    strong = [g for g in order if quality["group_scores"].get(g, 0.0) >= STRONG_GROUP_QUALITY]
    strong_phrases = [DATA_PHRASES[g] for g in strong]
    purchases = next((f for f in quality["fields"] if f["feature"] == PURCHASE_FEATURE), None)
    if purchases and purchases["code"] in (OBSERVED, ENRICHED):
        strong_phrases.insert(0, "historical purchases")
    gaps = []
    for code, singular, plural in (
        (MISSING, "is not available", "are not available"),
        (IMPUTED, "is estimated", "are estimated"),
        (CARRY_FORWARD, "is based on older records", "are based on older records"),
    ):
        named = [f for f in issues if f["code"] == code]
        if named:
            many = len(named) > 3
            names = _named_fields(named, limit=2 if many else 3)
            gaps.append(f"{names} {plural if many or len(named) > 1 else singular}")
    if gaps:
        negatives.append((shortfall("D"), _join(gaps) + " for this account"))

    # --- history (H) ------------------------------------------------------------
    h_groups = similarity["group_scores"]
    unusual_groups = [g for g in sorted(h_groups, key=h_groups.get) if h_groups[g] < TYPICAL_GROUP_SIMILARITY]
    if any(f["feature"] in PRICE_LINKED for f in similarity["unusual"]) and "price" not in unusual_groups:
        unusual_groups.insert(0, "price")
    history_typical = not similarity["cap_applied"] and not unusual_groups
    if similarity["cap_applied"]:
        negatives.append((shortfall("H"), "only part of the account's information could be compared "
                          "with historical customers"))
    elif unusual_groups:
        negatives.append((shortfall("H"), f"the account's {_join([HISTORY_PHRASES[g] for g in unusual_groups[:2]])} "
                          "are less commonly observed among historical customers"))

    # --- model versions (S) and margin (M) -----------------------------------------
    if stable["disagreement"]:
        negatives.append((1000.0, "the model versions disagree on win versus loss"))
    elif components["S"] <= 50:
        negatives.append((shortfall("S"), "the model versions differ noticeably in their estimates"))
    if abs(gap_pp) <= NEAR_THRESHOLD_PP:
        negatives.append((shortfall("M"), "the win probability is close to the product's threshold"))
    if band == "Low" and not negatives:
        negatives.append((shortfall("M"), "the win probability is not far from the product's threshold"))
    # The two biggest limitations only (user, 11/16: two sentences in all).
    limits = [text for _, text in sorted(negatives, key=lambda n: n[0], reverse=True)][:2]
    limit_text = ", and ".join(limits)  # each may already contain an "and"

    # --- exactly two sentences ------------------------------------------------------
    # Opening by band (user, 11/16). Still true when an input is missing: only
    # areas whose every applicable input is recorded are listed by name.
    subject = _join(strong_phrases)
    if not strong_phrases:
        supported = "Some of the account's information is available"
    elif band == "High":
        supported = f"The prediction is supported by complete and reliable information on the account's {subject}"
    elif band == "Moderate":
        supported = f"Reliable information is available on the account's {subject}"
    else:
        supported = "Some of the account's information is available"

    if band == "High":
        # Supportive only (user, 11/16): a High summary names what backs the
        # prediction and no limitations; each phrase is still true for the row.
        first = supported
        if components["M"] >= HIGH_SUMMARY_MARGIN:
            side = "above" if gap_pp >= 0 else "below"
            first += f", and the win probability is well {side} the product's threshold"
        if history_typical:
            second = ("These characteristics are commonly observed among historical customers, "
                      f"providing strong support for the predicted {outcome}")
        elif components["S"] >= HIGH_SUMMARY_STABILITY and not stable["disagreement"]:
            second = f"The model versions give consistent estimates, providing strong support for the predicted {outcome}"
        else:
            second = f"Together, these provide strong support for the predicted {outcome}"
    elif band == "Moderate":
        first = supported
        if history_typical:
            first += ", and its characteristics are generally consistent with historical customer patterns"
        first += ", providing reasonable support for the prediction"
        if limits:
            second = f"However, {limit_text}, limiting the evidence supporting the prediction"
        else:
            second = "Some uncertainty remains, as the win probability is not far from the product's threshold"
    else:
        favour = "a favorable" if outcome == "win" else "an unfavorable"
        if len(limits) >= 2:
            first = f"Although the model indicates {favour} outcome, {limits[0]}"
            second = f"Additionally, {limits[1]}, reducing confidence in the prediction"
        else:
            first = supported
            second = f"However, {limits[0]}, reducing confidence in the prediction"
    return f"{first}. {second}."


def explain(
    product: str,
    win_probability: float,
    threshold: float,
    combined: dict[str, Any],
    components: Mapping[str, float],
    quality: dict[str, Any],
    similarity: dict[str, Any],
    stable: dict[str, Any],
    lever_features: Iterable[str] = (),
) -> dict[str, Any]:
    """Deterministic, rule-based confidence explanation (guide section 9):
    a headline, at most one strength, at most two limitations from different
    factors, at most one action, and one line per factor."""
    lever_features = set(lever_features)
    outcome = "win" if win_probability >= threshold else "loss"
    gap_pp = 100.0 * (win_probability - threshold)
    shortfall = combined["weighted_shortfall"]
    D, H, M, S = (components[k] for k in ("D", "H", "M", "S"))

    # --- limitations: (priority, factor, code, sentence, action) -------------
    candidates: list[tuple[float, str, str, str, str | None]] = []
    if stable["disagreement"]:
        candidates.append((100.0, "S", "STABILITY_DISAGREE",
                           "The five model versions disagree on win versus loss, so treat the call as uncertain.",
                           "Review this call with the account team before acting on it."))
    elif S <= 50:
        candidates.append((shortfall["S"], "S", "STABILITY_WEAK",
                           "The five model versions give noticeably different win probabilities.", None))

    if D < 85:
        limitation = _data_limitation(quality, lever_features)
        if limitation:
            code, sentence, action = limitation
            candidates.append((shortfall["D"] + 0.1, "D", code, sentence, action))

    unusual = similarity["unusual"]
    if unusual and unusual[0]["feature"] in PRICE_LINKED:
        candidates.append((shortfall["H"] + 0.1, "H", "HISTORY_PRICE_UNUSUAL",
                           "This price is outside the range usually seen in past contracts for this product.",
                           "Treat results at this price as directional."))
    elif similarity["cap_applied"]:
        candidates.append((shortfall["H"], "H", "HISTORY_COVERAGE_LIMITED",
                           "Only part of this account's information could be compared with past accounts, "
                           "because several inputs are missing or from last season.", None))
    elif unusual:
        candidates.append((shortfall["H"] + 0.1, "H", "HISTORY_FEATURE_UNUSUAL",
                           f"{unusual[0]['label']} is unusual compared with past accounts.",
                           f"Check that {unusual[0]['label']} is correct for this account."))
    elif H < 60:
        candidates.append((shortfall["H"], "H", "HISTORY_LIMITED",
                           "This account is less like the past accounts the model learned from.", None))
    elif H < 80:
        candidates.append((shortfall["H"], "H", "HISTORY_MODERATE",
                           "Some of this account's conditions are less common among past accounts.", None))

    if abs(gap_pp) <= NEAR_THRESHOLD_PP:
        candidates.append((shortfall["M"], "M", "MARGIN_NEAR",
                           f"The win probability is within {NEAR_THRESHOLD_PP:.0f} points of the threshold, "
                           "so a small change could flip the call.", None))

    chosen, used = [], set()
    for item in sorted(candidates, key=lambda c: c[0], reverse=True):
        if item[1] not in used:
            chosen.append(item)
            used.add(item[1])
        if len(chosen) == 2:
            break

    # --- strength: the best-supported factor that clears its bar ----------------
    strengths = []
    if M >= 75:
        strengths.append((FACTOR_WEIGHTS["M"] * M, "MARGIN_FAR", "The win probability is well clear of the threshold."))
    if S >= 80 and not stable["disagreement"]:
        strengths.append((FACTOR_WEIGHTS["S"] * S, "STABILITY_STRONG", "All five model versions give the same answer."))
    if D >= 85:
        # Don't call the inputs complete while any is missing or estimated.
        text = ("The account's key inputs are complete and recorded." if not quality["issues"]
                else "Most of the account's key inputs are available and recorded.")
        strengths.append((FACTOR_WEIGHTS["D"] * D, "DATA_STRONG", text))
    if H >= 80 and not similarity["cap_applied"]:
        strengths.append((FACTOR_WEIGHTS["H"] * H, "HISTORY_STRONG",
                          "This account and price look like the accounts the model learned from."))
    strength = max(strengths, key=lambda s: s[0]) if strengths else None

    headline = f"{combined['band']} confidence in a predicted {outcome}"
    # The reason codes above stay for audit; the text shown is the business
    # summary in the style the user set on 11/16 (see business_summary()).
    summary = business_summary(combined["band"], outcome, gap_pp, quality, similarity, stable, components)
    action = next((item[4] for item in chosen if item[4]), None)

    # --- one line per factor -----------------------------------------------------
    counts = quality["counts"]
    parts = []
    applicable = sum(n for code, n in counts.items() if code not in EXCLUDED_FROM_D)
    recorded = counts.get(OBSERVED, 0) + counts.get(ENRICHED, 0)
    # "recorded", not "this season's": a 2027-28 prediction is made from the
    # 2026-27 values by design (11/16), so they are the expected inputs.
    parts.append(f"{recorded} of {applicable} inputs are recorded values")
    if counts.get(MANUAL):
        parts.append(f"{counts[MANUAL]} set in the levers")
    if counts.get(CARRY_FORWARD):
        n = counts[CARRY_FORWARD]
        parts.append(f"{n} {'uses' if n == 1 else 'use'} last season's value{'' if n == 1 else 's'}")
    if counts.get(IMPUTED):
        n = counts[IMPUTED]
        parts.append(f"{n} {'is' if n == 1 else 'are'} estimated")
    if counts.get(MISSING):
        n = counts[MISSING]
        parts.append(f"{n} {'is' if n == 1 else 'are'} missing")
    data_line = "; ".join(parts) + "."

    coverage_pct = 100.0 * similarity["coverage"]
    history_line = f"{coverage_pct:.0f}% of inputs could be compared with past accounts"
    if similarity["group_scores"]:
        weakest = min(similarity["group_scores"], key=similarity["group_scores"].get)
        if similarity["group_scores"][weakest] < 80:
            history_line += f"; least typical: {GROUPS[weakest]['label']}"
    if similarity["cap_applied"]:
        cap = CAP_LOW if similarity["coverage"] < COVERAGE_PARTIAL else CAP_PARTIAL
        history_line += f" (score capped at {cap:.0f} because under {COVERAGE_FULL:.0%} could be compared)"
    history_line += "."

    side = "above" if gap_pp >= 0 else "below"
    margin_line = f"Win probability is {abs(gap_pp):.0f} points {side} the {product} threshold of {threshold:.0%}."

    spread = f"The {stable['n_models']} model versions are up to {stable['spread_pp']:.1f} points apart"
    if stable["disagreement"]:
        stability_line = f"{spread} and disagree on win versus loss."
    else:
        stability_line = f"{spread} and all predict a {outcome}."

    factors = [
        {"key": "D", "name": FACTOR_NAMES["D"], "score": D, "weight": FACTOR_WEIGHTS["D"], "detail": data_line},
        {"key": "H", "name": FACTOR_NAMES["H"], "score": H, "weight": FACTOR_WEIGHTS["H"], "detail": history_line},
        {"key": "M", "name": FACTOR_NAMES["M"], "score": M, "weight": FACTOR_WEIGHTS["M"], "detail": margin_line},
        {"key": "S", "name": FACTOR_NAMES["S"], "score": S, "weight": FACTOR_WEIGHTS["S"], "detail": stability_line},
    ]

    return {
        "headline": headline,
        "summary": summary,
        "action": action,
        "factors": factors,
        "caveat": CAVEAT,
        "reason_codes": ([strength[1]] if strength else []) + [item[2] for item in chosen],
    }


def assess(
    product: str,
    features: Mapping[str, Any],
    provenance: Mapping[str, str],
    win_probability: float,
    threshold: float,
    fold_probabilities: Iterable[float],
    reference: Mapping[str, Any],
    lever_features: Iterable[str] = (),
) -> dict[str, Any]:
    """The full confidence result for one account x product x scenario."""
    unknown = {code for code in provenance.values() if code not in KNOWN_CODES}
    if unknown:
        raise ValueError(f"Unknown provenance codes: {sorted(unknown)}")
    quality = data_quality(provenance)
    similarity = historical_similarity(features, provenance, reference)
    stable = stability(fold_probabilities, threshold)
    return _finish(product, win_probability, threshold, quality, similarity, stable, lever_features)


def _finish(product, win_probability, threshold, quality, similarity, stable, lever_features) -> dict[str, Any]:
    components = {
        "D": quality["score"],
        "H": similarity["score"],
        "M": decision_margin(win_probability, threshold),
        "S": stable["score"],
    }
    combined = combine(components)
    return {
        "method_version": METHOD_VERSION,
        "product": product,
        "score": combined["score"],
        "band": combined["band"],
        "components": components,
        "weighted_shortfall": combined["weighted_shortfall"],
        "win_probability": float(win_probability),
        "threshold": float(threshold),
        "predicted_outcome": "Win" if win_probability >= threshold else "Loss",
        "gap_pp": 100.0 * (win_probability - threshold),
        "data_quality": quality,
        "historical_similarity": similarity,
        "stability": stable,
        "lever_features": sorted(set(lever_features)),
        "explanation": explain(
            product, win_probability, threshold, combined, components, quality, similarity, stable, lever_features
        ),
    }


def at_probability(detail: Mapping[str, Any], win_probability: float) -> dict[str, Any]:
    """The same assessment re-centred on a different win probability (the
    app's baseline view shows the notebook's own prob0 rather than a fresh
    model call). D, H and S do not depend on it; M, the total and the text do."""
    return _finish(
        detail["product"],
        win_probability,
        detail["threshold"],
        detail["data_quality"],
        detail["historical_similarity"],
        detail["stability"],
        detail["lever_features"],
    )
