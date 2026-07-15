from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from src.utils import (
        FEATURE_COLUMNS_PATH,
        ID_TO_OUTCOME,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        OUTCOME_LABELS,
        calculate_derived_fields,
        read_json,
    )
except ModuleNotFoundError:
    from utils import (
        FEATURE_COLUMNS_PATH,
        ID_TO_OUTCOME,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        OUTCOME_LABELS,
        calculate_derived_fields,
        read_json,
    )


DRIVER_CATEGORIES = {
    "commercial_offer": "Offer Dynamics",
    "customer_relationship": "Customer Context",
    "competitive_pressure": "Competitive Pressure",
    "deal_context": "Negotiation Context",
}


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained sklearn pipeline."""
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


def build_input_dataframe(user_inputs: dict[str, Any], feature_metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """Convert app inputs into the exact feature DataFrame expected by the model."""
    feature_metadata = feature_metadata or load_feature_metadata()
    feature_columns = feature_metadata["feature_columns"]
    row = dict(user_inputs)
    row.update(calculate_derived_fields(row))

    missing_columns = [column for column in feature_columns if column not in row]
    if missing_columns:
        raise ValueError(f"Missing model input columns: {', '.join(missing_columns)}")

    return pd.DataFrame([{column: row[column] for column in feature_columns}])


def predict_customer_response(model, input_df: pd.DataFrame) -> dict[str, Any]:
    probabilities_array = model.predict_proba(input_df)[0]
    probabilities = {label: float(probabilities_array[index]) for index, label in enumerate(OUTCOME_LABELS)}

    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    prediction_id = int(probabilities_array.argmax())
    predicted_outcome = ID_TO_OUTCOME[prediction_id]
    base_confidence = probabilities[predicted_outcome] * 100
    top_two_gap = (sorted_probabilities[0] - sorted_probabilities[1]) * 100 if len(sorted_probabilities) > 1 else 100
    confidence_score = max(base_confidence - 10, 0) if top_two_gap < 10 else base_confidence
    confidence_score = round(min(confidence_score, 100), 1)

    if confidence_score >= 75:
        confidence_band = "High"
    elif confidence_score >= 55:
        confidence_band = "Moderate"
    else:
        confidence_band = "Low"

    return {
        "probabilities": probabilities,
        "predicted_outcome": predicted_outcome,
        "recommended_outcome": predicted_outcome,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "top_two_gap_pp": round(top_two_gap, 1),
    }


def _driver(name: str, direction: str, impact_pp: int, explanation: str, category: str) -> dict[str, Any]:
    return {
        "name": name,
        "direction": direction,
        "impact_pp": impact_pp,
        "explanation": explanation,
        "category": category,
    }


def calculate_driver_influence(user_inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    derived = calculate_derived_fields(user_inputs)
    inputs = {**user_inputs, **derived}
    drivers: dict[str, list[dict[str, Any]]] = {
        "commercial_offer": [],
        "customer_relationship": [],
        "competitive_pressure": [],
        "deal_context": [],
        "watchouts": [],
    }

    if inputs["rebate_pct"] >= inputs["prior_rebate_pct"] + 3:
        drivers["commercial_offer"].append(
            _driver("Rebate support", "Positive", 5, "Proposed rebate is above the prior contract level.", "commercial_offer")
        )
    elif inputs["rebate_pct"] < inputs["prior_rebate_pct"]:
        drivers["commercial_offer"].append(
            _driver("Rebate support", "Negative", -4, "Proposed rebate is below the prior contract level.", "commercial_offer")
        )

    if inputs["price_gap_vs_prior"] > 25:
        drivers["commercial_offer"].append(
            _driver("Net price movement", "Negative", -6, "Net price is materially above prior year.", "commercial_offer")
        )
    elif inputs["price_gap_vs_prior"] > 4:
        drivers["commercial_offer"].append(
            _driver("Net price movement", "Negative", -3, "Net price is above prior year.", "commercial_offer")
        )
    elif inputs["price_gap_vs_prior"] < -3:
        drivers["commercial_offer"].append(
            _driver("Net price movement", "Positive", 4, "Net price is more favorable than prior year.", "commercial_offer")
        )

    if inputs["reimbursement_rate_pct"] >= 86:
        drivers["commercial_offer"].append(
            _driver("Reimbursement support", "Positive", 5, "Reimbursement rate is favorable for access and pull-through.", "commercial_offer")
        )
    elif inputs["reimbursement_rate_pct"] < 60:
        drivers["commercial_offer"].append(
            _driver("Reimbursement support", "Negative", -7, "Reimbursement rate falls materially below the level typically required for pull-through.", "commercial_offer")
        )
    elif inputs["reimbursement_rate_pct"] < 76:
        drivers["commercial_offer"].append(
            _driver("Reimbursement support", "Negative", -4, "Reimbursement rate is below the level this customer has historically required.", "commercial_offer")
        )

    if inputs["volume_gap_vs_prior"] >= 5:
        drivers["commercial_offer"].append(
            _driver("Volume commitment", "Positive", 3, "Volume commitment is higher than prior volume.", "commercial_offer")
        )

    if inputs["market_share_gap_vs_prior"] >= 4:
        drivers["commercial_offer"].append(
            _driver("Market share commitment", "Positive", 3, "Market share commitment improves versus prior year.", "commercial_offer")
        )
    if inputs["market_share_commitment_pct"] > 80:
        drivers["commercial_offer"].append(
            _driver("Aggressive share ask", "Negative", -4, "Market share commitment is above 80%.", "commercial_offer")
        )

    if inputs["historical_win_rate"] >= 0.70:
        drivers["customer_relationship"].append(
            _driver("Historical relationship", "Positive", 5, "Customer has a strong historical win rate.", "customer_relationship")
        )
    elif inputs["historical_win_rate"] < 0.45:
        drivers["customer_relationship"].append(
            _driver("Historical relationship", "Negative", -5, "Customer has a weaker historical acceptance pattern.", "customer_relationship")
        )

    if inputs["prior_year_attainment_pct"] >= 95:
        drivers["customer_relationship"].append(
            _driver("Prior attainment", "Positive", 4, "Customer has strong prior attainment against contracted goals.", "customer_relationship")
        )
    elif inputs["prior_year_attainment_pct"] < 75:
        drivers["customer_relationship"].append(
            _driver("Prior attainment", "Negative", -4, "Weak prior attainment lowers confidence in proposed commitments.", "customer_relationship")
        )

    if inputs["prior_acceptance_flag"] == 1:
        drivers["customer_relationship"].append(
            _driver("Prior acceptance", "Positive", 2, "Customer previously accepted a contracting offer.", "customer_relationship")
        )

    if inputs["competitive_pressure_score"] >= 70:
        drivers["competitive_pressure"].append(
            _driver("Competitive pressure", "Negative", -6, "Competitive pressure is high in this account.", "competitive_pressure")
        )
    elif inputs["competitive_pressure_score"] <= 35:
        drivers["competitive_pressure"].append(
            _driver("Competitive position", "Positive", 3, "Competitive pressure is relatively low.", "competitive_pressure")
        )
    else:
        drivers["competitive_pressure"].append(
            _driver("Competitive position", "Neutral", 0, "Competitive pressure is moderate.", "competitive_pressure")
        )

    if inputs["competitor_rebate_estimate_pct"] > inputs["rebate_pct"] + 3:
        drivers["competitive_pressure"].append(
            _driver("Competitor rebate", "Negative", -3, "The estimated competitor rebate exceeds the level offered in this proposal.", "competitive_pressure")
        )

    if inputs["returnability_pct"] < inputs["prior_returnability_pct"] - 2:
        drivers["deal_context"].append(
            _driver("Returnability", "Negative", -3, "Returnability is below customer expectation based on prior terms.", "deal_context")
        )
    if inputs["cancellation_pct"] < inputs["prior_cancellation_pct"] - 2:
        drivers["deal_context"].append(
            _driver("Cancellation allowance", "Negative", -3, "Cancellation allowance is lower than prior contract.", "deal_context")
        )
    if inputs["deal_attractiveness_score"] >= 68:
        drivers["deal_context"].append(
            _driver("Deal attractiveness", "Positive", 5, "Overall economics compare favorably to this customer's historical acceptance range.", "deal_context")
        )
    elif inputs["deal_attractiveness_score"] < 45:
        drivers["deal_context"].append(
            _driver("Deal attractiveness", "Negative", -5, "Overall economics fall below the range this customer has historically accepted.", "deal_context")
        )

    if inputs["reimbursement_rate_pct"] < 60:
        drivers["watchouts"].append(
            _driver("Low reimbursement", "Watchout", 0, "Reimbursement below 60% may create customer pushback.", "commercial_offer")
        )
    if inputs["volume_gap_vs_prior"] > 50:
        drivers["watchouts"].append(
            _driver("Volume validation", "Watchout", 0, "Volume commitment exceeds 150% of prior volume.", "commercial_offer")
        )
    if inputs["price_gap_vs_prior"] > 25:
        drivers["watchouts"].append(
            _driver("Price increase", "Watchout", 0, "Net price is more than 25% above prior net price.", "commercial_offer")
        )
    if inputs["rebate_pct"] < inputs["prior_rebate_pct"] and inputs["price_gap_vs_prior"] > 0:
        drivers["watchouts"].append(
            _driver("Price-rebate mismatch", "Watchout", 0, "Rebate is lower while price is higher.", "commercial_offer")
        )
    if inputs["market_share_commitment_pct"] > 80:
        drivers["watchouts"].append(
            _driver("Aggressive share target", "Watchout", 0, "Market share commitment is above 80%.", "commercial_offer")
        )

    defaults = {
        "commercial_offer": _driver("Offer balance", "Neutral", 0, "Commercial terms are broadly in line with the baseline.", "commercial_offer"),
        "customer_relationship": _driver("Relationship baseline", "Neutral", 0, "No dominant relationship signal was detected.", "customer_relationship"),
        "competitive_pressure": _driver("Competitive position", "Neutral", 0, "Competitive pressure is within the expected range.", "competitive_pressure"),
        "deal_context": _driver("Deal context", "Neutral", 0, "Deal context does not show a major directional signal.", "deal_context"),
    }
    for key, default in defaults.items():
        if not drivers[key]:
            drivers[key].append(default)
    if not drivers["watchouts"]:
        drivers["watchouts"].append(
            _driver("Monitor assumptions", "Watchout", 0, "Validate assumptions before using this scenario outside the demo.", "deal_context")
        )

    return drivers


def _top_driver(drivers: dict[str, list[dict[str, Any]]], direction: str) -> dict[str, Any]:
    flattened = [
        driver
        for key in ["commercial_offer", "customer_relationship", "competitive_pressure", "deal_context"]
        for driver in drivers.get(key, [])
        if driver["direction"] == direction
    ]
    if not flattened:
        return _driver("balanced economics", "Neutral", 0, "No single dominant driver was detected.", "deal_context")
    return sorted(flattened, key=lambda item: abs(item["impact_pp"]), reverse=True)[0]


def generate_narrative_summary(
    prediction: dict[str, Any],
    drivers: dict[str, list[dict[str, Any]]],
    user_inputs: dict[str, Any],
) -> str:
    positive = _top_driver(drivers, "Positive")
    negative = _top_driver(drivers, "Negative")
    watchout = drivers.get("watchouts", [])[0]

    lever = "rebate support"
    if user_inputs["reimbursement_rate_pct"] < 82:
        lever = "reimbursement support"
    elif user_inputs["competitive_pressure_score"] >= 70:
        lever = "competitive positioning"
    elif calculate_derived_fields(user_inputs)["price_gap_vs_prior"] > 4:
        lever = "net price movement"

    return (
        f"The scenario is currently recommended as {prediction['recommended_outcome']} with a "
        f"{prediction['confidence_band'].lower()} confidence score of {prediction['confidence_score']:.1f}%. "
        f"The strongest supportive signal is {positive['name'].lower()}, while {negative['name'].lower()} is the main "
        f"pressure point to manage. The team should watch {watchout['name'].lower()} before advancing the offer. "
        f"The most practical lever to improve the outcome is {lever}."
    )


def score_customer_response(model, user_inputs: dict[str, Any], feature_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    input_df = build_input_dataframe(user_inputs, feature_metadata)
    prediction = predict_customer_response(model, input_df)
    drivers = calculate_driver_influence(user_inputs)
    narrative = generate_narrative_summary(prediction, drivers, user_inputs)
    return {**prediction, "drivers": drivers, "watchouts": drivers["watchouts"], "narrative_summary": narrative}


def flatten_driver_cards(drivers: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for key in ["commercial_offer", "customer_relationship", "competitive_pressure", "deal_context"]:
        cards.extend(drivers.get(key, []))
    return cards
