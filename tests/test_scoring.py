from __future__ import annotations

from src.scoring import (
    build_input_dataframe,
    calculate_driver_influence,
    generate_narrative_summary,
    load_feature_metadata,
    load_model,
    predict_customer_response,
)


def test_build_input_dataframe_returns_single_row(sample_input):
    feature_metadata = load_feature_metadata()
    input_df = build_input_dataframe(sample_input, feature_metadata)
    assert input_df.shape[0] == 1
    assert set(feature_metadata["feature_columns"]).issubset(input_df.columns)


def test_predict_customer_response_shape(sample_input):
    model = load_model()
    feature_metadata = load_feature_metadata()
    input_df = build_input_dataframe(sample_input, feature_metadata)
    prediction = predict_customer_response(model, input_df)

    assert set(prediction["probabilities"].keys()) == {"Accepted", "Negotiated", "Rejected"}
    assert 0 <= prediction["confidence_score"] <= 100
    assert prediction["confidence_band"] in {"High", "Moderate", "Low"}


def test_driver_influence_categories(sample_input):
    drivers = calculate_driver_influence(sample_input)
    assert {"commercial_offer", "customer_relationship", "competitive_pressure", "deal_context", "watchouts"}.issubset(drivers)


def test_generate_narrative_summary_returns_text(sample_input):
    model = load_model()
    feature_metadata = load_feature_metadata()
    input_df = build_input_dataframe(sample_input, feature_metadata)
    prediction = predict_customer_response(model, input_df)
    drivers = calculate_driver_influence(sample_input)
    narrative = generate_narrative_summary(prediction, drivers, sample_input)
    assert isinstance(narrative, str)
    assert len(narrative) > 50
