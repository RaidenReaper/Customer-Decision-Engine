from __future__ import annotations

from typing import Any

import pytest
from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    app = AppTest.from_file("app.py")
    app.run(timeout=45)
    _assert_clean(app)
    return app


def _assert_clean(app: AppTest) -> None:
    assert len(app.exception) == 0


def _by_label(elements: Any, label: str):
    for element in elements:
        if element.label == label:
            return element
    raise AssertionError(f"Could not find widget labeled {label!r}")


def _run_simulation(app: AppTest) -> dict:
    _by_label(app.button, "Run Simulation").click().run(timeout=45)
    _assert_clean(app)
    result = app.session_state["latest_result"]
    assert result is not None
    assert set(result["probabilities"]) == {"Accepted", "Negotiated", "Rejected"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.001
    assert 0 <= result["confidence_score"] <= 100
    return result


def _set_number_inputs(app: AppTest, values: dict[str, float | int]) -> AppTest:
    for label, value in values.items():
        _by_label(app.number_input, label).set_value(value)
    app.run(timeout=45)
    _assert_clean(app)
    return app


def test_save_before_run_does_not_crash_or_append():
    app = _app()
    assert app.session_state["latest_result"] is None
    assert app.session_state["saved_scenarios"] == []

    _by_label(app.button, "Save Scenario").click().run(timeout=45)

    _assert_clean(app)
    assert app.session_state["latest_result"] is None
    assert app.session_state["saved_scenarios"] == []


def test_all_context_filters_and_single_product_runs_for_every_segment():
    app = _app()
    segment_filter = _by_label(app.selectbox, "Customer Segment")

    for segment in segment_filter.options:
        _by_label(app.selectbox, "Customer Segment").set_value(segment).run(timeout=45)
        _assert_clean(app)

        customer_filter = _by_label(app.selectbox, "Customer")
        customer_filter.set_value(customer_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        contract_filter = _by_label(app.selectbox, "Contract Name")
        contract_filter.set_value(contract_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        season_filter = _by_label(app.selectbox, "Season")
        season_filter.set_value(season_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        round_filter = _by_label(app.selectbox, "Round Number")
        round_filter.set_value(round_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        _by_label(app.selectbox, "Product Scope").set_value("Single Product").run(timeout=45)
        _assert_clean(app)
        brand_filter = _by_label(app.selectbox, "Brand / Brands")
        brand_filter.set_value(brand_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        result = _run_simulation(app)
        assert result["recommended_outcome"] in {"Accepted", "Negotiated", "Rejected"}


def test_multi_product_filter_and_save_scenario_table():
    app = _app()
    _by_label(app.selectbox, "Product Scope").set_value("Multi Product").run(timeout=45)
    _assert_clean(app)

    assert len(app.multiselect) == 1
    brand_filter = _by_label(app.multiselect, "Brand / Brands")
    selected_brands = brand_filter.options[: min(2, len(brand_filter.options))]
    brand_filter.set_value(selected_brands).run(timeout=45)
    _assert_clean(app)
    assert len(app.number_input) == 8 * len(selected_brands)

    result = _run_simulation(app)
    assert len(result["product_results"]) == len(selected_brands)
    _by_label(app.button, "Save Scenario").click().run(timeout=45)

    _assert_clean(app)
    saved = app.session_state["saved_scenarios"]
    assert len(saved) == 1
    assert saved[0]["Recommended outcome"] in {"Accepted", "Negotiated", "Rejected"}
    assert {"Accepted probability", "Negotiated probability", "Rejected probability"}.issubset(saved[0])


def test_filter_switching_scope_toggle_and_empty_multi_select_stay_stable():
    app = _app()

    segment_filter = _by_label(app.selectbox, "Customer Segment")
    for segment in segment_filter.options:
        _by_label(app.selectbox, "Customer Segment").set_value(segment).run(timeout=45)
        _assert_clean(app)

        customer_filter = _by_label(app.selectbox, "Customer")
        customer_filter.set_value(customer_filter.options[-1]).run(timeout=45)
        _assert_clean(app)

        _by_label(app.selectbox, "Product Scope").set_value("Multi Product").run(timeout=45)
        _assert_clean(app)

        brand_filter = _by_label(app.multiselect, "Brand / Brands")
        brand_filter.set_value([]).run(timeout=45)
        _assert_clean(app)
        assert len(app.number_input) >= 8

        result = _run_simulation(app)
        assert result["recommended_outcome"] in {"Accepted", "Negotiated", "Rejected"}

        _by_label(app.button, "Reset Inputs").click().run(timeout=45)
        _assert_clean(app)
        assert app.session_state["latest_result"] is None

        _by_label(app.selectbox, "Product Scope").set_value("Single Product").run(timeout=45)
        _assert_clean(app)
        brand_select = _by_label(app.selectbox, "Brand / Brands")
        brand_select.set_value(brand_select.options[0]).run(timeout=45)
        _assert_clean(app)

        result = _run_simulation(app)
        assert result["recommended_outcome"] in {"Accepted", "Negotiated", "Rejected"}


@pytest.mark.parametrize(
    "values",
    [
        {
            "Proposed Net Price": 65.0,
            "Proposed Rebate %": 32.0,
            "Proposed Multi-year Discount %": 8.0,
            "Proposed Cancellation %": 10.0,
            "Proposed Reimbursement Rate %": 105.0,
            "Proposed Volume Commitment": 80_000,
            "Proposed Market Share Commitment %": 55.0,
            "Proposed Returnability %": 14.0,
        },
        {
            "Proposed Net Price": 260.0,
            "Proposed Rebate %": 1.0,
            "Proposed Multi-year Discount %": 0.0,
            "Proposed Cancellation %": 0.0,
            "Proposed Reimbursement Rate %": 45.0,
            "Proposed Volume Commitment": 1_900_000,
            "Proposed Market Share Commitment %": 95.0,
            "Proposed Returnability %": 0.0,
        },
        {
            "Proposed Net Price": 118.0,
            "Proposed Rebate %": 16.0,
            "Proposed Multi-year Discount %": 2.5,
            "Proposed Cancellation %": 5.0,
            "Proposed Reimbursement Rate %": 84.0,
            "Proposed Volume Commitment": 45_000,
            "Proposed Market Share Commitment %": 42.0,
            "Proposed Returnability %": 8.0,
        },
    ],
)
def test_run_simulation_with_different_input_combinations(values):
    app = _app()
    _set_number_inputs(app, values)

    result = _run_simulation(app)

    assert result["narrative_summary"]
    assert {"commercial_offer", "customer_relationship", "competitive_pressure", "deal_context", "watchouts"}.issubset(
        result["drivers"]
    )


def test_reset_inputs_restores_default_values():
    app = _app()
    original_net_price = _by_label(app.number_input, "Proposed Net Price").value
    original_rebate = _by_label(app.number_input, "Proposed Rebate %").value

    _set_number_inputs(app, {"Proposed Net Price": 250.0, "Proposed Rebate %": 1.0})
    assert _by_label(app.number_input, "Proposed Net Price").value == 250.0
    assert _by_label(app.number_input, "Proposed Rebate %").value == 1.0

    _by_label(app.button, "Reset Inputs").click().run(timeout=45)

    _assert_clean(app)
    assert _by_label(app.number_input, "Proposed Net Price").value == original_net_price
    assert _by_label(app.number_input, "Proposed Rebate %").value == original_rebate
    assert app.session_state["latest_result"] is None
