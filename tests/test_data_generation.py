from __future__ import annotations

from src.generate_dummy_data import generate_dummy_data
from src.utils import OUTCOME_LABELS


REQUIRED_COLUMNS = {
    "deal_id",
    "contract_id",
    "customer_id",
    "product_id",
    "brand_name",
    "season_id",
    "round_number",
    "customer_segment",
    "customer_name",
    "contract_name",
    "prior_volume",
    "prior_net_price",
    "net_price",
    "rebate_pct",
    "reimbursement_rate_pct",
    "volume_commitment",
    "market_share_commitment_pct",
    "returnability_pct",
    "outcome",
}


def test_generate_dummy_data_default_rows():
    data = generate_dummy_data()
    assert data.shape[0] == 500


def test_required_columns_present(dummy_data):
    assert REQUIRED_COLUMNS.issubset(set(dummy_data.columns))


def test_outcome_values_are_expected(dummy_data):
    assert set(dummy_data["outcome"].unique()).issubset(set(OUTCOME_LABELS))


def test_critical_columns_not_entirely_null(dummy_data):
    critical_columns = ["deal_id", "customer_name", "brand_name", "net_price", "rebate_pct", "outcome"]
    assert not dummy_data[critical_columns].isnull().all().any()


def test_numeric_ranges(dummy_data):
    assert dummy_data["net_price"].between(0, 300).all()
    assert dummy_data["rebate_pct"].between(0, 50).all()
    assert dummy_data["reimbursement_rate_pct"].between(0, 150).all()
    assert dummy_data["volume_commitment"].between(0, 2_000_000).all()
    assert dummy_data["market_share_commitment_pct"].between(0, 100).all()
    assert dummy_data["returnability_pct"].between(0, 30).all()


def test_outcome_distribution_not_extremely_imbalanced(dummy_data):
    distribution = dummy_data["outcome"].value_counts(normalize=True)
    assert distribution.min() > 0.10
    assert distribution.max() < 0.70


def test_deal_id_is_unique(dummy_data):
    assert dummy_data["deal_id"].is_unique


def test_customer_segments_limited_to_retail_and_health_system(dummy_data):
    assert set(dummy_data["customer_segment"].unique()) == {"Retail", "Health System"}
    assert dummy_data["customer_name"].nunique() >= 24
