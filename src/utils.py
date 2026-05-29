from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
DATA_PATH = DATA_DIR / "dummy_contracting_data.csv"
MODEL_PATH = MODEL_DIR / "xgb_customer_response_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

OUTCOME_LABELS = ["Accepted", "Negotiated", "Rejected"]
OUTCOME_TO_ID = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}
ID_TO_OUTCOME = {idx: label for label, idx in OUTCOME_TO_ID.items()}

IDENTIFIER_COLUMNS = [
    "deal_id",
    "contract_id",
    "customer_id",
    "product_id",
    "l1_parent",
]
TARGET_COLUMN = "outcome"

DISPLAY_COLUMNS = [
    "customer_segment",
    "customer_name",
    "contract_name",
    "product_scope",
    "brand_scope",
]

MODEL_EXCLUDED_COLUMNS = IDENTIFIER_COLUMNS + [TARGET_COLUMN]


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


def load_contracting_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def today_iso() -> str:
    return date.today().isoformat()


def calculate_derived_fields(inputs: dict[str, Any]) -> dict[str, Any]:
    net_price = float(inputs["net_price"])
    rebate_pct = float(inputs["rebate_pct"])
    volume_commitment = int(inputs["volume_commitment"])
    admin_fee_pct = float(inputs.get("admin_fee_pct", 2.0))
    prior_volume = max(float(inputs["prior_volume"]), 1.0)
    prior_market_share = float(inputs["prior_year_market_share"])
    prior_net_price = max(float(inputs["prior_net_price"]), 0.01)
    reimbursement_rate_pct = float(inputs["reimbursement_rate_pct"])

    safe_net_price = max(net_price, 0.01)
    gross_revenue = net_price * volume_commitment
    estimated_rebate_dollars = gross_revenue * rebate_pct / 100
    estimated_net_revenue = gross_revenue - estimated_rebate_dollars
    margin_proxy = max(min((safe_net_price - 45) / safe_net_price - rebate_pct / 100 - admin_fee_pct / 100, 0.62), 0.08)
    volume_gap_vs_prior = (volume_commitment - prior_volume) / prior_volume * 100
    market_share_gap_vs_prior = float(inputs["market_share_commitment_pct"]) - prior_market_share
    price_gap_vs_prior = (net_price - prior_net_price) / prior_net_price * 100
    reimbursement_gap = reimbursement_rate_pct - 80

    deal_attractiveness_score = (
        52
        + (rebate_pct - float(inputs["prior_rebate_pct"])) * 1.15
        - price_gap_vs_prior * 0.55
        + reimbursement_gap * 0.55
        + min(volume_gap_vs_prior, 35) * 0.20
        + (float(inputs["prior_year_attainment_pct"]) - 85) * 0.18
        + float(inputs["historical_win_rate"]) * 18
        - float(inputs["competitive_pressure_score"]) * 0.16
        - max(market_share_gap_vs_prior - 12, 0) * 0.50
        + int(inputs["strategic_account_flag"]) * 3.0
        + int(inputs["price_protection_flag"]) * 2.2
    )

    return {
        "gross_revenue": round(gross_revenue, 2),
        "estimated_rebate_dollars": round(estimated_rebate_dollars, 2),
        "estimated_net_revenue": round(estimated_net_revenue, 2),
        "margin_proxy": round(margin_proxy, 3),
        "volume_gap_vs_prior": round(volume_gap_vs_prior, 2),
        "market_share_gap_vs_prior": round(market_share_gap_vs_prior, 2),
        "price_gap_vs_prior": round(price_gap_vs_prior, 2),
        "reimbursement_gap": round(reimbursement_gap, 2),
        "deal_attractiveness_score": round(max(min(deal_attractiveness_score, 100), 0), 1),
    }
