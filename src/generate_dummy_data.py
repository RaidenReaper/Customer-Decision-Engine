from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils import DATA_PATH, OUTCOME_LABELS, PROJECT_ROOT, ensure_directories
except ModuleNotFoundError:
    from utils import DATA_PATH, OUTCOME_LABELS, PROJECT_ROOT, ensure_directories


def _choice(rng: np.random.Generator, values: list[str], size: int, probs: list[float] | None = None) -> np.ndarray:
    return rng.choice(values, size=size, p=probs)


def generate_dummy_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic vaccine contracting scenarios with business-shaped outcomes."""
    rng = np.random.default_rng(seed)

    segments = ["Retail", "Health System"]
    customer_segment = _choice(rng, segments, rows, [0.48, 0.52])
    customer_class_map = {
        "Retail": "Pharmacy Chain",
        "Health System": "Integrated Delivery Network",
    }
    customer_class = np.array([customer_class_map[item] for item in customer_segment])

    customer_names = {
        "Retail": [
            "WellWay Pharmacy",
            "CarePlus Retail",
            "CommunityRx",
            "HealthMart Alliance",
            "PrimeDose Pharmacy",
            "CityScript Retail",
            "Neighborhood Care Pharmacy",
            "VitaPoint Pharmacy",
            "FamilyFirst Rx",
            "MetroMeds Pharmacy",
            "OakStreet Pharmacy",
            "Pioneer Retail Health",
            "Sunrise Pharmacy Group",
            "EverCare Retail",
            "BlueLeaf Pharmacy",
            "Frontier Rx Network",
        ],
        "Health System": [
            "NorthStar Health",
            "Evergreen Integrated Care",
            "Summit Regional Health",
            "Cedar Valley Health System",
            "MercyBridge Health",
            "Lakeside Medical Network",
            "Pinecrest Health Partners",
            "Riverside Integrated Care",
            "MountainView Health",
            "HarborPoint Health System",
            "PrairieCare Health",
            "Silverline Medical Center",
            "Unity Regional Health",
            "Stonegate Health Network",
            "Horizon Valley Health",
            "Redwood Health Alliance",
        ],
    }
    customer_name = np.array([rng.choice(customer_names[segment]) for segment in customer_segment])

    brands = ["FluShield QIV", "FluShield HD", "VaxiGuard RSV", "PneuProtect", "TravelVax"]
    product_families = {
        "FluShield QIV": "Influenza",
        "FluShield HD": "Influenza",
        "VaxiGuard RSV": "Respiratory",
        "PneuProtect": "Pneumococcal",
        "TravelVax": "Travel Vaccine",
    }
    product_ids = {brand: f"PRD-{idx + 1001}" for idx, brand in enumerate(brands)}
    brand_name = _choice(rng, brands, rows, [0.30, 0.20, 0.20, 0.18, 0.12])
    product_name = brand_name
    product_family = np.array([product_families[item] for item in brand_name])
    product_id = np.array([product_ids[item] for item in brand_name])

    account_tier = _choice(rng, ["National", "Regional", "Local"], rows, [0.32, 0.44, 0.24])
    region = _choice(rng, ["Northeast", "Southeast", "Midwest", "West", "National"], rows, [0.20, 0.19, 0.22, 0.21, 0.18])
    product_scope = _choice(rng, ["Single Product", "Multi Product"], rows, [0.62, 0.38])
    brand_scope = np.where(product_scope == "Single Product", brand_name, "Multi-brand vaccine portfolio")
    new_vs_existing_customer = _choice(rng, ["New", "Existing"], rows, [0.28, 0.72])
    prior_contract_status = np.where(
        new_vs_existing_customer == "New",
        "No Prior Contract",
        _choice(rng, ["Accepted", "Negotiated", "Rejected"], rows, [0.56, 0.34, 0.10]),
    )

    strategic_account_flag = (rng.random(rows) < np.where(account_tier == "National", 0.72, 0.28)).astype(int)
    is_child_customer = (rng.random(rows) < 0.34).astype(int)
    negotiation_round_number = rng.choice([1, 2, 3, 4], size=rows, p=[0.52, 0.27, 0.15, 0.06])
    round_number = negotiation_round_number

    prior_volume = rng.lognormal(mean=10.45, sigma=0.68, size=rows).clip(8_000, 120_000).round().astype(int)
    prior_year_market_share = rng.normal(42, 14, rows).clip(8, 78)
    prior_year_attainment_pct = rng.normal(91, 13, rows).clip(45, 125)
    historical_win_rate = rng.beta(5.5, 2.8, rows).clip(0.18, 0.96)
    historical_negotiation_rate = rng.beta(3.0, 4.5, rows).clip(0.05, 0.70)
    historical_rejection_rate = (1 - historical_win_rate - historical_negotiation_rate * 0.35).clip(0.02, 0.48)
    prior_acceptance_flag = (rng.random(rows) < historical_win_rate).astype(int)
    prior_rebate_pct = rng.normal(13.5, 5.0, rows).clip(2, 28)
    prior_net_price = rng.normal(112, 18, rows).clip(58, 172)
    prior_returnability_pct = rng.normal(7.5, 2.4, rows).clip(0, 15)
    prior_cancellation_pct = rng.normal(5.0, 2.2, rows).clip(0, 12)

    competitive_pressure_score = rng.normal(52, 22, rows).clip(5, 98)
    competitor_intensity = pd.cut(
        competitive_pressure_score,
        bins=[0, 35, 70, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    competitor_rebate_estimate_pct = (prior_rebate_pct + rng.normal(2.5, 4.0, rows)).clip(3, 34)
    estimated_competitor_price = (prior_net_price * rng.normal(0.98, 0.10, rows)).clip(48, 180)
    competitor_contract_activity_score = rng.normal(50, 20, rows).clip(5, 95)
    competitor_brand = _choice(rng, ["GSK", "Pfizer", "Moderna", "Seqirus", "Merck"], rows, [0.26, 0.24, 0.20, 0.18, 0.12])
    market_growth_pct = rng.normal(4.5, 5.5, rows).clip(-8, 18)
    demand_forecast = (prior_volume * rng.normal(1.07, 0.20, rows)).clip(5_000, 160_000).round().astype(int)
    supply_constraint_score = rng.normal(36, 20, rows).clip(2, 92)

    price_pressure = (competitive_pressure_score - 50) / 100
    net_price = (prior_net_price * (1 + rng.normal(0.01, 0.075, rows) - price_pressure * 0.08)).clip(50, 185)
    rebate_pct = (prior_rebate_pct + rng.normal(2.0, 4.2, rows) + price_pressure * 6).clip(1, 35)
    multi_year_discount_pct = rng.choice([0, 1, 2, 3, 4, 5, 6, 7], size=rows, p=[0.18, 0.12, 0.18, 0.20, 0.14, 0.10, 0.05, 0.03])
    cancellation_pct = (prior_cancellation_pct + rng.normal(0.5, 2.0, rows)).clip(0, 15)
    reimbursement_rate_pct = rng.normal(86, 8.5, rows).clip(55, 99)
    volume_commitment = (prior_volume * rng.normal(1.07, 0.18, rows)).clip(5_000, 160_000).round().astype(int)
    market_share_commitment_pct = (prior_year_market_share + rng.normal(4.5, 8, rows)).clip(5, 90)
    contract_duration_years = rng.choice([1, 2, 3], size=rows, p=[0.46, 0.36, 0.18])
    returnability_pct = (prior_returnability_pct + rng.normal(0.3, 2.0, rows)).clip(0, 18)
    shipping_commitment = _choice(rng, ["Standard", "Expedited", "Cold-chain Priority"], rows, [0.53, 0.29, 0.18])
    payment_terms_days = rng.choice([30, 45, 60, 75, 90], size=rows, p=[0.26, 0.24, 0.30, 0.12, 0.08])
    price_protection_flag = (rng.random(rows) < 0.44).astype(int)
    early_pay_discount_pct = rng.choice([0, 0.5, 1.0, 1.5, 2.0], size=rows, p=[0.38, 0.20, 0.24, 0.12, 0.06])
    admin_fee_pct = rng.normal(2.1, 0.9, rows).clip(0, 5)
    product_priority = _choice(rng, ["Core", "Growth", "Defend", "Selective"], rows, [0.40, 0.25, 0.25, 0.10])

    gross_revenue = net_price * volume_commitment
    estimated_rebate_dollars = gross_revenue * rebate_pct / 100
    estimated_net_revenue = gross_revenue - estimated_rebate_dollars
    margin_proxy = ((net_price - 45) / net_price - rebate_pct / 100 - admin_fee_pct / 100).clip(0.08, 0.62)
    volume_gap_vs_prior = (volume_commitment - prior_volume) / prior_volume * 100
    market_share_gap_vs_prior = market_share_commitment_pct - prior_year_market_share
    price_gap_vs_prior = (net_price - prior_net_price) / prior_net_price * 100
    reimbursement_gap = reimbursement_rate_pct - 80

    deal_attractiveness_score = (
        52
        + (rebate_pct - prior_rebate_pct) * 1.15
        - price_gap_vs_prior * 0.55
        + reimbursement_gap * 0.55
        + np.minimum(volume_gap_vs_prior, 35) * 0.20
        + (prior_year_attainment_pct - 85) * 0.18
        + historical_win_rate * 18
        - competitive_pressure_score * 0.16
        - np.maximum(market_share_gap_vs_prior - 12, 0) * 0.50
        + strategic_account_flag * 3.0
        + price_protection_flag * 2.2
        + rng.normal(0, 6, rows)
    ).clip(0, 100)

    accepted_score = (
        -1.0
        + (deal_attractiveness_score - 50) / 13
        + historical_win_rate * 1.5
        + prior_acceptance_flag * 0.35
        + strategic_account_flag * 0.25
        + (reimbursement_rate_pct - 82) / 20
        + np.minimum(rebate_pct - prior_rebate_pct, 8) / 10
        - np.maximum(price_gap_vs_prior, 0) / 13
        - np.maximum(competitive_pressure_score - 70, 0) / 26
        + rng.normal(0, 0.45, rows)
    )
    negotiated_score = (
        0.35
        + np.abs(price_gap_vs_prior) / 18
        + np.abs(market_share_gap_vs_prior) / 18
        + (competitive_pressure_score - 40) / 60
        + historical_negotiation_rate * 1.0
        + (negotiation_round_number - 1) * 0.18
        + np.maximum(volume_gap_vs_prior - 18, 0) / 24
        + rng.normal(0, 0.40, rows)
    )
    rejected_score = (
        -0.10
        + np.maximum(price_gap_vs_prior, 0) / 7
        + np.maximum(80 - reimbursement_rate_pct, 0) / 12
        + np.maximum(10 - rebate_pct, 0) / 5
        + np.maximum(competitive_pressure_score - 65, 0) / 16
        + np.maximum(75 - prior_year_attainment_pct, 0) / 16
        + historical_rejection_rate * 1.1
        + (prior_contract_status == "Rejected") * 0.50
        - price_protection_flag * 0.18
        + rng.normal(0, 0.40, rows)
    )

    scores = np.vstack([accepted_score, negotiated_score, rejected_score]).T
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    sampled_indexes = np.array([rng.choice(3, p=probabilities[idx]) for idx in range(rows)])
    business_rule_indexes = probabilities.argmax(axis=1)
    outcome_indexes = np.where(rng.random(rows) < 0.72, business_rule_indexes, sampled_indexes)
    outcome = np.array(OUTCOME_LABELS)[outcome_indexes]

    season_id = _choice(rng, ["2025-2026", "2026-2027", "2027-2028"], rows, [0.26, 0.52, 0.22])
    l1_parent = np.where(is_child_customer == 1, np.char.add(customer_name.astype(str), " Parent"), customer_name)

    return pd.DataFrame(
        {
            "deal_id": [f"DEAL-{100000 + idx}" for idx in range(rows)],
            "contract_id": [f"CTR-{24000 + idx}" for idx in range(rows)],
            "customer_id": [f"CUST-{rng.integers(1000, 9999)}" for _ in range(rows)],
            "product_id": product_id,
            "brand_name": brand_name,
            "season_id": season_id,
            "round_number": round_number,
            "l1_parent": l1_parent,
            "is_child_customer": is_child_customer,
            "customer_segment": customer_segment,
            "customer_class": customer_class,
            "customer_name": customer_name,
            "contract_name": [f"{segment} {season} Vaccine Agreement" for segment, season in zip(customer_segment, season_id)],
            "product_scope": product_scope,
            "brand_scope": brand_scope,
            "new_vs_existing_customer": new_vs_existing_customer,
            "account_tier": account_tier,
            "region": region,
            "strategic_account_flag": strategic_account_flag,
            "prior_contract_status": prior_contract_status,
            "negotiation_round_number": negotiation_round_number,
            "product_name": product_name,
            "product_family": product_family,
            "flu_season": season_id,
            "product_priority": product_priority,
            "competitor_intensity": competitor_intensity,
            "prior_volume": prior_volume,
            "prior_year_market_share": prior_year_market_share.round(1),
            "prior_year_attainment_pct": prior_year_attainment_pct.round(1),
            "prior_acceptance_flag": prior_acceptance_flag,
            "prior_rebate_pct": prior_rebate_pct.round(2),
            "prior_net_price": prior_net_price.round(2),
            "prior_returnability_pct": prior_returnability_pct.round(2),
            "prior_cancellation_pct": prior_cancellation_pct.round(2),
            "historical_win_rate": historical_win_rate.round(3),
            "historical_negotiation_rate": historical_negotiation_rate.round(3),
            "historical_rejection_rate": historical_rejection_rate.round(3),
            "net_price": net_price.round(2),
            "rebate_pct": rebate_pct.round(2),
            "multi_year_discount_pct": multi_year_discount_pct.astype(float),
            "cancellation_pct": cancellation_pct.round(2),
            "reimbursement_rate_pct": reimbursement_rate_pct.round(2),
            "volume_commitment": volume_commitment,
            "market_share_commitment_pct": market_share_commitment_pct.round(1),
            "contract_duration_years": contract_duration_years,
            "returnability_pct": returnability_pct.round(2),
            "shipping_commitment": shipping_commitment,
            "payment_terms_days": payment_terms_days,
            "price_protection_flag": price_protection_flag,
            "early_pay_discount_pct": early_pay_discount_pct.astype(float),
            "admin_fee_pct": admin_fee_pct.round(2),
            "competitive_pressure_score": competitive_pressure_score.round(1),
            "estimated_competitor_price": estimated_competitor_price.round(2),
            "competitor_rebate_estimate_pct": competitor_rebate_estimate_pct.round(2),
            "competitor_contract_activity_score": competitor_contract_activity_score.round(1),
            "competitor_brand": competitor_brand,
            "market_growth_pct": market_growth_pct.round(2),
            "demand_forecast": demand_forecast,
            "supply_constraint_score": supply_constraint_score.round(1),
            "gross_revenue": gross_revenue.round(2),
            "estimated_rebate_dollars": estimated_rebate_dollars.round(2),
            "estimated_net_revenue": estimated_net_revenue.round(2),
            "margin_proxy": margin_proxy.round(3),
            "volume_gap_vs_prior": volume_gap_vs_prior.round(2),
            "market_share_gap_vs_prior": market_share_gap_vs_prior.round(2),
            "price_gap_vs_prior": price_gap_vs_prior.round(2),
            "reimbursement_gap": reimbursement_gap.round(2),
            "deal_attractiveness_score": deal_attractiveness_score.round(1),
            "outcome": outcome,
        }
    )


def save_dummy_data(path: Path = DATA_PATH, rows: int = 500, seed: int = 42) -> pd.DataFrame:
    ensure_directories()
    data = generate_dummy_data(rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


if __name__ == "__main__":
    dataset = save_dummy_data()
    print(f"Saved synthetic data to {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Dataset shape: {dataset.shape}")
    print("Outcome distribution:")
    print(dataset["outcome"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
