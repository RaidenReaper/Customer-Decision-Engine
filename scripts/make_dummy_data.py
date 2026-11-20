"""Generate the demo build's synthetic data (data/hs_features.parquet, data/hs_history.parquet).

Every account is fictional: names, IDs, addresses and contracts are invented,
and each account's attributes are drawn from a *different* random donor row
per attribute group, so no real account's profile survives. Values keep the
schema and the value ranges the app and models expect (prices per dose are
kept at their existing per-product levels). Everything derived is recomputed:
price-vs-benchmark, premium gap, log doses, encodings, provenance codes, and
the model-scored columns (prob0, sensitivity bands, quadrant, account level).

    python scripts/make_dummy_data.py            # reads data/ as the template, rewrites it
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import load_models  # noqa: E402
from src.utils import (  # noqa: E402
    BUYING_GROUP_CODES,
    DATA_PATH,
    FEATURE_COLUMNS,
    HISTORY_PATH,
    LEVER_SPECS,
    PRODUCT_KEYS,
    load_model_metadata,
)

SEED = 20261120
N_ACCOUNTS = 120
DEFAULT_ACCOUNT = "Northwind Health"

PRODUCT_RENAME = {"Viresta": "Viresta", "Immunexa": "Immunexa", "Noventra": "Noventra"}
INCUMBENT_RENAME = {"Customer": "Customer", "Competitor 1": "Competitor 1", "Competitor 2": "Competitor 2"}
# The template data already carries generic competitor labels; anything else maps to "Competitor E".
COMPETITOR_PRODUCT_RENAME = {f"competitor {c}": f"Competitor {c.upper()}" for c in "abcde"}
WALLET_RENAME = {"Customer Loyalist": "Customer Loyalist", "Customer Leaning": "Customer Leaning"}
ACCOUNT_PRIORITY = {"Viresta": 1.5, "Noventra": 1.0, "Immunexa": 0.5}

CUSTOMER = "bob_L1-Buying Group"
PRODUCT = "product_standardised"
WALLET_TEXT = "bob_Prior Season Shr of Wallet (L1BG)"
PRIOR_DOSES = "bob_L1BG Prior Season Customer Doses"

# Account-level attributes (constant across an account's products in the
# source). Each tuple is sampled together, from its own donor account.
ACCOUNT_GROUPS = [
    (WALLET_TEXT, "bob_Prior Season Shr of Wallet_enc"),
    ("bob_Prior Season HS Partnership Advocacy (L1BG+Brand)", "bob_Prior Season HS Partnership Advocacy_enc"),
    ("bob_Prior Season HS Segmentation Flag (L1BG)",),
    ("buying_group_5", "buying_group_category_cat"),
    ("bob_contract_type_cd",),
    ("multi_year_flag", "multi_year_contract_flag"),
    ("clinical_bucket", "apt_clinical_bucket_enc"),
    ("negotiation_stance_bucket", "negotiation_receptivity_score"),
    ("cde_engagement_length_log",),
    ("cde_net_sentiment_score",),
    ("competitive_landscape_bucket",),
    ("competitor_strategy_bucket",),
    ("returnability_pct",),
    ("account_confidence_score", "account_confidence_label"),
]
# Row-level groups, sampled from a donor row of the same product.
PRICE_GROUP = [
    "dead_net_price_per_dose", "standard_contract_price_per_dose", "discount_gap_pct",
    "competitor_price_used", "competitor_list_price", "competitor1_product_name",
]
COMPETITIVE_GROUP = [
    "flips_prior_2_seasons", "flip_risk_score", "yoy_ms_change_pct_prior",
    "sam_risk_flag_product", "sam_feat_high_or_medium_risk_flag",
    "bob_prior_year_incumbent_manufacturer", "is_customer_prior_incumbent",
]
VOLUME_GROUP = [
    "ddd_customer_doses", "market_volume_total", "ddd_competitor1_doses", "ddd_competitor2_doses",
    "ddd_competitor3_doses", "tam_doses", "tam_dollars", PRIOR_DOSES, "bob_L1BG Prior Season Customer MS",
]

PLACES = [
    "Northwind", "Brightwater", "Larkspur", "Silverpine", "Copperfield", "Amberlake", "Willowmere",
    "Stonebridge", "Elmhaven", "Foxglove", "Ironwood", "Oakhollow", "Ravenwood", "Sunmeadow",
    "Thistledown", "Bramblecrest", "Clearbrook", "Driftwood", "Evergale", "Fernhill", "Glenmoor",
    "Hollowcreek", "Juniper Bay", "Kestrel Point", "Lumen Valley", "Maplestone", "Nettlefield",
    "Orchard Ridge", "Pinehaven", "Quarry Hill", "Rosewater", "Sablewood", "Tidewater", "Umberfield",
    "Vale Crossing", "Wrenfield", "Yarrow Glen", "Zephyr Coast", "Ashgrove", "Bluefen", "Cobalt Ridge",
    "Dunmere", "Emberly", "Falconcrest", "Greystone Bay", "Harrowgate", "Indigo Falls", "Jasper Hollow",
]
SUFFIXES = ["Health", "Health System", "Healthcare", "Medical Group", "Care Network", "Regional Medical Center"]
STREETS = ["Harbor", "Maple", "Summit", "Lantern", "Meadow", "Beacon", "Granite", "Willow", "Orchard", "Cedar"]
STREET_TYPES = ["Ave", "Blvd", "St", "Pkwy", "Dr", "Way"]
CITIES = ["Port Alder", "Fairhollow", "Brookmere", "East Tilden", "Marlow Springs", "Carrow", "Halston", "Westvale"]
STATES = ["OH", "TX", "CA", "PA", "NY", "IL", "GA", "NC", "MI", "WA", "AZ", "MN"]


def account_names(rng: np.random.Generator) -> list[str]:
    combos = [f"{p} {s}" for p in PLACES for s in SUFFIXES if f"{p} {s}" != DEFAULT_ACCOUNT]
    picked = list(rng.choice(combos, size=N_ACCOUNTS - 1, replace=False))
    return sorted([DEFAULT_ACCOUNT, *picked])


def rename_competitor_product(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    key = str(value).replace("\xa0", "").strip().lower()
    return COMPETITOR_PRODUCT_RENAME.get(key, "Competitor E")


def build_features(template: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    template = template.copy()
    template[PRODUCT] = template[PRODUCT].replace(PRODUCT_RENAME)
    accounts = template.drop_duplicates(CUSTOMER)
    by_product = {p: template[template[PRODUCT] == p].reset_index(drop=True) for p in PRODUCT_KEYS}

    rows = []
    for i, name in enumerate(account_names(rng)):
        account_values = {}
        for group in ACCOUNT_GROUPS:
            donor = accounts.iloc[rng.integers(len(accounts))]
            account_values.update({c: donor[c] for c in group})
        account_id = f"IDN9{rng.integers(10**6, 10**7)}"
        address = (
            f"{rng.integers(10, 9900)} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}, "
            f"{rng.choice(CITIES)}, {rng.choice(STATES)}"
        )
        contract = f"PG-{1001 + i}"
        size = float(rng.lognormal(0.0, 0.35))  # one account-size factor for every volume
        for product, pool in by_product.items():
            row = pool.iloc[rng.integers(len(pool))].copy()  # constants / all-NaN columns
            for group in (PRICE_GROUP, COMPETITIVE_GROUP, VOLUME_GROUP):
                donor = pool.iloc[rng.integers(len(pool))]
                for c in group:
                    row[c] = donor[c]
            for c, v in account_values.items():
                row[c] = v
            row[CUSTOMER] = name
            row["IQVIA LOC ID"] = account_id
            row["bob_address"] = address
            row["bob_buying_group_name"] = name.upper()
            row["contract_id"] = contract
            row["cy_rmus_price_group"] = contract
            row["rmus_contract_title"] = f"FL {name.upper()} 27-28"
            for c in VOLUME_GROUP:
                if c != "bob_L1BG Prior Season Customer MS" and pd.notna(row[c]):
                    row[c] = float(np.round(float(row[c]) * size))
            if pd.notna(row["cde_engagement_length_log"]):
                row["cde_engagement_length_log"] = float(row["cde_engagement_length_log"]) + rng.normal(0, 0.25)
            rows.append(row)

    frame = pd.DataFrame(rows).reset_index(drop=True)

    # Brand-neutral text values.
    frame[WALLET_TEXT] = frame[WALLET_TEXT].replace(WALLET_RENAME)
    frame["bob_prior_year_incumbent_manufacturer"] = frame["bob_prior_year_incumbent_manufacturer"].replace(INCUMBENT_RENAME)
    frame["is_customer_prior_incumbent"] = np.where(
        frame["bob_prior_year_incumbent_manufacturer"].isna(), np.nan,
        (frame["bob_prior_year_incumbent_manufacturer"] == "Customer").astype(float),
    )
    frame["competitor1_product_name"] = frame["competitor1_product_name"].map(rename_competitor_product)

    # Encodings follow their labels (lever specs are the source of truth).
    wallet = LEVER_SPECS["bob_Prior Season Shr of Wallet_enc"]["values"]
    frame["bob_Prior Season Shr of Wallet_enc"] = frame[WALLET_TEXT].map(wallet)
    frame["buying_group_category_cat"] = frame["buying_group_5"].map(BUYING_GROUP_CODES).astype("int64")

    # Price-linked columns: one price per row, benchmark per product.
    price = frame["dead_net_price_per_dose"].astype(float)
    for c in ("py_dead_price_customer", "dead_net_price_per_dose_pre_offer", "customer_list_price"):
        frame[c] = price
    bench = frame["standard_price_benchmark"].astype(float)
    frame["price_vs_std_benchmark_updated"] = 100 * (bench - price) / bench
    comp = frame["competitor_price_used"].astype(float)
    frame["premium_gap_vs_min_competitor_pct"] = 100 * (price / comp - 1)  # NaN where no competitor price

    # Volume-derived columns.
    frame["log_prior_customer_doses"] = np.log1p(frame[PRIOR_DOSES].astype(float).clip(lower=0))
    frame["ddd_customer_market_share"] = frame["ddd_customer_doses"] / frame["market_volume_total"].replace(0, np.nan)

    # Provenance codes (src/confidence.py reads them).
    for c in [c for c in frame.columns if c.startswith("prov__")]:
        feature = c[len("prov__"):]
        present = frame[feature].notna()
        missing = "not_applicable" if feature in ("log_prior_customer_doses", "yoy_ms_change_pct_prior") else "missing"
        frame[c] = np.where(present, "observed", missing)
        if feature == "yoy_price_change_pct":
            frame[c] = "not_applicable"

    return score(frame)


def _vw_quantile(s: pd.Series, volume: pd.Series, q: float) -> float:
    order = s.sort_values(kind="mergesort").index
    weights = pd.to_numeric(volume.loc[order], errors="coerce").fillna(0.0)
    if weights.sum() <= 0:
        return float(s.quantile(q))
    reached = (weights.cumsum() / weights.sum()).ge(q)
    return float(s.loc[order][reached.to_numpy()].iloc[0])


def _matrix(block: pd.DataFrame, price: pd.Series | None = None) -> pd.DataFrame:
    X = block[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    if price is not None:
        base = block["dead_net_price_per_dose"]
        bench = block["standard_price_benchmark"]
        m = price / base
        X["price_vs_std_benchmark_updated"] = (100 * (bench - price) / bench).to_numpy()
        X["premium_gap_vs_min_competitor_pct"] = (100 * (m * (1 + block["premium_gap_vs_min_competitor_pct"] / 100) - 1)).to_numpy()
        prior = block["py_dead_price_customer"]
        X["yoy_price_change_pct"] = (100 * (price - prior) / prior.where(prior != 0)).to_numpy()
    return X


def score(frame: pd.DataFrame) -> pd.DataFrame:
    """prepare_real_data.score_population() + assign_account_level(), on the shipped models."""
    models = load_models(load_model_metadata())
    for product, key in PRODUCT_KEYS.items():
        mask = frame[PRODUCT] == product
        block = frame.loc[mask]
        model = models[key]
        p0 = model.predict_proba(_matrix(block))[:, 1]
        p10 = model.predict_proba(_matrix(block, block["dead_net_price_per_dose"] * 1.1))[:, 1]
        p_bm = model.predict_proba(_matrix(block, block["standard_price_benchmark"]))[:, 1]
        frame.loc[mask, "prob0"] = 100 * p0.astype(float)
        frame.loc[mask, "prob10"] = 100 * p10.astype(float)
        frame.loc[mask, "prob_benchmark"] = 100 * p_bm.astype(float)
        frame.loc[mask, "sensitivity_pp"] = 100 * (p0 - p10).astype(float)
        frame.loc[mask, "sensitivity_benchmark_signed_pp"] = 100 * (p_bm - p0).astype(float)

    dead, bench = frame["dead_net_price_per_dose"], frame["standard_price_benchmark"]
    frame["benchmark_price_change_pct"] = (100 * (bench / dead - 1)).mask((dead - bench).abs() < 0.01, 0.0)
    pct = frame["benchmark_price_change_pct"]
    frame["benchmark_direction"] = np.select(
        [pct > 0.01, pct < -0.01], ["price up to benchmark", "price down to benchmark"], default="at benchmark"
    )
    frame["sensitivity_benchmark_pp"] = frame["sensitivity_benchmark_signed_pp"].abs()
    for product in PRODUCT_KEYS:
        mask = frame[PRODUCT] == product
        s, volume = frame.loc[mask, "sensitivity_benchmark_pp"], frame.loc[mask, "market_volume_total"]
        frame.loc[mask, "sensitivity_low_cutoff_pp"] = _vw_quantile(s, volume, 0.5)
        frame.loc[mask, "sensitivity_high_cutoff_pp"] = _vw_quantile(s, volume, 0.7)
    s = frame["sensitivity_benchmark_pp"]
    frame["sensitivity_band_lmh"] = np.select(
        [(s < frame["sensitivity_low_cutoff_pp"]) | (s < 0.05), s >= frame["sensitivity_high_cutoff_pp"]],
        ["Low", "High"], default="Medium",
    )
    sensitive = frame["sensitivity_band_lmh"].isin(["Medium", "High"])
    frame["sensitivity_flag"] = np.where(sensitive, "Sensitive", "Non-sensitive")
    ours = frame["bob_prior_year_incumbent_manufacturer"].eq("Customer")
    frame["quadrant_2x2"] = np.where(
        ours,
        np.where(sensitive, "Targeted Retention", "Protect Margin"),
        np.where(sensitive, "Competitive Price Focus", "Build Value & Relationship"),
    )

    weighted = frame.assign(_w=frame["sensitivity_benchmark_pp"] * frame[PRODUCT].map(ACCOUNT_PRIORITY))
    driving = weighted.sort_values("_w", ascending=False, kind="stable").drop_duplicates(CUSTOMER).set_index(CUSTOMER)
    for target, source in {
        "account_driving_product": PRODUCT,
        "account_sensitivity_pp": "sensitivity_benchmark_pp",
        "account_sensitivity_band": "sensitivity_band_lmh",
        "account_sensitivity_basis": "benchmark_direction",
    }.items():
        frame[target] = frame[CUSTOMER].map(driving[source])
    return frame


def build_history(template: pd.DataFrame, features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    template = template.copy()
    template[PRODUCT] = template[PRODUCT].replace(PRODUCT_RENAME)
    volume_cols = [
        "bob_totalfinalsales", "ddd_customer_doses", "ddd_competitor1_doses", "ddd_competitor2_doses",
        "ddd_competitor3_doses", "total_prebook_cy",
    ]
    pools = {key: block.reset_index(drop=True) for key, block in template.groupby([PRODUCT, "season"])}
    rows = []
    for _, feat in features.iterrows():
        size = float(rng.lognormal(0.0, 0.35))
        for season in sorted(template["season"].unique()):
            pool = pools.get((feat[PRODUCT], season))
            if pool is None or pool.empty:
                continue
            row = pool.iloc[rng.integers(len(pool))].copy()
            row[CUSTOMER] = feat[CUSTOMER]
            row["IQVIA LOC ID"] = feat["IQVIA LOC ID"]
            row["contract_id"] = feat["contract_id"]
            row["cy_rmus_price_group"] = feat["cy_rmus_price_group"]
            row[WALLET_TEXT] = feat[WALLET_TEXT]
            for c in volume_cols:
                if pd.notna(row[c]):
                    row[c] = float(np.round(float(row[c]) * size))
            if season == feat["season"]:
                row["dead_net_price_per_dose"] = feat["dead_net_price_per_dose"]
            rows.append(row)
    history = pd.DataFrame(rows).reset_index(drop=True)
    history[WALLET_TEXT] = history[WALLET_TEXT].replace(WALLET_RENAME)
    history["competitor1_product_name"] = history["competitor1_product_name"].map(rename_competitor_product)
    total = history[["ddd_customer_doses", "ddd_competitor1_doses", "ddd_competitor2_doses", "ddd_competitor3_doses"]].sum(axis=1, min_count=1)
    history["ddd_customer_market_share"] = history["ddd_customer_doses"] / total.replace(0, np.nan)
    return history[template.columns]


def main() -> None:
    rng = np.random.default_rng(SEED)
    features_template = pd.read_parquet(DATA_PATH)
    history_template = pd.read_parquet(HISTORY_PATH)

    features = build_features(features_template, rng)[features_template.columns]
    for c in features.columns:  # keep the template's dtypes
        try:
            features[c] = features[c].astype(features_template[c].dtype)
        except (TypeError, ValueError):
            pass
    history = build_history(history_template, features, rng)
    for c in history.columns:
        try:
            history[c] = history[c].astype(history_template[c].dtype)
        except (TypeError, ValueError):
            pass

    features.to_parquet(DATA_PATH, index=False)
    history.to_parquet(HISTORY_PATH, index=False)
    print(f"features: {features.shape}, {features[CUSTOMER].nunique()} accounts -> {DATA_PATH}")
    print(f"history:  {history.shape} -> {HISTORY_PATH}")
    print(features.groupby(PRODUCT)["prob0"].describe().round(1))


if __name__ == "__main__":
    main()
