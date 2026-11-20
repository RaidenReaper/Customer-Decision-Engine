"""Scoring against the three per-product XGBoost win/loss models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from src import confidence
    from src.utils import (
        PRODUCT_KEYS,
        FEATURE_COLUMNS,
        FEATURE_TYPES,
        FEATURE_LABELS,
        LEVER_FEATURES,
        MODEL_DIR,
        NUMERIC_FEATURE_COLUMNS,
        PRICE_BASE_COLUMN,
        PRODUCT_COLUMN,
        CATEGORICAL_FEATURES,
        WIN_INDEX,
        build_feature_row,
        load_model_metadata,
        product_key,
        threshold_for,
    )
except ModuleNotFoundError:  # running as a script from inside src/
    import confidence
    from utils import (
        PRODUCT_KEYS,
        FEATURE_COLUMNS,
        FEATURE_TYPES,
        FEATURE_LABELS,
        LEVER_FEATURES,
        MODEL_DIR,
        NUMERIC_FEATURE_COLUMNS,
        PRICE_BASE_COLUMN,
        PRODUCT_COLUMN,
        CATEGORICAL_FEATURES,
        WIN_INDEX,
        build_feature_row,
        load_model_metadata,
        product_key,
        threshold_for,
    )


# ---------------------------------------------------------------------------
# Version guard (added 8/28). The v3 model bundles are serialised under
# xgboost 3.3.0. xgboost 2.x loads them WITHOUT WARNING but misreads the 3.x
# base_score encoding (a JSON array, e.g. [7.829787E-1]) as 0.5, shifting
# every prediction by a constant logit offset. Fail fast instead.
_XGB_MAJOR_REQUIRED = 3
_xgb_major = int(xgb.__version__.split(".")[0])
if _xgb_major < _XGB_MAJOR_REQUIRED:
    raise RuntimeError(
        f"xgboost {xgb.__version__} cannot safely load the v3 model bundles "
        f"(built under xgboost 3.3.0): 2.x silently drops base_score and every "
        f"prediction shifts by a constant logit offset. Install xgboost>=3.3.0 "
        f"(see requirements.txt)."
    )


def load_models(
    metadata: dict[str, Any] | None = None,
    model_dir: Path = MODEL_DIR,
) -> dict[str, Any]:
    """Load one XGBClassifier per product key (HD / STD / FLK).

    The v2 (8/20) model package saves each product as a dict bundle
    ({"model": XGBClassifier, "features": [...], ...}), not a bare
    sklearn.Pipeline like the prior model — it carries no preprocessing step,
    since the classifier was trained with missing=nan and handles missing
    values natively. Unwrapped here so every caller gets the classifier
    directly, same as before.
    """
    metadata = metadata or load_model_metadata()
    models: dict[str, Any] = {}
    for key, entry in metadata["products"].items():
        path = model_dir / entry["model_file"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact {path.name}. "
                "Run python scripts/make_dummy_data.py first."
            )
        bundle = joblib.load(path)
        models[key] = bundle["model"] if isinstance(bundle, dict) else bundle
    return models


def load_fold_models(model_dir: Path = MODEL_DIR) -> dict[str, list[xgb.Booster]]:
    """5 out-of-fold boosters per product (v3 package), used only for the
    win/loss confidence "stability" component (S, src/confidence.py) — the
    spread of P(win) across the 5 folds at the scenario's feature values. Native
    booster JSONs, not joblib, so they load regardless of xgboost version.
    """
    fold_dir = model_dir / "fold_models"
    result: dict[str, list[xgb.Booster]] = {}
    for product_value, prefix in PRODUCT_KEYS.items():  # files: fold_{HD,STD,FLK}_k.json
        boosters = []
        for k in range(5):
            path = fold_dir / f"fold_{prefix}_{k}.json"
            if not path.exists():
                break
            booster = xgb.Booster()
            booster.load_model(str(path))
            boosters.append(booster)
        if boosters:
            result[product_value] = boosters
    return result


def feature_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Single-row frame with the exact column order and dtypes the models
    expect: float for every numeric feature, and any categorical feature
    (CATEGORICAL_FEATURES; none in v8) as a pandas Categorical over its
    trained categories — XGBoost reads category codes from the dtype, so a
    plain string column would not score."""
    frame = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    for name in NUMERIC_FEATURE_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
    for name, categories in CATEGORICAL_FEATURES.items():
        frame[name] = pd.Categorical(frame[name], categories=list(categories))
    return frame


def _dmatrix(frame: pd.DataFrame) -> xgb.DMatrix:
    # feature_types marks v27's two float-coded categories (buying group,
    # season) as categorical, as the model was trained; a hand-built DMatrix
    # would otherwise read them as numbers. (predict_proba on the classifier
    # already applies the model's own feature_types.)
    return xgb.DMatrix(
        frame, feature_names=FEATURE_COLUMNS, feature_types=FEATURE_TYPES, missing=np.nan, enable_categorical=True
    )


def confidence_band(margin: float) -> str:
    """How far the win probability sits from this product's decision threshold."""
    if margin >= 0.25:
        return "High"
    if margin >= 0.10:
        return "Moderate"
    return "Low"


def _same_value(a: Any, b: Any) -> bool:
    """Equal, treating NaN == NaN (an untouched "No Data" lever)."""
    a_missing = a is None or pd.isna(a)
    b_missing = b is None or pd.isna(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def scenario_provenance(
    base_row: pd.Series,
    features: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, str]:
    """Provenance code per confidence input for this scenario.

    Starts from the per-row codes src/prepare_real_data.py wrote into the
    parquet (``prov__<feature>``; see derive_provenance() there for the
    rules). Then the scenario's own edits:
      * a Contracting Lever moved off the account's own value -> manual
        (or missing, when moved to "No Data" / "Unknown", which feed NaN);
      * a moved price makes Year-over-year Price Change a scenario input —
        still outside D, but now compared with history in H.
    Rows without the columns (an older parquet) fall back to "recorded if
    present, else missing".
    """
    provenance: dict[str, str] = {}
    for feature in confidence.CONFIDENCE_FEATURES:
        code = base_row.get(confidence.PROVENANCE_PREFIX + feature)
        if code is None or pd.isna(code):
            code = confidence.MISSING if pd.isna(base_row.get(feature)) else confidence.OBSERVED
        provenance[feature] = str(code)

    for feature in LEVER_FEATURES:
        value = features.get(feature)
        if not _same_value(value, base_row.get(feature)):
            provenance[feature] = confidence.MISSING if pd.isna(value) else confidence.MANUAL

    price = overrides.get("dead_net_price")
    base_price = base_row.get(PRICE_BASE_COLUMN)
    if price is not None and base_price is not None and not pd.isna(base_price):
        if abs(float(price) - float(base_price)) >= 0.005:  # build_feature_row's own tolerance
            provenance["yoy_price_change_pct"] = confidence.SCENARIO
    return provenance


def winloss_confidence(
    product_value: str,
    features: dict[str, Any],
    win_probability: float,
    threshold: float,
    fold_boosters: list[xgb.Booster],
    spec: dict[str, Any],
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Per-product prediction confidence, v5 (11/13 — see src/confidence.py):

        confidence = 0.30 x D + 0.30 x H + 0.30 x M + 0.10 x S

    Scenario-aware: pass whatever ``features`` the current lever overrides
    produced, and the fold predictions (S), the margin (M) and the
    historical comparison (H) all follow that scenario. Returns the v4 keys
    the app and report already read, plus ``winloss_confidence_detail`` —
    the full assessment, components and explanation text.
    """
    frame = feature_frame(features)
    matrix = _dmatrix(frame)
    fold_win_probabilities = [float(booster.predict(matrix)[0]) for booster in fold_boosters]
    detail = confidence.assess(
        product=product_value,
        features=features,
        provenance=provenance,
        win_probability=win_probability,
        threshold=threshold,
        fold_probabilities=fold_win_probabilities,
        reference=spec["historical_reference"][product_value],
        lever_features=LEVER_FEATURES,
    )
    return confidence_keys(detail)


def confidence_keys(detail: dict[str, Any] | None) -> dict[str, Any]:
    """The result-dict keys for one confidence assessment (None -> all None)."""
    if detail is None:
        return {
            "winloss_confidence": None,
            "winloss_confidence_label": None,
            "winloss_confidence_stability_score": None,
            "winloss_confidence_margin_score": None,
            "winloss_confidence_detail": None,
        }
    return {
        "winloss_confidence": detail["score"],
        "winloss_confidence_label": detail["band"],
        "winloss_confidence_stability_score": detail["components"]["S"],
        "winloss_confidence_margin_score": detail["components"]["M"],
        "winloss_confidence_detail": detail,
    }


def confidence_at_probability(result: dict[str, Any], win_probability: float) -> dict[str, Any]:
    """Re-centre a scored result's confidence on another win probability
    (the app's baseline card shows the notebook's prob0). Only M, the total,
    the band and the text move."""
    detail = result.get("winloss_confidence_detail")
    if detail is None:
        return confidence_keys(None)
    return confidence_keys(confidence.at_probability(detail, win_probability))


def score_scenario(
    models: dict[str, Any],
    metadata: dict[str, Any],
    base_row: pd.Series,
    overrides: dict[str, Any] | None = None,
    fold_models: dict[str, list[xgb.Booster]] | None = None,
) -> dict[str, Any]:
    """Score one contract/product scenario.

    Returns the win probability, the product's own threshold, the resulting
    Win/Loss call, a margin-based confidence band, and how many features had
    to be imputed. When ``fold_models`` is given (see load_fold_models()),
    also returns the v3 package's win/loss model-confidence score and label
    (winloss_confidence / winloss_confidence_label) — omitted (None) otherwise,
    since the internal price-sensitivity/scenario-sweep callers below don't
    need it and it would be wasted work on every point of a sweep.
    """
    overrides = overrides or {}
    product_value = str(base_row[PRODUCT_COLUMN])
    key = product_key(product_value)
    model = models[key]

    features = build_feature_row(base_row, overrides, metadata)
    frame = feature_frame(features)

    probabilities = model.predict_proba(frame)[0]
    win_probability = float(probabilities[WIN_INDEX])
    threshold = threshold_for(product_value, metadata)
    margin = abs(win_probability - threshold)

    imputed = [name for name, value in features.items() if pd.isna(value)]

    result = {
        "product": product_value,
        "product_key": key,
        "features": features,
        "win_probability": win_probability,
        "loss_probability": float(probabilities[1 - WIN_INDEX]),
        "threshold": threshold,
        "predicted_outcome": "Win" if win_probability >= threshold else "Loss",
        "margin_to_threshold": margin,
        "confidence": confidence_band(margin),
        "imputed_features": imputed,
        "feature_completeness": 1.0 - len(imputed) / len(FEATURE_COLUMNS),
        # Nested under oof_metrics in the current metadata; the previous model
        # carried it flat on the product entry.
        "model_roc_auc": metadata["products"][key]["oof_metrics"]["roc_auc"],
    }

    if fold_models is not None and product_value in fold_models:
        result.update(
            winloss_confidence(
                product_value=product_value,
                features=features,
                win_probability=win_probability,
                threshold=threshold,
                fold_boosters=fold_models[product_value],
                spec=metadata["winloss_confidence"],
                provenance=scenario_provenance(base_row, features, overrides),
            )
        )
    else:
        result.update(confidence_keys(None))

    return result


def compute_shap_contributions(
    models: dict[str, Any],
    metadata: dict[str, Any],
    base_row: pd.Series,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Per-feature TreeSHAP contributions for the scored scenario.

    Uses XGBoost's own ``pred_contribs=True`` rather than the shap package.
    Two reasons:

    * shap 0.48 cannot read XGBoost 3.x models — it parses ``base_score`` with
      ``float()`` but XGBoost now serialises it as the JSON array ``'[5E-1]'``,
      raising ``ValueError: could not convert string to float: '[5E-1]'``.
    * These are the same exact TreeSHAP values. Verified by the additivity
      identity: ``sum(contributions) + bias == raw margin == logit(p)`` to
      within 1e-6 on all three models.

    Raises on failure rather than returning None, so the caller can surface the
    reason instead of silently rendering an empty panel.
    """
    overrides = overrides or {}
    key = product_key(str(base_row[PRODUCT_COLUMN]))
    model = models[key]

    features = build_feature_row(base_row, overrides, metadata)
    frame = feature_frame(features)

    # No "prep" step to run first: the v2 model is a bare XGBClassifier with
    # no preprocessing pipeline, so the raw feature frame goes straight to the
    # booster (missing values included — see load_models()).
    booster = model.get_booster()
    matrix = _dmatrix(frame)

    # Trailing column is the bias (base value), not a feature.
    contribs = np.asarray(booster.predict(matrix, pred_contribs=True))
    values = contribs[0][:-1]

    if values.shape[0] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"expected {len(FEATURE_COLUMNS)} contributions, got {values.shape[0]}"
        )

    contributions = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "label": [FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
            "value": [features[name] for name in FEATURE_COLUMNS],
            "shap": values,
        }
    )
    contributions["direction"] = np.where(
        contributions["shap"] >= 0, "Increases win", "Decreases win"
    )
    contributions["magnitude"] = contributions["shap"].abs()
    ordered = contributions.sort_values("magnitude", ascending=False).reset_index(drop=True)

    # The bias term the waterfall starts from: E[f(X)] in log-odds. Set after
    # sorting because .attrs does not survive every pandas operation.
    ordered.attrs["base_value"] = float(contribs[0][-1])
    return ordered


def price_sensitivity_curve(
    models: dict[str, Any],
    metadata: dict[str, Any],
    base_row: pd.Series,
    overrides: dict[str, Any] | None = None,
    points: int = 41,
) -> pd.DataFrame:
    """Sweep the dead net price and return the win-probability response.

    Bounds come from the 1st and 99th percentile of the product's observed
    dead net price, so the curve stays inside the range the model has seen.
    """
    overrides = dict(overrides or {})
    key = product_key(str(base_row[PRODUCT_COLUMN]))
    # lever_ranges is nested inside each product entry in the current metadata.
    bounds = metadata["products"][key]["lever_ranges"]
    low = float(bounds["dead_net_price_p01"])
    high = float(bounds["dead_net_price_p99"])

    rows = []
    for price in np.linspace(low, high, points):
        scenario_overrides = {**overrides, "dead_net_price": float(price)}
        result = score_scenario(models, metadata, base_row, scenario_overrides)
        rows.append(
            {
                "dead_net_price": float(price),
                "win_probability": result["win_probability"],
                "threshold": result["threshold"],
                "predicted_outcome": result["predicted_outcome"],
            }
        )
    return pd.DataFrame(rows)


def symmetric_price_sensitivity(
    models: dict[str, Any],
    metadata: dict[str, Any],
    base_row: pd.Series,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Modeled win-probability response to a fixed +/-10% price move off the
    scenario's own current price (the lever's overridden price if set, else
    the row's dead-net price):

        upward_price_risk_pp        = 100 x [P(current) - P(+10%)]
        downward_price_opportunity_pp = 100 x [P(-10%) - P(current)]

    Unlike price_increase_impact()'s smallest-tested-step search, this is
    always the fixed +/-10% pair — the primary business-facing sensitivity
    measure.
    """
    overrides = dict(overrides or {})
    current_price = float(overrides.get("dead_net_price", base_row[PRICE_BASE_COLUMN]))
    baseline = score_scenario(models, metadata, base_row, overrides)
    plus10 = score_scenario(
        models, metadata, base_row, {**overrides, "dead_net_price": current_price * 1.10}
    )
    minus10 = score_scenario(
        models, metadata, base_row, {**overrides, "dead_net_price": current_price * 0.90}
    )
    return {
        "baseline_win_probability": baseline["win_probability"],
        "plus10_win_probability": plus10["win_probability"],
        "minus10_win_probability": minus10["win_probability"],
        "upward_price_risk_pp": 100 * (baseline["win_probability"] - plus10["win_probability"]),
        "downward_price_opportunity_pp": 100 * (minus10["win_probability"] - baseline["win_probability"]),
    }


# Tested price increases, smallest first — price_increase_impact() reports the
# smallest one whose win-probability drop reaches the materiality threshold,
# rather than a single fixed step.
PRICE_INCREASE_STEPS_PCT: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
# Materiality threshold, in win-probability percentage points.
PRICE_SENSITIVITY_THRESHOLD_PP = 2.0


def price_increase_impact(
    models: dict[str, Any],
    metadata: dict[str, Any],
    base_row: pd.Series,
    overrides: dict[str, Any] | None = None,
    steps: tuple[float, ...] = PRICE_INCREASE_STEPS_PCT,
    threshold_pp: float = PRICE_SENSITIVITY_THRESHOLD_PP,
) -> dict[str, Any] | None:
    """Smallest tested price increase whose modeled win-probability drop
    reaches ``threshold_pp``, evaluated against the scenario's own current
    price (the lever's overridden price if set, else the row's dead-net price).

    Walks the increases smallest-first: reports the first increase that
    produces a meaningful drop rather than the largest one tested. Returns
    None when every tested increase stays under the threshold.
    """
    overrides = dict(overrides or {})
    current_price = float(overrides.get("dead_net_price", base_row[PRICE_BASE_COLUMN]))
    baseline = score_scenario(models, metadata, base_row, overrides)
    baseline_win_probability = baseline["win_probability"]

    for pct in steps:
        new_price = current_price * (1 + pct / 100)
        scenario = score_scenario(models, metadata, base_row, {**overrides, "dead_net_price": new_price})
        drop_pp = 100 * (baseline_win_probability - scenario["win_probability"])
        if drop_pp >= threshold_pp:
            return {
                "price_change_pct": pct,
                "current_price": current_price,
                "new_price": new_price,
                "baseline_win_probability": baseline_win_probability,
                "scenario_win_probability": scenario["win_probability"],
                "drop_pp": drop_pp,
            }
    return None
