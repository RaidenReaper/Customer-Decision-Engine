from __future__ import annotations

import joblib

from src.generate_dummy_data import save_dummy_data
from src.scoring import build_input_dataframe
from src.train_model import train_model


def test_training_pipeline_creates_artifacts(tmp_path):
    data_path = tmp_path / "data" / "dummy_contracting_data.csv"
    model_path = tmp_path / "model" / "xgb_customer_response_model.pkl"
    feature_path = tmp_path / "model" / "feature_columns.json"
    metadata_path = tmp_path / "model" / "model_metadata.json"

    data = save_dummy_data(data_path, rows=500, seed=42)
    train_model(data_path, model_path, feature_path, metadata_path)

    assert model_path.exists()
    assert feature_path.exists()
    assert metadata_path.exists()

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    feature_metadata = __import__("json").load(feature_path.open("r", encoding="utf-8"))
    input_df = build_input_dataframe(data.iloc[0].to_dict(), feature_metadata)
    probabilities = model.predict_proba(input_df)[0]

    assert len(probabilities) == 3
    assert abs(float(probabilities.sum()) - 1.0) < 0.001
