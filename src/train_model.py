from __future__ import annotations

import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from src.generate_dummy_data import save_dummy_data
    from src.utils import (
        DATA_PATH,
        FEATURE_COLUMNS_PATH,
        MODEL_EXCLUDED_COLUMNS,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        OUTCOME_LABELS,
        OUTCOME_TO_ID,
        PROJECT_ROOT,
        TARGET_COLUMN,
        ensure_directories,
        today_iso,
        write_json,
    )
except ModuleNotFoundError:
    from generate_dummy_data import save_dummy_data
    from utils import (
        DATA_PATH,
        FEATURE_COLUMNS_PATH,
        MODEL_EXCLUDED_COLUMNS,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        OUTCOME_LABELS,
        OUTCOME_TO_ID,
        PROJECT_ROOT,
        TARGET_COLUMN,
        ensure_directories,
        today_iso,
        write_json,
    )


def _get_feature_groups(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns = [column for column in data.columns if column not in MODEL_EXCLUDED_COLUMNS]
    categorical_columns = data[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = [column for column in feature_columns if column not in categorical_columns]
    return categorical_columns, numerical_columns


def build_pipeline(categorical_columns: list[str], numerical_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("numerical", StandardScaler(), numerical_columns),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        num_class=len(OUTCOME_LABELS),
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    feature_columns_path=FEATURE_COLUMNS_PATH,
    model_metadata_path=MODEL_METADATA_PATH,
) -> dict:
    ensure_directories()
    if not data_path.exists():
        save_dummy_data(data_path)

    data = pd.read_csv(data_path)
    categorical_columns, numerical_columns = _get_feature_groups(data)
    feature_columns = categorical_columns + numerical_columns

    x = data[feature_columns]
    y = data[TARGET_COLUMN].map(OUTCOME_TO_ID)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(categorical_columns, numerical_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=OUTCOME_LABELS,
        output_dict=True,
        zero_division=0,
    )
    text_report = classification_report(
        y_test,
        predictions,
        target_names=OUTCOME_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions).tolist()

    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["classifier"].feature_importances_
    top_features = (
        pd.DataFrame({"feature": transformed_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    feature_payload = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "excluded_columns": MODEL_EXCLUDED_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_mapping": OUTCOME_TO_ID,
        "class_labels": OUTCOME_LABELS,
        "transformed_feature_count": len(transformed_feature_names),
    }
    write_json(feature_columns_path, feature_payload)

    metadata = {
        "model_version": "v0.1",
        "model_type": "XGBoost Classifier",
        "objective": "multi:softprob",
        "synthetic_poc_only": True,
        "disclaimer": "Synthetic PoC model for demonstration only. Not production-ready and not validated for real contracting decisions.",
        "training_sample_size": int(len(data)),
        "test_sample_size": int(len(x_test)),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix,
        "target_encoding": OUTCOME_TO_ID,
        "last_refreshed": today_iso(),
        "top_features": top_features,
    }
    write_json(model_metadata_path, metadata)

    print(f"Loaded data: {_display_path(data_path)} shape={data.shape}")
    print(f"Saved model: {_display_path(model_path)}")
    print(f"Saved feature metadata: {_display_path(feature_columns_path)}")
    print(f"Saved model metadata: {_display_path(model_metadata_path)}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Classification report:")
    print(text_report)
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=OUTCOME_LABELS, columns=OUTCOME_LABELS))
    print("Top model features:")
    for item in top_features[:10]:
        print(f"- {item['feature']}: {item['importance']:.4f}")

    return metadata


def _display_path(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    train_model()
