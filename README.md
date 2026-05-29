# Customer Decision Engine

An end-to-end synthetic proof-of-concept for pharma and vaccine contracting scenario decisioning. The Streamlit app lets business users select customer context, adjust proposed commercial terms, and simulate customer response probabilities for `Accepted`, `Negotiated`, and `Rejected`.

## What It Does

- Generates 500 synthetic vaccine contracting scenarios.
- Trains an XGBoost multi-class classifier in an sklearn Pipeline.
- Scores proposed pricing scenarios in real time.
- Shows probabilities, recommendation, confidence band, driver cards, watchouts, and a narrative summary.
- Saves simulated scenarios in session state for comparison.

## Project Structure

```text
customer_decision_engine/
+-- data/
|   +-- dummy_contracting_data.csv
+-- model/
|   +-- xgb_customer_response_model.pkl
|   +-- feature_columns.json
|   +-- model_metadata.json
+-- src/
|   +-- generate_dummy_data.py
|   +-- train_model.py
|   +-- scoring.py
|   +-- utils.py
+-- tests/
|   +-- test_data_generation.py
|   +-- test_training_pipeline.py
|   +-- test_scoring.py
|   +-- test_app_smoke.py
|   +-- conftest.py
+-- .streamlit/
|   +-- config.toml
+-- app.py
+-- requirements.txt
+-- pytest.ini
+-- README.md
```

## Setup

```powershell
pip install -r requirements.txt
```

## Generate Data

```powershell
python src/generate_dummy_data.py
```

Creates `data/dummy_contracting_data.csv` with 500 synthetic deal scenarios.

## Train Model

```powershell
python src/train_model.py
```

Saves:

- `model/xgb_customer_response_model.pkl`
- `model/feature_columns.json`
- `model/model_metadata.json`

## Run App

```powershell
streamlit run app.py
```

The app uses a light theme, scenario context filters, current-vs-proposed pricing levers, compact result cards, a stacked probability bar, driver cards, rationale expander, and saved scenario table.

## Deploy On Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Confirm these files are committed:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
   - `src/`
   - `data/dummy_contracting_data.csv`
   - `model/xgb_customer_response_model.pkl`
   - `model/feature_columns.json`
   - `model/model_metadata.json`
3. Go to Streamlit Community Cloud and create a new app from the GitHub repo.
4. Set the main file path to:

```text
app.py
```

5. Open **Advanced settings** and select:

```text
Python 3.11
```

6. Deploy the app.

The app loads the committed synthetic data and trained model artifacts directly, so no secrets or database setup are required for this PoC.

Important: the saved model was trained with the dependency versions pinned in `requirements.txt`. If Streamlit Cloud uses Python 3.14, packages such as `pandas==2.1.4` and `scikit-learn==1.4.2` can fail to install from source. If the app was already created with the wrong Python version, delete the app and redeploy it with Python 3.11 selected in Advanced settings.

## Run Tests

```powershell
pytest
```

Optional coverage:

```powershell
pytest --cov=src tests/
```

Testing covers data generation, required schema, value ranges, outcome balance, model artifact creation, prediction probabilities, scoring output, driver categories, narrative generation, and app import smoke behavior.

## Example Workflow

1. Select customer segment, customer, contract, season, round, product scope, and brand scope.
2. Edit proposed net price, rebate, reimbursement, volume, share, cancellation, and returnability.
3. Review compact scenario watchouts.
4. Click `Run Simulation`.
5. Review probabilities, recommended outcome, confidence, drivers, and narrative summary.
6. Click `Save Scenario` to keep the result in the session table.

## Disclaimer

This project uses synthetic dummy data only. It is a proof-of-concept for demonstration and workflow design, not a production model and not validated for real contracting, pricing, legal, medical, or financial decisions.
