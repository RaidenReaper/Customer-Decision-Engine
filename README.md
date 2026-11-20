# Customer Response Prediction Engine (demo)

A Streamlit demo of win/loss simulation for vaccine contracting. Pick a health
system and brands, adjust the commercial levers, and see the predicted win
probability per brand with the drivers behind it.

**All data in this repo is synthetic.** Accounts, IDs, addresses and contracts
are fictional, and the brand names (Viresta, Immunexa, Noventra) are
hypothetical.

## Tabs

- **Simulation**: levers (dead net price, share of wallet, partnership
  advocacy, engagement, negotiation stance, clinical focus) re-score the
  scenario against one XGBoost model per brand, each with its own decision
  threshold. It shows the win probability, a SHAP driver breakdown, a narrative
  summary, a confidence summary and a downloadable PDF report.
- **Customer Deep Dive**: KPI cards, account signals grouped by driver
  category, and trend charts.

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On share.streamlit.io choose **Create app**, then pick this repo, branch
   `main` and file `app.py`.
3. Under **Advanced settings**, set Python to **3.12**.
4. Deploy. No secrets are needed.

## Layout

```
app.py                       the Streamlit app
src/utils.py                 schema, feature definitions, loaders
src/scoring.py               model loading, scoring, TreeSHAP contributions
src/confidence.py            win/loss confidence
src/narrative_*.py           narrative summaries
src/report.py                PDF reports
landing/                     optional landing pages (?landing=1..5)
data/*.parquet               synthetic data
model/                       one XGBoost model per brand + fold models + metadata
scripts/make_dummy_data.py   regenerates the synthetic data
```

## Options

- Sign-in (Microsoft Entra) is off. Set `APP_REQUIRE_AUTH=1` and provide an
  `[auth]` block in `.streamlit/secrets.toml` to switch it on.
- `python scripts/make_dummy_data.py` draws a fresh synthetic data set from the
  current one.
