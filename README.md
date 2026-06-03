# Biogas STP Predictor

Streamlit web UI for predicting daily biogas production at STP from anaerobic-digestion operating conditions, weather, and recent biogas history.

## What The App Does

- Accepts single-sample inputs in a form.
- Predicts `y_biogas_STP` in mL using a cleaned ExtraTrees regression model.
- Supports CSV batch prediction.
- Provides a downloadable CSV for single and batch predictions.
- Keeps model diagnostics and figures in the sidebar as auxiliary context only.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads:

- `merged_train_df.csv`
- `tables/*.csv`
- `figures/Fig01_*` to `figures/Fig09_*`
- `figures/FigS1_*` to `figures/FigS4_*`

## Batch Input Columns

Recommended CSV columns:

```text
reactor_id, reactor_type, manure_fed_kg, water_kg, air_temp_in_situ,
biogas_lag1, biogas_lag2, biogas_roll3, biogas_roll7, days_since_prev,
daily_mean_air_temp, daily_max_air_temp, daily_min_air_temp, daily_solar,
daily_precip, daily_atm_p, daily_vpd, daily_wind
```

The app also provides a downloadable input template.

## Streamlit Cloud

Use `app.py` as the entry point. The app retrains a lightweight ExtraTrees model from the cleaned CSV data at startup.
