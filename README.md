# Biogas Machine Learning Explorer

Streamlit web UI for the anaerobic digestion machine-learning results.

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

## Streamlit Cloud

Use `app.py` as the entry point. The app retrains a lightweight ExtraTrees demo model from the cleaned CSV data and displays the manuscript figures and tables.
