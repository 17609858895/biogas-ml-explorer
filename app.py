from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "merged_train_df.csv"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
TARGET = "y_biogas_STP"
REACTOR_LABELS = {"RI-FLEX": "R1-FLEX"}
REACTOR_VALUES = {v: k for k, v in REACTOR_LABELS.items()}


def rlab(name: str) -> str:
    return REACTOR_LABELS.get(name, name)


def raw_reactor(label: str) -> str:
    return REACTOR_VALUES.get(label, label)


def row_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + frame["reactor_id"].astype(str).map(rlab)
        + "|"
        + frame[TARGET].round(6).astype(str)
    )


@st.cache_data(show_spinner=False)
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table1 = pd.read_csv(TABLE_DIR / "Table1_dataset_reactor_summary.csv")
    table2 = pd.read_csv(TABLE_DIR / "Table2_model_performance.csv")
    summary = pd.read_csv(TABLE_DIR / "ml_error_filter_summary.csv")
    return table1, table2, summary


@st.cache_data(show_spinner=False)
def prepare_data() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df_raw[TARGET].dropna()
    q1, q3 = y.quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    df = df_raw[df_raw[TARGET].notna() & (df_raw[TARGET] >= 0) & (df_raw[TARGET] <= upper)].copy()
    df = df.sort_values(["reactor_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("reactor_id", group_keys=False)
    df["biogas_lag1"] = grouped[TARGET].shift(1)
    df["biogas_lag2"] = grouped[TARGET].shift(2)
    df["biogas_roll3"] = grouped[TARGET].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["biogas_roll7"] = grouped[TARGET].apply(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    df["biogas_delta1"] = df["biogas_lag1"] - df["biogas_lag2"]
    df["days_since_prev"] = grouped["date"].diff().dt.days

    exclusions = TABLE_DIR / "ml_error_removed_rows.csv"
    if exclusions.exists():
        removed = pd.read_csv(exclusions)
        if "row_key" in removed.columns:
            df = df.loc[~row_keys(df).isin(set(removed["row_key"].astype(str)))].copy()

    df = df.sort_values(["date", "reactor_id"]).reset_index(drop=True)
    hist = ["biogas_lag1", "biogas_lag2", "biogas_roll3", "biogas_roll7", "biogas_delta1", "days_since_prev"]
    oper = ["manure_fed_kg", "water_kg", "air_temp_in_situ"]
    weather = [
        "daily_mean_air_temp",
        "daily_max_air_temp",
        "daily_min_air_temp",
        "daily_solar",
        "daily_precip",
        "daily_atm_p",
        "daily_vpd",
        "daily_wind",
        "daily_temp_range",
    ]
    numeric = [c for c in hist + oper + weather if c in df.columns]
    categorical = ["reactor_id", "reactor_type"]
    return df, numeric, categorical, numeric + categorical


def build_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical),
        ]
    )
    model = ExtraTreesRegressor(
        n_estimators=360,
        max_depth=6,
        min_samples_leaf=8,
        max_features=0.75,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_demo_model() -> tuple[Pipeline, pd.DataFrame, dict[str, float], list[str], list[str], list[str]]:
    df, numeric, categorical, features = prepare_data()
    dates = np.array(sorted(df["date"].unique()))
    test_cut = int(len(dates) * 0.8)
    train = df[df["date"].isin(set(dates[:test_cut]))].copy()
    test = df[df["date"].isin(set(dates[test_cut:]))].copy()
    pipe = build_model(numeric, categorical)
    pipe.fit(train[features], train[TARGET])
    pred_train = pipe.predict(train[features])
    pred_test = pipe.predict(test[features])
    metrics = {
        "train_r2": r2_score(train[TARGET], pred_train),
        "test_r2": r2_score(test[TARGET], pred_test),
        "rmse": math.sqrt(mean_squared_error(test[TARGET], pred_test)),
        "mae": mean_absolute_error(test[TARGET], pred_test),
        "n_train": float(len(train)),
        "n_test": float(len(test)),
    }
    final_pipe = build_model(numeric, categorical)
    final_pipe.fit(df[features], df[TARGET])
    return final_pipe, df, metrics, numeric, categorical, features


def show_metric_cards(metrics: dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R2", f"{metrics['train_r2']:.3f}")
    c2.metric("Test R2", f"{metrics['test_r2']:.3f}")
    c3.metric("Test RMSE", f"{metrics['rmse']:.1f} mL")
    c4.metric("Test MAE", f"{metrics['mae']:.1f} mL")


def image_path(folder: str, stem: str) -> Path:
    return FIG_DIR / folder / f"{stem}.png"


st.set_page_config(page_title="Biogas ML Explorer", page_icon=":bar_chart:", layout="wide")

st.title("Biogas Machine Learning Explorer")
st.caption("Farm-scale anaerobic digestion data, cleaned model diagnostics, and a lightweight prediction demo.")

table1, table2, filter_summary = load_tables()
model, clean_df, demo_metrics, numeric_features, categorical_features, all_features = train_demo_model()

tab_overview, tab_figures, tab_predict, tab_data = st.tabs(["Overview", "Figures", "Predictor", "Data"])

with tab_overview:
    st.subheader("Cleaned modelling dataset")
    s = filter_summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("After 3xIQR", int(s.iloc[0]))
    c2.metric("Kept records", int(s["Kept records"]))
    c3.metric("Removed records", int(s["Removed records"]))
    c4.metric("CV-error cutoff", f"{s['ML error cutoff']:.1f}")

    st.subheader("ExtraTrees demo model")
    show_metric_cards(demo_metrics)
    st.caption("The demo model is fitted from the same cleaned records used in the manuscript tables.")

    st.subheader("Published model comparison")
    st.dataframe(table2, use_container_width=True, hide_index=True)

with tab_figures:
    st.subheader("Main figures")
    figure_options = {
        "Fig. 1 Target distribution": ("Fig01_target_distribution", "Fig01_target_distribution"),
        "Fig. 2 Correlation": ("Fig02_correlation", "Fig02_correlation"),
        "Fig. 3 Model comparison": ("Fig03_model_comparison", "Fig03_model_comparison"),
        "Fig. 4 Predicted vs observed": ("Fig04_pred_vs_obs", "Fig04_pred_vs_obs"),
        "Fig. 5 Temporal validation": ("Fig05_temporal_validation", "Fig05_temporal_validation"),
        "Fig. 6 Residual diagnostics": ("Fig06_residual_reliability", "Fig06_residual_reliability"),
        "Fig. 7 SHAP importance": ("Fig07_SHAP_importance", "Fig07_SHAP_importance"),
        "Fig. 8 PDP/ALE interaction": ("Fig08_PDP_ALE_interaction", "Fig08_PDP_ALE_interaction"),
        "Fig. 9 Feature ablation": ("Fig09_feature_ablation", "Fig09_feature_ablation"),
        "Fig. S1 Data audit": ("FigS1_data_audit", "FigS1_data_audit"),
    }
    selected = st.selectbox("Choose a figure", list(figure_options))
    folder, stem = figure_options[selected]
    path = image_path(folder, stem)
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Missing figure file: {path}")

with tab_predict:
    st.subheader("Single-sample prediction")
    st.caption("Use this as an exploratory model demo, not as a calibrated operational controller.")
    med = clean_df[numeric_features].median(numeric_only=True)
    left, right = st.columns(2)

    with left:
        reactor_display = st.selectbox("Reactor", sorted(clean_df["reactor_id"].map(rlab).unique()))
        reactor_id = raw_reactor(reactor_display)
        reactor_type = clean_df.loc[clean_df["reactor_id"] == reactor_id, "reactor_type"].mode().iat[0]
        manure = st.number_input("Manure fed (kg)", value=float(med.get("manure_fed_kg", 20.0)), min_value=0.0)
        water = st.number_input("Water (kg)", value=float(med.get("water_kg", 20.0)), min_value=0.0)
        local_temp = st.number_input("Local air temperature", value=float(med.get("air_temp_in_situ", 20.0)))
        lag1 = st.number_input("Biogas lag 1 (mL)", value=float(med.get("biogas_lag1", 500.0)), min_value=0.0)
        lag2 = st.number_input("Biogas lag 2 (mL)", value=float(med.get("biogas_lag2", 500.0)), min_value=0.0)

    with right:
        mean_temp = st.number_input("Daily mean air temperature", value=float(med.get("daily_mean_air_temp", 18.0)))
        max_temp = st.number_input("Daily max air temperature", value=float(med.get("daily_max_air_temp", 24.0)))
        min_temp = st.number_input("Daily min air temperature", value=float(med.get("daily_min_air_temp", 14.0)))
        solar = st.number_input("Daily solar", value=float(med.get("daily_solar", 140.0)), min_value=0.0)
        precip = st.number_input("Daily precipitation", value=float(med.get("daily_precip", 0.0)), min_value=0.0)
        wind = st.number_input("Daily wind", value=float(med.get("daily_wind", 0.8)), min_value=0.0)

    row = {col: float(med.get(col, 0.0)) for col in numeric_features}
    row.update(
        {
            "reactor_id": reactor_id,
            "reactor_type": reactor_type,
            "manure_fed_kg": manure,
            "water_kg": water,
            "air_temp_in_situ": local_temp,
            "biogas_lag1": lag1,
            "biogas_lag2": lag2,
            "biogas_roll3": np.mean([lag1, lag2]),
            "biogas_roll7": float(med.get("biogas_roll7", np.mean([lag1, lag2]))),
            "biogas_delta1": lag1 - lag2,
            "days_since_prev": 1.0,
            "daily_mean_air_temp": mean_temp,
            "daily_max_air_temp": max_temp,
            "daily_min_air_temp": min_temp,
            "daily_solar": solar,
            "daily_precip": precip,
            "daily_wind": wind,
            "daily_temp_range": max_temp - min_temp,
        }
    )
    sample = pd.DataFrame([row])[all_features]
    pred = float(model.predict(sample)[0])
    st.metric("Predicted daily biogas at STP", f"{pred:.1f} mL")

with tab_data:
    st.subheader("Reactor summary")
    st.dataframe(table1, use_container_width=True, hide_index=True)
    st.subheader("Cleaned modelling records")
    preview = clean_df.copy()
    preview["reactor_id"] = preview["reactor_id"].map(rlab)
    st.dataframe(preview[["date", "reactor_id", "reactor_type", TARGET] + numeric_features[:6]].head(200), use_container_width=True)
