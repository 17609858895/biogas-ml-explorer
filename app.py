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


def build_sample_row(
    med: pd.Series,
    numeric_features: list[str],
    reactor_id: str,
    reactor_type: str,
    values: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {col: float(med.get(col, 0.0)) for col in numeric_features}
    lag1 = values["biogas_lag1"]
    lag2 = values["biogas_lag2"]
    max_temp = values["daily_max_air_temp"]
    min_temp = values["daily_min_air_temp"]
    row.update(values)
    row.update(
        {
            "reactor_id": reactor_id,
            "reactor_type": reactor_type,
            "biogas_roll3": np.mean([lag1, lag2, values.get("biogas_roll3", lag1)]),
            "biogas_roll7": values.get("biogas_roll7", float(med.get("biogas_roll7", np.mean([lag1, lag2])))),
            "biogas_delta1": lag1 - lag2,
            "daily_temp_range": max_temp - min_temp,
        }
    )
    return row


def predict_frame(
    input_df: pd.DataFrame,
    model: Pipeline,
    clean_df: pd.DataFrame,
    numeric_features: list[str],
    all_features: list[str],
) -> pd.DataFrame:
    med = clean_df[numeric_features].median(numeric_only=True)
    rows = []
    for _, raw in input_df.iterrows():
        reactor_label = str(raw.get("reactor_id", raw.get("Reactor", "R1-FLEX")))
        reactor_id = raw_reactor(reactor_label)
        if reactor_id not in set(clean_df["reactor_id"]):
            reactor_id = clean_df["reactor_id"].mode().iat[0]
        reactor_type = raw.get("reactor_type")
        if pd.isna(reactor_type) or reactor_type is None:
            reactor_type = clean_df.loc[clean_df["reactor_id"] == reactor_id, "reactor_type"].mode().iat[0]
        values = {}
        for col in numeric_features:
            value = raw.get(col, med.get(col, 0.0))
            values[col] = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(med.get(col, 0.0)).iat[0])
        rows.append(build_sample_row(med, numeric_features, reactor_id, str(reactor_type), values))
    features = pd.DataFrame(rows)[all_features]
    out = input_df.copy()
    out["predicted_biogas_STP_mL"] = model.predict(features)
    return out


def template_dataframe(clean_df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    med = clean_df[numeric_features].median(numeric_only=True)
    return pd.DataFrame(
        [
            {
                "reactor_id": "R1-FLEX",
                "reactor_type": "FLEX",
                "manure_fed_kg": round(float(med.get("manure_fed_kg", 20.0)), 3),
                "water_kg": round(float(med.get("water_kg", 20.0)), 3),
                "air_temp_in_situ": round(float(med.get("air_temp_in_situ", 20.0)), 3),
                "biogas_lag1": round(float(med.get("biogas_lag1", 500.0)), 3),
                "biogas_lag2": round(float(med.get("biogas_lag2", 500.0)), 3),
                "biogas_roll3": round(float(med.get("biogas_roll3", 500.0)), 3),
                "biogas_roll7": round(float(med.get("biogas_roll7", 500.0)), 3),
                "days_since_prev": 1,
                "daily_mean_air_temp": round(float(med.get("daily_mean_air_temp", 18.0)), 3),
                "daily_max_air_temp": round(float(med.get("daily_max_air_temp", 24.0)), 3),
                "daily_min_air_temp": round(float(med.get("daily_min_air_temp", 14.0)), 3),
                "daily_solar": round(float(med.get("daily_solar", 140.0)), 3),
                "daily_precip": round(float(med.get("daily_precip", 0.0)), 3),
                "daily_atm_p": round(float(med.get("daily_atm_p", 81.7)), 3),
                "daily_vpd": round(float(med.get("daily_vpd", 0.3)), 3),
                "daily_wind": round(float(med.get("daily_wind", 0.8)), 3),
            }
        ]
    )


st.set_page_config(page_title="Biogas STP Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #24313d;
        --muted: #667785;
        --line: #dce7e2;
        --panel: #ffffff;
        --soft: #f6faf8;
        --soft-2: #eef7f3;
        --accent: #2f8f83;
        --accent-2: #d8efe8;
        --accent-3: #f7c873;
    }
    .stApp { background: linear-gradient(180deg, #fbfdfc 0%, #f3f8f6 100%); color: var(--ink); }
    .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1160px; }
    section[data-testid="stSidebar"] { background: #f8fbfa; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--ink); }
    .hero {
        background:
            radial-gradient(circle at 96% 16%, rgba(247, 200, 115, 0.22), transparent 28%),
            linear-gradient(135deg, #ffffff 0%, #f1faf6 100%);
        border: 1px solid var(--line);
        border-left: 6px solid var(--accent);
        border-radius: 12px;
        padding: 24px 28px;
        color: var(--ink);
        margin-bottom: 16px;
        box-shadow: 0 12px 28px rgba(36, 49, 61, 0.06);
    }
    .hero h1 { margin: 0 0 0.35rem 0; font-size: 2.0rem; letter-spacing: 0; line-height: 1.15; color: var(--ink); }
    .hero p { margin: 0; max-width: 840px; font-size: 1.01rem; line-height: 1.55; color: var(--muted); }
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 11px;
        padding: 6px;
        margin-bottom: 12px;
        box-shadow: 0 8px 22px rgba(36, 49, 61, 0.04);
    }
    button[data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--muted);
        padding: 8px 16px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--soft-2);
        color: var(--accent);
        font-weight: 750;
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 10px 24px rgba(36, 49, 61, 0.045);
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 98px;
        box-shadow: 0 8px 20px rgba(36, 49, 61, 0.04);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--accent); }
    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fcfa 100%);
        border: 1px solid var(--line);
        border-top: 4px solid var(--accent-3);
        border-radius: 12px;
        padding: 20px 22px 18px 22px;
        box-shadow: 0 12px 26px rgba(36, 49, 61, 0.055);
        margin-bottom: 16px;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.0rem;
        font-weight: 800;
        margin: 0.1rem 0 0.65rem 0;
    }
    .small-note { color: var(--muted); font-size: 0.92rem; line-height: 1.52; }
    label, .stCaptionContainer, div[data-testid="stMarkdownContainer"] p { color: var(--muted); }
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stFileUploader"] label {
        color: var(--ink);
        font-size: 1.06rem;
        font-weight: 720;
        line-height: 1.35;
        margin-bottom: 0.28rem;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        font-size: 1.02rem;
    }
    div[data-baseweb="input"] {
        border-radius: 8px;
        background: #fbfdfc;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 750;
        border: 1px solid #b8dcd2;
        background: #eef8f4;
        color: #206c63;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: #e4f3ee;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


table1, table2, filter_summary = load_tables()
model, clean_df, demo_metrics, numeric_features, categorical_features, all_features = train_demo_model()
med = clean_df[numeric_features].median(numeric_only=True)

st.markdown(
    """
    <div class="hero">
      <h1>Biogas STP Predictor</h1>
      <p>Enter reactor operation, weather, and recent biogas history to predict daily biogas production at STP.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model status")
    st.metric("Test R2", f"{demo_metrics['test_r2']:.3f}")
    st.metric("Test RMSE", f"{demo_metrics['rmse']:.1f} mL")
    st.metric("Training records", f"{int(demo_metrics['n_train'])}")
    st.metric("Test records", f"{int(demo_metrics['n_test'])}")
    st.divider()
    st.subheader("Auxiliary")
    st.dataframe(table2[["Model", "R²", "RMSE"]].head(5), use_container_width=True, hide_index=True)
    fig_path = FIG_DIR / "Fig03_model_comparison" / "Fig03_model_comparison.png"
    if fig_path.exists():
        with st.expander("Model comparison figure"):
            st.image(str(fig_path), use_column_width=True)

single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

with single_tab:
    st.markdown('<div class="section-title">Input parameters</div>', unsafe_allow_html=True)
    with st.form("single_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            reactor_display = st.selectbox("Reactor", sorted(clean_df["reactor_id"].map(rlab).unique()))
            reactor_id = raw_reactor(reactor_display)
            reactor_type = clean_df.loc[clean_df["reactor_id"] == reactor_id, "reactor_type"].mode().iat[0]
            manure = st.number_input("Manure fed (kg)", value=float(med.get("manure_fed_kg", 20.0)), min_value=0.0)
            water = st.number_input("Water (kg)", value=float(med.get("water_kg", 20.0)), min_value=0.0)
            local_temp = st.number_input("Local air temperature", value=float(med.get("air_temp_in_situ", 20.0)))
            lag1 = st.number_input("Biogas lag 1 (mL)", value=float(med.get("biogas_lag1", 500.0)), min_value=0.0)
            lag2 = st.number_input("Biogas lag 2 (mL)", value=float(med.get("biogas_lag2", 500.0)), min_value=0.0)
            roll7 = st.number_input("7-day historical average (mL)", value=float(med.get("biogas_roll7", 500.0)), min_value=0.0)
        with c2:
            mean_temp = st.number_input("Daily mean air temperature", value=float(med.get("daily_mean_air_temp", 18.0)))
            max_temp = st.number_input("Daily max air temperature", value=float(med.get("daily_max_air_temp", 24.0)))
            min_temp = st.number_input("Daily min air temperature", value=float(med.get("daily_min_air_temp", 14.0)))
            solar = st.number_input("Daily solar", value=float(med.get("daily_solar", 140.0)), min_value=0.0)
            precip = st.number_input("Daily precipitation", value=float(med.get("daily_precip", 0.0)), min_value=0.0)
            atm_p = st.number_input("Daily atmospheric pressure", value=float(med.get("daily_atm_p", 81.7)))
            vpd = st.number_input("Daily VPD", value=float(med.get("daily_vpd", 0.3)), min_value=0.0)
            wind = st.number_input("Daily wind", value=float(med.get("daily_wind", 0.8)), min_value=0.0)
        submitted = st.form_submit_button("Predict biogas production", use_container_width=True)

    values = {
        "manure_fed_kg": manure,
        "water_kg": water,
        "air_temp_in_situ": local_temp,
        "biogas_lag1": lag1,
        "biogas_lag2": lag2,
        "biogas_roll3": np.mean([lag1, lag2]),
        "biogas_roll7": roll7,
        "days_since_prev": 1.0,
        "daily_mean_air_temp": mean_temp,
        "daily_max_air_temp": max_temp,
        "daily_min_air_temp": min_temp,
        "daily_solar": solar,
        "daily_precip": precip,
        "daily_atm_p": atm_p,
        "daily_vpd": vpd,
        "daily_wind": wind,
    }
    sample_row = build_sample_row(med, numeric_features, reactor_id, reactor_type, values)
    sample = pd.DataFrame([sample_row])[all_features]
    prediction = float(model.predict(sample)[0])

    result_col, detail_col = st.columns([0.42, 0.58], gap="large")
    with result_col:
        st.markdown('<div class="section-title">Prediction output</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if submitted:
            st.success("Prediction completed.")
        st.metric("Predicted daily biogas at STP", f"{prediction:.1f} mL")
        st.caption("Prediction is generated from the cleaned ExtraTrees regression model.")
        st.markdown("</div>", unsafe_allow_html=True)
    with detail_col:
        st.markdown('<div class="section-title">Prediction record</div>', unsafe_allow_html=True)
        output = pd.DataFrame(
            [
                {
                    "reactor_id": reactor_display,
                    "reactor_type": reactor_type,
                    **{k: round(float(v), 4) for k, v in values.items()},
                    "predicted_biogas_STP_mL": prediction,
                }
            ]
        )
        st.dataframe(output, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction CSV",
            data=output.to_csv(index=False).encode("utf-8-sig"),
            file_name="biogas_single_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

with batch_tab:
    st.markdown('<div class="section-title">Batch prediction from CSV</div>', unsafe_allow_html=True)
    template = template_dataframe(clean_df, numeric_features)
    st.download_button(
        "Download input template",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="biogas_prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload a CSV file with one or more candidate operating conditions", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        result = predict_frame(batch, model, clean_df, numeric_features, all_features)
        st.success(f"Predicted {len(result)} rows.")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download batch predictions",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name="biogas_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Upload the template after editing values, or provide a compatible CSV with matching column names.")
