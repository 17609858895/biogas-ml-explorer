"""Recompute Fig 5 R2 (train/test) using a RANDOM 80/20 split instead of date-based."""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = ROOT / "merged_train_df.csv"
ML_ERROR_EXCLUSION_PATH = ROOT / "tables" / "ml_error_removed_rows.csv"
TARGET = "y_biogas_STP"
RANDOM_STATE = 42
REACTOR_LABELS = {"RI-FLEX": "R1-FLEX"}


def rlab(name):
    return REACTOR_LABELS.get(name, name)


def row_keys(frame):
    return (
        frame["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + frame["reactor_id"].astype(str).map(rlab)
        + "|"
        + frame[TARGET].round(6).astype(str)
    )


def prepare():
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df_raw[TARGET].dropna()
    q1, q3 = y.quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    df = df_raw[df_raw[TARGET].notna() & (df_raw[TARGET] >= 0) & (df_raw[TARGET] <= upper)].copy()
    df = df.sort_values(["reactor_id", "date"]).reset_index(drop=True)
    g = df.groupby("reactor_id", group_keys=False)
    df["biogas_lag1"] = g[TARGET].shift(1)
    df["biogas_lag2"] = g[TARGET].shift(2)
    df["biogas_roll3"] = g[TARGET].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["biogas_roll7"] = g[TARGET].apply(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    df["biogas_delta1"] = df["biogas_lag1"] - df["biogas_lag2"]
    df["days_since_prev"] = g["date"].diff().dt.days
    if ML_ERROR_EXCLUSION_PATH.exists():
        exclusions = pd.read_csv(ML_ERROR_EXCLUSION_PATH)
        if "row_key" in exclusions.columns:
            df = df.loc[~row_keys(df).isin(set(exclusions["row_key"].astype(str)))].copy()
    df = df.sort_values(["date", "reactor_id"]).reset_index(drop=True)

    hist_f = ["biogas_lag1", "biogas_lag2", "biogas_roll3", "biogas_roll7", "biogas_delta1", "days_since_prev"]
    oper_f = ["manure_fed_kg", "water_kg", "air_temp_in_situ"]
    weat_f = ["daily_mean_air_temp", "daily_max_air_temp", "daily_min_air_temp", "daily_solar",
              "daily_precip", "daily_atm_p", "daily_vpd", "daily_wind", "daily_temp_range"]
    hist_f = [c for c in hist_f if c in df.columns]
    oper_f = [c for c in oper_f if c in df.columns]
    weat_f = [c for c in weat_f if c in df.columns]
    cat_f = ["reactor_id", "reactor_type"]
    num_f = hist_f + oper_f + weat_f
    all_f = num_f + cat_f
    return df, num_f, cat_f, all_f


def build_pipe(est, num_f, cat_f):
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_f),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]), cat_f),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("mdl", est)])


def models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=18, weights="uniform"),
        "SVR": SVR(C=10, epsilon=30, gamma="scale"),
        "RF": RandomForestRegressor(n_estimators=360, max_depth=8, min_samples_leaf=8,
                                    max_features=0.75, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=360, max_depth=6, min_samples_leaf=8,
                                          max_features=0.75, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=120, learning_rate=0.035, max_depth=2, subsample=0.70,
                                    colsample_bytree=0.70, min_child_weight=12, reg_lambda=12, reg_alpha=1,
                                    random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=120, learning_rate=0.035, max_depth=2, num_leaves=5,
                                      min_child_samples=25, subsample=0.70, colsample_bytree=0.70,
                                      reg_lambda=12, random_state=RANDOM_STATE, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(iterations=130, learning_rate=0.035, depth=2, l2_leaf_reg=15,
                                         random_strength=1.5, random_state=RANDOM_STATE, verbose=0),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.002, learning_rate_init=0.003,
                            early_stopping=True, max_iter=1500, random_state=RANDOM_STATE),
    }


def main():
    df, num_f, cat_f, all_f = prepare()
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    print(f"n_total={len(df)}  n_train={len(train)}  n_test={len(test)}\n")
    rows = []
    for name, est in models().items():
        pipe = build_pipe(clone(est), num_f, cat_f)
        pipe.fit(train[all_f], train[TARGET])
        tr = r2_score(train[TARGET], pipe.predict(train[all_f]))
        te = r2_score(test[TARGET], pipe.predict(test[all_f]))
        rows.append({"Model": name, "Train_R2": tr, "Test_R2": te})
    res = pd.DataFrame(rows)
    res = res.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(res.to_string(index=False))
    res.to_csv(BASE / "random_split_r2_results.csv", index=False)


if __name__ == "__main__":
    main()
