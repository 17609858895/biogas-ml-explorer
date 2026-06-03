"""All 9 models, Bayesian-optimized (BayesSearchCV, n_iter=32, cv=5) like RF.ipynb,
random 80/20 split, StandardScaler-style preprocessing pipeline."""
import os, math, random, warnings
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
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

os.environ["PYTHONHASHSEED"] = "1"
np.random.seed(1); random.seed(1)
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = ROOT / "merged_train_df.csv"
ML_ERROR_EXCLUSION_PATH = ROOT / "tables" / "ml_error_removed_rows.csv"
TARGET = "y_biogas_STP"
RS = 1
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
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df[TARGET].dropna()
    q1, q3 = y.quantile([0.25, 0.75]); upper = q3 + 3 * (q3 - q1)
    df = df[df[TARGET].notna() & (df[TARGET] >= 0) & (df[TARGET] <= upper)].copy()
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
    hist = ["biogas_lag1","biogas_lag2","biogas_roll3","biogas_roll7","biogas_delta1","days_since_prev"]
    oper = ["manure_fed_kg","water_kg","air_temp_in_situ"]
    weat = ["daily_mean_air_temp","daily_max_air_temp","daily_min_air_temp","daily_solar",
            "daily_precip","daily_atm_p","daily_vpd","daily_wind","daily_temp_range"]
    num = [c for c in hist+oper+weat if c in df.columns]
    cat = ["reactor_id","reactor_type"]
    return df, num, cat, num+cat


def pipe(est, num, cat):
    try: enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", enc)]), cat),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("mdl", est)])


def specs():
    p = "mdl__"
    return {
        "Ridge": (Ridge(random_state=None) if False else Ridge(), {
            p+"alpha": Real(1e-3, 100, prior="log-uniform")}),
        "KNN": (KNeighborsRegressor(), {
            p+"n_neighbors": Integer(3, 30), p+"weights": Categorical(["uniform","distance"]),
            p+"p": Integer(1, 2)}),
        "SVR": (SVR(), {
            p+"C": Real(0.1, 200, prior="log-uniform"), p+"epsilon": Real(0.1, 50, prior="uniform"),
            p+"gamma": Categorical(["scale","auto"])}),
        "RF": (RandomForestRegressor(random_state=RS, n_jobs=-1), {
            p+"n_estimators": Integer(50, 200), p+"max_depth": Integer(3, 20),
            p+"min_samples_split": Integer(2, 10), p+"min_samples_leaf": Integer(1, 10),
            p+"max_features": Real(0.1, 1.0, prior="uniform")}),
        "ExtraTrees": (ExtraTreesRegressor(random_state=RS, n_jobs=-1), {
            p+"n_estimators": Integer(50, 200), p+"max_depth": Integer(3, 20),
            p+"min_samples_split": Integer(2, 10), p+"min_samples_leaf": Integer(1, 10),
            p+"max_features": Real(0.1, 1.0, prior="uniform")}),
        "XGBoost": (xgb.XGBRegressor(random_state=RS, verbosity=0), {
            p+"n_estimators": Integer(50, 300), p+"learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            p+"max_depth": Integer(2, 8), p+"subsample": Real(0.5, 1.0),
            p+"colsample_bytree": Real(0.5, 1.0), p+"min_child_weight": Integer(1, 12),
            p+"reg_lambda": Real(0.1, 15, prior="log-uniform")}),
        "LightGBM": (lgb.LGBMRegressor(random_state=RS, verbose=-1), {
            p+"n_estimators": Integer(50, 300), p+"learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            p+"max_depth": Integer(2, 8), p+"num_leaves": Integer(5, 50),
            p+"min_child_samples": Integer(5, 40), p+"subsample": Real(0.5, 1.0),
            p+"colsample_bytree": Real(0.5, 1.0), p+"reg_lambda": Real(0.1, 15, prior="log-uniform")}),
        "CatBoost": (cb.CatBoostRegressor(random_state=RS, verbose=0), {
            p+"iterations": Integer(50, 300), p+"learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            p+"depth": Integer(2, 8), p+"l2_leaf_reg": Real(1, 20, prior="log-uniform")}),
        "MLP": (MLPRegressor(max_iter=1500, early_stopping=True, random_state=RS), {
            p+"hidden_layer_sizes": Categorical([(64,), (64,32), (128,64), (100,)]),
            p+"alpha": Real(1e-4, 1e-1, prior="log-uniform"),
            p+"learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform")}),
    }


def run_one(name, df, num, cat, allf, tr, te):
    est, space = specs()[name]
    pl = pipe(clone(est), num, cat)
    opt = BayesSearchCV(pl, space, n_iter=32, scoring="neg_mean_squared_error",
                        cv=5, random_state=RS, n_jobs=-1)
    opt.fit(tr[allf], tr[TARGET])
    best = opt.best_estimator_
    r2tr = r2_score(tr[TARGET], best.predict(tr[allf]))
    r2te = r2_score(te[TARGET], best.predict(te[allf]))
    return r2tr, r2te


def main():
    import sys, json
    df, num, cat, allf = prepare()
    tr, te = train_test_split(df, test_size=0.2, random_state=RS, shuffle=True)
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(specs().keys())
    out = BASE / "random_split_bayes_r2_results.csv"
    existing = {}
    if out.exists():
        for _, r in pd.read_csv(out).iterrows():
            existing[r["Model"]] = (r["Train_R2"], r["Test_R2"])
    for name in targets:
        r2tr, r2te = run_one(name, df, num, cat, allf, tr, te)
        existing[name] = (r2tr, r2te)
        print(f"DONE {name}: train={r2tr:.3f} test={r2te:.3f}", flush=True)
    rows = []
    for k, v in existing.items():
        rows.append({"Model": k, "Train_R2": v[0], "Test_R2": v[1]})
    res = pd.DataFrame(rows)
    res = res.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)


if __name__ == "__main__":
    main()
