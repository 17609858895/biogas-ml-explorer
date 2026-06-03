"""Create a reproducible deletion list for high ML-error biogas records."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(FIG_DIR))

from run_all_figures_refined import (  # noqa: E402
    TARGET,
    build_pipe,
    load_and_prepare,
    model_library,
    rlab,
)

warnings.filterwarnings("ignore")

KEEP_QUANTILE = 0.65
FILTER_MODELS = ["Ridge", "KNN", "SVR", "RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]


def add_row_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_key"] = (
        out["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["reactor_id"].astype(str).map(rlab)
        + "|"
        + out[TARGET].round(6).astype(str)
    )
    return out


def compute_ml_error_scores(df: pd.DataFrame, numeric_f, cat_f, all_f) -> pd.DataFrame:
    models = {name: est for name, est in model_library().items() if name in FILTER_MODELS}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    residuals = pd.DataFrame(index=df.index)

    for name, estimator in models.items():
        pred = np.full(len(df), np.nan)
        for train_idx, val_idx in kfold.split(df):
            pipe = build_pipe(clone(estimator), numeric_f, cat_f)
            pipe.fit(df.iloc[train_idx][all_f], df.iloc[train_idx][TARGET])
            pred[val_idx] = pipe.predict(df.iloc[val_idx][all_f])
        residuals[name] = np.abs(df[TARGET].to_numpy() - pred)
        print(
            f"{name:10s} OOF R2={r2_score(df[TARGET], pred):.3f} "
            f"RMSE={math.sqrt(mean_squared_error(df[TARGET], pred)):.1f}"
        )

    scored = add_row_key(df)
    scored["ml_error_score"] = residuals.median(axis=1)
    scored["ml_error_q"] = scored["ml_error_score"].rank(pct=True, method="average")
    return scored


def evaluate_temporal(scored: pd.DataFrame, keep_mask: pd.Series, numeric_f, cat_f, all_f) -> pd.DataFrame:
    clean = scored.loc[keep_mask].copy()
    dates = np.array(sorted(clean["date"].unique()))
    test_cut = int(len(dates) * 0.8)
    train = clean[clean["date"].isin(set(dates[:test_cut]))]
    test = clean[clean["date"].isin(set(dates[test_cut:]))]

    rows = []
    for name, estimator in model_library().items():
        pipe = build_pipe(clone(estimator), numeric_f, cat_f)
        pipe.fit(train[all_f], train[TARGET])
        train_pred = pipe.predict(train[all_f])
        test_pred = pipe.predict(test[all_f])
        train_r2 = r2_score(train[TARGET], train_pred)
        test_r2 = r2_score(test[TARGET], test_pred)
        rows.append(
            {
                "Model": name,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Gap": train_r2 - test_r2,
                "RMSE": math.sqrt(mean_squared_error(test[TARGET], test_pred)),
            }
        )
    return pd.DataFrame(rows).sort_values("Test_R2", ascending=False)


def main() -> None:
    df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f = load_and_prepare(
        apply_ml_error_filter=False
    )
    numeric_f = hist_f + oper_f + weat_f
    scored = compute_ml_error_scores(df, numeric_f, cat_f, all_f)
    cutoff = scored["ml_error_score"].quantile(KEEP_QUANTILE)
    keep_mask = scored["ml_error_score"] <= cutoff
    removed = scored.loc[~keep_mask].sort_values("ml_error_score", ascending=False).copy()
    removed["reactor_id"] = removed["reactor_id"].map(rlab)

    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    removed_cols = [
        "row_key",
        "date",
        "reactor_id",
        "reactor_type",
        TARGET,
        "ml_error_score",
        "ml_error_q",
    ]
    removed[removed_cols].to_csv(out_dir / "ml_error_removed_rows.csv", index=False, encoding="utf-8-sig")

    diagnostics = evaluate_temporal(scored, keep_mask, numeric_f, cat_f, all_f)
    diagnostics.to_csv(out_dir / "ml_error_filter_model_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "Raw records after 3×IQR": len(scored),
                "Kept records": int(keep_mask.sum()),
                "Removed records": int((~keep_mask).sum()),
                "Keep quantile": KEEP_QUANTILE,
                "ML error cutoff": cutoff,
            }
        ]
    )
    summary.to_csv(out_dir / "ml_error_filter_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nRemoved {len(removed)} high-error records; kept {int(keep_mask.sum())}.")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote deletion list to {out_dir / 'ml_error_removed_rows.csv'}")


if __name__ == "__main__":
    main()
