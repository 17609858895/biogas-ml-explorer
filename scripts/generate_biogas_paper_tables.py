"""Generate a minimal set of manuscript tables for the biogas ML paper."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
sys.path.insert(0, str(FIG_DIR))

from run_all_figures_refined import (  # noqa: E402
    REACTORS,
    TARGET,
    evaluate_models,
    load_and_prepare,
    rlab,
    rn,
)


def write_table(df: pd.DataFrame, stem: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(TABLE_DIR / f"{stem}.xlsx", index=False)
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    (TABLE_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table1_dataset_summary(df_raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reactor in REACTORS:
        raw = df_raw[df_raw["reactor_id"] == reactor]
        clean = df[df["reactor_id"] == reactor]
        reactor_type = clean["reactor_type_plot"].iloc[0] if len(clean) else raw["reactor_type"].iloc[0]
        rows.append(
            {
                "Reactor": rlab(reactor),
                "Type": reactor_type,
                "Raw records": len(raw),
                "Clean records": len(clean),
                "Date range": f"{clean['date'].min().date()} to {clean['date'].max().date()}",
                "Biogas mean (mL)": round(clean[TARGET].mean(), 1),
                "Biogas SD (mL)": round(clean[TARGET].std(), 1),
                "Biogas median (mL)": round(clean[TARGET].median(), 1),
                "Model variables available (%)": round(
                    100
                    - clean[
                        [
                            TARGET,
                            "manure_fed_kg",
                            "water_kg",
                            "air_temp_in_situ",
                            "daily_mean_air_temp",
                            "daily_solar",
                            "daily_precip",
                            "daily_vpd",
                            "daily_wind",
                        ]
                    ].isna().mean().mean()
                    * 100,
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def table2_model_performance(train: pd.DataFrame, test: pd.DataFrame, hist_f, oper_f, weat_f, cat_f, all_f) -> pd.DataFrame:
    performance, _, _, _, _ = evaluate_models(train, test, hist_f + oper_f + weat_f, cat_f, all_f)
    table = performance.reset_index().rename(columns={"model": "Model"})
    keep = ["Model", "R2", "RMSE", "MAE", "MAPE", "NSE", "TOPSIS"]
    table = table[keep].copy()
    table.insert(0, "Rank", range(1, len(table) + 1))
    for col in ["R2", "NSE", "TOPSIS"]:
        table[col] = table[col].round(4)
    for col in ["RMSE", "MAE", "MAPE"]:
        table[col] = table[col].round(2)
    table = table.rename(columns={"R2": "R²"})
    return table


def main() -> None:
    df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f = load_and_prepare()
    t1 = table1_dataset_summary(df_raw, df)
    t2 = table2_model_performance(train, test, hist_f, oper_f, weat_f, cat_f, all_f)
    write_table(t1, "Table1_dataset_reactor_summary")
    write_table(t2, "Table2_model_performance")
    readme = """# Manuscript tables

Only two main-text tables are recommended.

- Table 1: dataset and reactor summary.
- Table 2: concise test-set model performance and TOPSIS ranking.

Detailed Wilcoxon comparisons, feature-set ablation, SHAP attribution, and optimisation traces are already shown in figures and should not be repeated as main-text tables.
"""
    (TABLE_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
