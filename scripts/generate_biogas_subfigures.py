"""
Export individual subfigures for every paper figure.

Use from the project root:
    python scripts/generate_biogas_subfigures.py

Use from Jupyter:
    %run scripts/generate_biogas_subfigures.py --only Fig03

The subfigures are exported from the same Matplotlib objects used for the full
paper figures, so fonts, colours, axes, legends and tick styling stay aligned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
sys.path.insert(0, str(FIGURES_DIR))

import run_all_figures_refined as mainfig  # noqa: E402
import run_figS1toS3_refined as suppfig  # noqa: E402


MAIN_MODULES = ["matplotlib", "seaborn", "sklearn", "xgboost", "lightgbm", "catboost", "shap"]
SUPPLEMENT_MODULES = MAIN_MODULES + ["optuna"]

PANEL_GROUPS = {
    "FigS1_data_audit": [["a"], ["b"], ["c"], ["d"]],
    "Fig01_target_distribution": [["a"], ["b"], ["c"], ["d"]],
    "Fig02_correlation": [["a"], ["b"], ["c"], ["d"]],
    "Fig03_model_comparison": [["a"], ["b"], ["c"], ["d"]],
    "Fig04_pred_vs_obs": [
        ["a", 0, 1],
        ["b", 2, 3],
        ["c", 4, 5],
        ["d", 6, 7],
        ["e", 8, 9],
        ["f", 10, 11],
        ["g", 12, 13],
        ["h", 14, 15],
        ["i", 16, 17],
    ],
    "Fig05_temporal_validation": [["a"], ["b"], ["c"], ["d"]],
    "Fig06_residual_reliability": [["a"], ["b"], ["c"], ["d"]],
    "Fig07_SHAP_importance": [["a"], ["b"], ["c"]],
    "Fig08_PDP_ALE_interaction": [
        ["a"],
        ["b"],
        ["c"],
        ["d"],
        ["e"],
        ["f"],
        ["g"],
        ["h"],
        ["i"],
        ["j"],
        ["k"],
        ["l"],
        ["m", 12, 13],
        ["n", 14, 15],
        ["o", 16],
        ["p", 17],
    ],
    "Fig09_feature_ablation": [["a"], ["b"], ["c"], ["d"]],
    "FigS2_learning_curves": [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["i"]],
    "FigS3_bayesian_optimisation": [["a"], ["b"], ["c"], ["d"]],
    "FigS4_residualized_weather": [["a"], ["b"], ["c"]],
}


FIGURE_FOLDERS = {
    "Fig01": "Fig01_target_distribution",
    "Fig02": "Fig02_correlation",
    "Fig03": "Fig03_model_comparison",
    "Fig04": "Fig04_pred_vs_obs",
    "Fig05": "Fig05_temporal_validation",
    "Fig06": "Fig06_residual_reliability",
    "Fig07": "Fig07_SHAP_importance",
    "Fig08": "Fig08_PDP_ALE_interaction",
    "Fig09": "Fig09_feature_ablation",
    "FigS1": "FigS1_data_audit",
    "FigS2": "FigS2_learning_curves",
    "FigS3": "FigS3_bayesian_optimisation",
    "FigS4": "FigS4_residualized_weather",
}


def check_environment(module_names: list[str]) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + f"\nCurrent Python: {sys.executable}\n"
            + r"Recommended: D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_subfigures.py"
        )


def normalise_selection(items: list[str] | None) -> set[str]:
    if not items:
        return set(FIGURE_FOLDERS)
    selected: set[str] = set()
    reverse = {v.lower(): k for k, v in FIGURE_FOLDERS.items()}
    for item in items:
        key = item.strip()
        low = key.lower()
        if key in FIGURE_FOLDERS:
            selected.add(key)
        elif low in reverse:
            selected.add(reverse[low])
        else:
            compact = low.replace(".", "").replace("_", "")
            matches = [k for k in FIGURE_FOLDERS if k.lower().replace(".", "").replace("_", "") == compact]
            if not matches:
                raise SystemExit(f"Unknown figure name: {item}")
            selected.add(matches[0])
    return selected


def bbox_for_axes(fig, axes, pad_inches=0.08):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    if not bboxes:
        return None
    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox.padded(pad_inches)


def resolve_panel_axes(fig, folder_name: str):
    axes = list(fig.axes)
    groups = PANEL_GROUPS.get(folder_name)
    if not groups:
        return []
    resolved = []
    for i, group in enumerate(groups):
        label = group[0]
        if len(group) == 1:
            idxs = [i]
        else:
            idxs = group[1:]
        panel_axes = [axes[idx] for idx in idxs if idx < len(axes)]
        if panel_axes:
            resolved.append((label, panel_axes))
    return resolved


def export_subfigures(fig, folder_name: str, stem: str) -> None:
    folder = FIGURES_DIR / folder_name
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    for old in list(subdir.glob("*.png")) + list(subdir.glob("*.pdf")):
        old.unlink()

    lines = [
        f"# {stem} 子图文件",
        "",
        "这些子图由同一套总图代码自动裁切导出，字体、配色、坐标轴刻度和总图保持一致。",
        "",
    ]

    for label, panel_axes in resolve_panel_axes(fig, folder_name):
        bbox = bbox_for_axes(fig, panel_axes)
        if bbox is None:
            continue
        out_stem = f"{stem}_{label}"
        fig.savefig(subdir / f"{out_stem}.png", dpi=600, bbox_inches=bbox, facecolor="white")
        try:
            fig.savefig(subdir / f"{out_stem}.pdf", bbox_inches=bbox, facecolor="white")
        except PermissionError:
            fig.savefig(subdir / f"{out_stem}_refined.pdf", bbox_inches=bbox, facecolor="white")
        lines.append(f"- `{out_stem}.png` / `{out_stem}.pdf`")

    (subdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_with_subfigures(fig: plt.Figure, folder_name: str, stem: str) -> None:
    folder = FIGURES_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    export_subfigures(fig, folder_name, stem)
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    pdf_path = folder / f"{stem}.pdf"
    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    except PermissionError:
        fig.savefig(folder / f"{stem}_refined.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  OK {folder_name}: full figure and subfigures")


def notebook_json(fig_key: str, folder_name: str) -> str:
    source = [
        "# 单独导出本图子图\n",
        "\n",
        "运行下面这个单元，会用当前论文统一配色、Arial 字体和高清设置重新生成本图及其子图。\n",
    ]
    code = [
        "%run ../../scripts/generate_biogas_subfigures.py --only "
        + fig_key
        + "\n",
    ]
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": source},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, indent=2)


def write_notebooks() -> None:
    for fig_key, folder_name in FIGURE_FOLDERS.items():
        folder = FIGURES_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{folder_name}_jupyter_code.ipynb").write_text(
            notebook_json(fig_key, folder_name), encoding="utf-8"
        )


def run_main_figures(selected: set[str]) -> None:
    mainfig.configure_style()
    df_raw, df, upper, train, calib, test, hist_f, oper_f, weat_f, cat_f, all_f = mainfig.load_and_prepare()
    numeric_f = hist_f + oper_f + weat_f

    need_models = any(k in selected for k in ["Fig03", "Fig04", "Fig05", "Fig06", "Fig09"])
    need_shap = any(k in selected for k in ["Fig07", "Fig08", "Fig09"])
    if need_models:
        performance, fold_df, fitted, preds_train, preds_test = mainfig.evaluate_models(
            train, test, numeric_f, cat_f, all_f
        )
        best_model = performance["TOPSIS"].idxmax()
    else:
        performance = fold_df = preds_train = preds_test = fitted = best_model = None

    if "FigS1" in selected:
        mainfig.fig1_data_audit(df_raw, df, upper)
    if "Fig01" in selected:
        mainfig.fig2_target_distribution(df_raw, df, upper)
    if "Fig02" in selected:
        mainfig.fig3_correlation(df, hist_f, oper_f, weat_f)
    if "Fig03" in selected:
        mainfig.fig4_model_comparison(performance, fold_df, preds_test, test)
    if "Fig04" in selected:
        mainfig.fig5_predicted_observed(performance, preds_train, preds_test, train, test)
    if "Fig05" in selected:
        mainfig.fig6_temporal_validation(test, calib, preds_test, fitted, best_model, all_f)
    if "Fig06" in selected:
        mainfig.fig7_residual_reliability(train, test, all_f, fitted, preds_test, best_model)

    if need_shap:
        shap_model, feats2, _tr2, te2, _xtr, xte, shap_values = mainfig.compute_shap_model(
            df, train, test, hist_f, oper_f, weat_f
        )
    else:
        shap_model = feats2 = te2 = xte = shap_values = None

    if "Fig07" in selected:
        mainfig.fig8_shap_importance(shap_model, feats2, te2, xte, shap_values)
    if "Fig08" in selected:
        mainfig.fig9_pdp_ale_interaction(shap_model, feats2, xte, shap_values)
    if "Fig09" in selected:
        mainfig.fig10_ablation(train, test, hist_f, oper_f, weat_f, cat_f, all_f, feats2, te2, shap_values)


def run_supp_figures(selected: set[str]) -> None:
    suppfig.configure_style()
    _df_raw, df, _upper, train, _calib, _test, hist_f, oper_f, weat_f, cat_f, all_f = suppfig.load_and_prepare()
    numeric_f = hist_f + oper_f + weat_f
    if "FigS2" in selected:
        suppfig.fig_s1_learning_curves(train, numeric_f, cat_f, all_f)
    if "FigS3" in selected:
        suppfig.fig_s2_bayesian(train, all_f, numeric_f, cat_f)
    if "FigS4" in selected:
        suppfig.fig_s3_causal(df, hist_f, oper_f, weat_f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate individual subfigures and Jupyter notebooks.")
    parser.add_argument("--only", nargs="+", help="Generate only selected figures, e.g. Fig03 FigS2.")
    parser.add_argument("--notebooks-only", action="store_true", help="Only write notebook entry files.")
    args = parser.parse_args()

    selected = normalise_selection(args.only)
    write_notebooks()
    if args.notebooks_only:
        print("Notebook entry files written.")
        return

    check_environment(SUPPLEMENT_MODULES if any(k.startswith("FigS") for k in selected) else MAIN_MODULES)
    mainfig.save = save_with_subfigures
    suppfig.save = save_with_subfigures

    main_selected = {k for k in selected if not k.startswith("FigS") or k == "FigS1"}
    supp_selected = {k for k in selected if k.startswith("FigS") and k != "FigS1"}
    if main_selected:
        run_main_figures(main_selected)
    if supp_selected:
        run_supp_figures(supp_selected)

    print("\nDone. Subfigures are in each figure folder under subfigures/.")


if __name__ == "__main__":
    main()
