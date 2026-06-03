"""
Unified figure-generation entry point for the biogas ML paper.

Use from the project root:
    python scripts/generate_biogas_paper_figures.py

Use from Jupyter:
    %run scripts/generate_biogas_paper_figures.py --main-only

The actual figure code lives in figures/, where each main figure also has a
dedicated notebook. This wrapper keeps scripts/ from drifting out of sync with
the current data columns and the final paper design.
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
DATA_FILE = ROOT / "merged_train_df.csv"

MAIN_MODULES = [
    "matplotlib",
    "seaborn",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
]
SUPPLEMENT_MODULES = MAIN_MODULES + ["optuna"]


def check_environment(module_names: list[str]) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        message = [
            "The current Python environment is missing required packages:",
            "  " + ", ".join(missing),
            "",
            f"Current Python: {sys.executable}",
            "On this computer, the complete environment appears to be:",
            r"  D:\Anaconda\envs\DL\python.exe",
            "",
            "Try:",
            r"  D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_figures.py",
        ]
        raise SystemExit("\n".join(message))


def run_script(script_name: str) -> None:
    script_path = FIGURES_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing figure script: {script_path}")
    print(f"\n=== Running {script_path.relative_to(ROOT)} ===")
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures in order.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Generate Fig01-Fig09 only.",
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Generate FigS1-FigS4 only.",
    )
    args = parser.parse_args()

    if args.main_only and args.supplement_only:
        raise SystemExit("Choose either --main-only or --supplement-only, not both.")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing required data file: {DATA_FILE}")

    if args.supplement_only:
        check_environment(SUPPLEMENT_MODULES)
    elif args.main_only:
        check_environment(MAIN_MODULES)
    else:
        check_environment(SUPPLEMENT_MODULES)

    if not args.supplement_only:
        run_script("run_all_figures_refined.py")
    if not args.main_only:
        run_script("run_figS1toS3_refined.py")

    print("\nDone. Figures are saved under the figures/FigXX_* and figures/FigS* folders.")


if __name__ == "__main__":
    main()
