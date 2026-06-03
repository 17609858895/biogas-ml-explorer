# scripts 文件夹说明

`scripts` 用来放项目级的自动化脚本，不直接存放最终图片。

当前推荐入口有三个：

```bash
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_figures.py
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_subfigures.py
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_tables.py
```

原因：默认 `base` 环境缺少 `xgboost`、`lightgbm`、`catboost`、`shap` 和 `optuna`，而 `DL` 环境已经具备完整绘图和建模依赖。

它会按论文顺序调用：

- `figures/run_all_figures_refined.py`：生成精修版 Fig01 到 Fig09，并生成数据审计附图 FigS1。
- `figures/run_figS1toS3_refined.py`：生成精修版 FigS2 到 FigS4。
- `scripts/generate_biogas_subfigures.py`：在每个图文件夹的 `subfigures` 子文件夹中导出独立子图，并生成对应 Jupyter 入口。
- `scripts/generate_biogas_paper_tables.py`：生成论文建议保留的少量表格。

实际逐图修改建议在 `figures/FigXX_*/*.ipynb` 中完成，因为你平时使用 Jupyter，这样最方便查看中间结果和局部调整。

旧版 `generate_biogas_paper_figures.py` 曾经包含一整份重复绘图代码，但它使用了过期列名和旧参考编号。现在已经改成统一入口，避免和 `figures` 下的新版代码不一致。
