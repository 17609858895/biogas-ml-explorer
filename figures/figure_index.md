# 论文图件索引

本文件夹已按论文结果章节顺序整理为 `Fig01` 到 `Fig09`，数据审计图移入附录 `FigS1`。每张图都有独立子文件夹，主图同时保存为 `.png` 和 `.pdf`。多数图也有对应 `.ipynb`，适合在 Jupyter 中逐张修改和重跑。

## 推荐运行方式

在项目根目录运行全部主图和附录图：

```bash
python scripts/generate_biogas_paper_figures.py
```

如果默认 Python 报缺少 `xgboost`、`lightgbm`、`catboost`、`shap` 或 `optuna`，请使用本机已安装完整依赖的 `DL` 环境：

```bash
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_figures.py
```

只重画 9 张主图：

```bash
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_figures.py --main-only
```

只重画 4 张附录图：

```bash
D:\Anaconda\envs\DL\python.exe scripts\generate_biogas_paper_figures.py --supplement-only
```

Jupyter 用户也可以打开 `figures/00_run_all_figures.ipynb`，或者逐个打开各 `FigXX_*` 子文件夹中的 notebook。

## 主文图

| 图号 | 论文位置 | 主题 | 参考模板 | 输出 |
|---|---|---|---|---|
| Fig 1 | 3.1 Target & outliers | 目标分布与异常值处理 | No.378 + No.388 | `figures/Fig01_target_distribution/Fig01_target_distribution.png` |
| Fig 2 | 3.2 Correlation & feature engineering | 运行参数与气象变量相关性 | No.367 + No.388 | `figures/Fig02_correlation/Fig02_correlation.png` |
| Fig 3 | 3.3 Model comparison & selection | 9 模型性能、Wilcoxon 和 TOPSIS | No.372 + No.377 + No.379 + No.376 | `figures/Fig03_model_comparison/Fig03_model_comparison.png` |
| Fig 4 | 3.3 Model comparison & selection | 预测值-实测值散点矩阵 | No.377 + No.382 | `figures/Fig04_pred_vs_obs/Fig04_pred_vs_obs.png` |
| Fig 5 | 3.4 Temporal validation & uncertainty | 最优模型时序验证和 conformal 区间 | No.388 + No.382 | `figures/Fig05_temporal_validation/Fig05_temporal_validation.png` |
| Fig 6 | 3.5 Residual & reliability | 残差诊断、Y 随机化和 Williams 图 | No.383 + No.365 | `figures/Fig06_residual_reliability/Fig06_residual_reliability.png` |
| Fig 7 | 3.6 Feature importance | SHAP 全局解释和 permutation importance | No.366 + No.373 + No.384 | `figures/Fig07_SHAP_importance/Fig07_SHAP_importance.png` |
| Fig 8 | 3.7 Non-linear & interaction | PDP、ALE 和双变量 SHAP 交互 | No.373 + No.381 | `figures/Fig08_PDP_ALE_interaction/Fig08_PDP_ALE_interaction.png` |
| Fig 9 | 3.8 Feature-set ablation | 历史、运行、气象特征集消融 | No.369 + No.384 + No.388 | `figures/Fig09_feature_ablation/Fig09_feature_ablation.png` |

## 附录图

| 图号 | 主题 | 参考模板 | 输出 |
|---|---|---|---|
| Fig S1 | 数据审计与可用建模窗口 | No.383 + No.388 | `figures/FigS1_data_audit/FigS1_data_audit.png` |
| Fig S2 | 学习曲线和过拟合诊断 | No.382 | `figures/FigS2_learning_curves/FigS2_learning_curves.png` |
| Fig S3 | 贝叶斯优化历史和超参重要性 | No.380 | `figures/FigS3_bayesian_optimisation/FigS3_bayesian_optimisation.png` |
| Fig S4 | DML 风格的气象因子因果补充 | No.370 / No.387 | `figures/FigS4_residualized_weather/FigS4_residualized_weather.png` |

## 我对论文框架的小修正

当前图序已调整：数据审计移入 Fig S1，原后续主图前移为 Fig 1-Fig 9；模型比较扩展为 9 模型；非线性解释扩展为 PDP、ALE、二维响应和 SHAP 交互组合。统一入口现在调用 `figures/run_all_figures_refined.py` 和 `figures/run_figS1toS3_refined.py`。
