# TFT vs LightGBM for Financial Time-Series Prediction

This portfolio project compares **Temporal Fusion Transformer (TFT)** and **LightGBM** across multiple financial time-series domains under aligned, fold-based evaluation settings. The current repository documents two validated sections of the study:

- **Cryptocurrency forecasting**, where the two models were nearly tied on average.
- **Stock forecasting**, where TFT showed a clear average advantage under the evaluated setup.

## Executive Summary

The main objective of this project is not to argue that one model is universally superior, but to show how model behavior changes across domains, folds, and feature sets.

| Domain | Evaluation View | TFT | LightGBM | Headline |
| --- | --- | ---: | ---: | --- |
| Crypto | Walk-forward AUC | Mean AUC = 0.601 | Mean AUC = 0.603 | Performance was effectively tied on average, with TFT showing regime-sensitive wins. |
| Stocks | 5-fold aligned comparison | Mean AUC = 0.5815 | Mean AUC = 0.5127 | TFT outperformed LightGBM on average, although fold-level variation remained important. |

A practical takeaway is that **model choice and feature interaction appear to depend on both domain and architecture**. In crypto, a strong tabular baseline remained highly competitive. In stocks, TFT captured a stronger average signal under the same evaluation framing.

## Project Overview

Financial prediction problems are difficult because markets are noisy, non-stationary, and often sensitive to regime changes. A model that works well in one period or asset class may not generalize cleanly to another.

This project studies that problem by comparing two different modeling approaches:

- **Temporal Fusion Transformer (TFT)** for sequence-aware learning over time.
- **LightGBM** as a strong, interpretable tabular baseline built on engineered features.

The comparison matters because these models make different assumptions about how signal is represented:

- TFT can exploit temporal context, variable interactions, and changing sequential structure.
- LightGBM is often extremely competitive when the feature set already summarizes the signal well.

The goal is therefore broader than leaderboard performance. The project asks:

1. How stable is each model across folds?
2. Does performance depend on domain or market regime?
3. Which feature groups appear useful, redundant, or potentially harmful?

## Motivation

A recurring issue in applied ML for markets is that aggregate metrics can hide important behavior.

Two models can have similar mean performance while behaving very differently:

- one may be more stable,
- another may win strongly in selective periods,
- and both may rely on very different parts of the feature space.

That is why this README separates:

- **overall fold-level model comparison**,
- **fold-by-fold detail**,
- and **LOFO-based interpretation**.

This structure is intended to make the project useful both as a portfolio artifact and as a technical discussion piece for ML and data science interviews.

## Modeling Approach

### Models

- **TFT**: sequence model designed for temporal forecasting with dynamic feature interaction.
- **LightGBM**: strong gradient boosting baseline for structured, feature-engineered data.

### Feature perspective

The project is designed to compare not only architectures, but also how each architecture responds to richer financial inputs. Across the experiments, the feature space includes combinations of price-derived, market, macro, and context-sensitive signals.

The working hypothesis is conservative: richer features do not automatically help every model in the same way.

## Evaluation Design

### Walk-forward and fold-based evaluation

The project uses time-respecting evaluation rather than random splits.

For the crypto experiment, the main evaluation is **walk-forward validation (WFA)**:

1. Train on historical data available up to a given point.
2. Evaluate on the next unseen time window.
3. Move forward and repeat across folds.

For the stock experiment, TFT and LightGBM are compared under **aligned fold-based settings** so that the comparison remains fair at the fold level.

### Why AUC

The primary comparison metric is **AUC (Area Under the ROC Curve)**. AUC is useful here because the task is directional classification and the project is focused on ranking quality across folds rather than any single decision threshold.

### Why regime-aware interpretation

Financial data are not stationary. Volatility, trend structure, liquidity conditions, and macro context can shift the relationship between features and outcomes. For that reason, model performance is interpreted with a **regime-aware mindset**, especially in crypto where fold-level behavior varies meaningfully through time.

## Crypto Results

### Overall fold-level comparison

The crypto experiment asks whether TFT can outperform a strong feature-based baseline under walk-forward evaluation.

| Metric | TFT | LightGBM |
| --- | ---: | ---: |
| Mean AUC | 0.6010 | 0.6030 |
| Mean ΔAUC (TFT - LGBM) | -0.0015 | — |
| TFT win rate | ~52% | — |

The headline result is that **crypto performance was almost equal on average**. LightGBM finished slightly ahead on mean AUC, while TFT won a little over half of the folds. That combination suggests a competitive but uneven profile rather than a clear overall winner.

### Fold-by-fold detail

The fold-by-fold AUC plot shows how both models behave across walk-forward splits.

![Fold AUC](reports/figures/run_real_compare_001/fig03_fold_auc.png)

The key observation is that fold-level behavior differs even when the average result is close.

- **Stability:** LightGBM remains a strong benchmark because it stays competitive across the evaluation.
- **Variance:** TFT appears to have periods of stronger relative upside, but those gains are not uniform across all folds.

This is why the average gap alone is not sufficient for interpretation.

### Distribution comparison

The distribution view helps compare robustness rather than just mean performance.

![AUC Distribution](reports/figures/run_real_compare_001/fig04_auc_dist.png)

A tighter spread suggests more consistent behavior across folds, while a wider spread may indicate more upside and downside exposure. In crypto, the distributional comparison supports the conclusion that the models are close overall, but not identical in how they achieve their results.

### Win vs non-win analysis

The win vs non-win contribution chart separates folds where TFT beat LightGBM from folds where it did not.

![Win vs Non-win Contribution](reports/figures/run_real_compare_001/fig07_winfold_contrib.png)

This figure is useful as a **secondary diagnostic**, not the main headline. It suggests that TFT's crypto performance is **regime-sensitive**:

- TFT wins decisively in some folds.
- TFT also loses decisively in others.

That pattern helps explain how TFT can post a modest fold win rate advantage while still ending close to, or slightly behind, LightGBM on mean AUC.

### LOFO-based interpretation

#### Group level

The group-level LOFO analysis measures how performance changes when a whole feature group is removed.

![LOFO Group Importance](reports/figures/run_real_compare_001/fig05_lofo_group_delta_auc.png)

Positive ΔAUC contribution suggests a group is helpful. Weak or negative contribution suggests the group may be redundant, noisy, or unstable under this setup. This is useful for deciding which feature families are worth keeping, simplifying, or rethinking.

#### Feature level

The feature-level LOFO chart ranks individual features by ΔAUC contribution.

![LOFO Feature Importance](reports/figures/run_real_compare_001/fig06_lofo_feature_topN_delta_auc.png)

In crypto, the feature-level results support a cautious but important conclusion: **on-chain signals appear to contribute meaningfully**, which suggests the project is capturing more than pure price momentum. At the same time, LOFO should be interpreted as a performance sensitivity analysis, not as a universal statement of causal importance.

## Stock Results

### Overall fold-level comparison

The stock experiment extends the project beyond crypto and asks whether the same model comparison changes under a different financial domain.

| Metric | TFT | LightGBM |
| --- | ---: | ---: |
| Mean AUC | 0.5815 | 0.5127 |
| Std AUC | 0.0511 | 0.0155 |
| Mean ΔAUC (TFT - LGBM) | 0.0689 | — |
| TFT fold wins | 4 / 5 | — |
| Best fold by ΔAUC | Fold 2 | — |
| Worst fold by ΔAUC | Fold 3 | — |

Under this aligned 5-fold stock evaluation, **TFT outperformed LightGBM on average**. That is the main stock result. At the same time, performance still varied by fold, so the result should be interpreted as a clear average advantage under this setup rather than evidence that TFT is always better.

LightGBM remains an important reference point here: it is a strong and interpretable baseline, and the comparison is meaningful precisely because beating it is non-trivial.

### Fold-by-fold detail

The stock fold-by-fold results are summarized below.

| Fold | TFT AUC | LightGBM AUC | ΔAUC (TFT - LGBM) |
| ---: | ---: | ---: | ---: |
| 0 | 0.585650 | 0.505614 | 0.080036 |
| 1 | 0.584992 | 0.515848 | 0.069143 |
| 2 | 0.613839 | 0.504185 | 0.109654 |
| 3 | 0.486788 | 0.496575 | -0.009787 |
| 4 | 0.636347 | 0.541082 | 0.095265 |

This table reinforces the right framing:

- TFT won **4 of 5 folds**.
- The strongest relative gain occurred in **fold 2**.
- TFT underperformed in **fold 3**, which shows that the advantage was not uniform.

The stock experiment therefore supports a **clear average edge for TFT**, while still showing the kind of fold-level variation that matters in real-world financial ML.

### LOFO-based interpretation

The stock LOFO results add useful architectural context.

For **TFT**, top LOFO features included:

- `fred_DGS10_ret1`
- `ATR_14`
- `volume`
- `log_volume`
- `ret_1_rolling_mean_20`
- `trend_60_120`
- `log_ret_5`
- `log_ret_1`
- `yf_^GSPC_close_ret1`
- `ma_120`

For **LightGBM**, LOFO behavior suggests stronger reliance on comparatively static or summary-style features such as:

- `ma_120`
- `ATR_14`
- `ret_1_rolling_mean_20`
- `log_volume`
- `fred_term_spread_10y_2y`
- `fred_DGS10_ret1`
- `yf_^VIX_close_ret1`
- `yf_^GSPC_close_ret1`

The interpretation should remain careful. Some feature removals improved LightGBM performance under this setup, including examples such as `log_ret_1`, `mkt_trend_regime_id`, `log_ret_5`, and `ma_120`. That suggests some inputs may be redundant or even harmful for LightGBM depending on how the feature space is constructed.

A conservative summary is:

- **TFT appears to benefit more from a richer combination of market, macro, and temporal-context features.**
- **LightGBM appears to behave more like a strong tabular baseline whose performance is tied more tightly to how summary-style handcrafted features are defined.**

## Cross-Domain Comparison

The two experiments together are more informative than either one alone.

| Dimension | Crypto | Stocks |
| --- | --- | --- |
| Average model comparison | Near tie; LightGBM slightly higher mean AUC | TFT clearly higher mean AUC under the evaluated setup |
| Fold behavior | Competitive but regime-sensitive | TFT won most folds, but not all |
| Interpretation | Strong baseline remains difficult to beat consistently | Sequence modeling showed a clearer average benefit |
| Feature takeaway | On-chain information appears useful | Feature interaction differs by architecture and domain |

These results suggest that **model choice and feature interaction differ by domain and by architecture**, which is one of the central motivations of the project.

A concise conclusion from the stock experiment is that **the same comparison can produce a different answer in a different financial setting**. In this case, stock forecasting provided stronger evidence for TFT's average advantage than the crypto benchmark did.

## Key Takeaways

- **TFT is not universally better than LightGBM.** The crypto results stayed nearly tied on average, while the stock results favored TFT more clearly.
- **Fold-level analysis matters.** Average metrics alone miss whether gains are steady, selective, or offset by weaker periods.
- **LightGBM remains a strong benchmark.** Its competitiveness is part of what makes the comparison informative.
- **Feature interaction is model-dependent.** The stock LOFO results suggest TFT and LightGBM respond differently to richer financial feature sets.
- **LOFO is most useful as a diagnostic tool.** It helps identify potentially useful, redundant, or harmful inputs, but should be interpreted carefully.

## Repository Structure / Reproducibility

The crypto figure-generation workflow expects run artifacts such as:

- `tft_vs_lgbm_compare.csv` — fold-level comparison table with TFT and LightGBM AUC values.
- `tft_vs_lgbm_summary.json` — aggregate summary metrics for the comparison.
- `lofo_group_agg.csv` — group-level LOFO results with ΔAUC by feature group.
- `lofo_feature_agg.csv` — feature-level LOFO results with ΔAUC by individual feature.

The current README embeds the available crypto figures from:

- `reports/figures/run_real_compare_001/fig03_fold_auc.png`
- `reports/figures/run_real_compare_001/fig04_auc_dist.png`
- `reports/figures/run_real_compare_001/fig05_lofo_group_delta_auc.png`
- `reports/figures/run_real_compare_001/fig06_lofo_feature_topN_delta_auc.png`
- `reports/figures/run_real_compare_001/fig07_winfold_contrib.png`

The stock section in this README is currently presented from validated summary and fold-level results rather than repository figure files.

## Future Work

- Add stock-side visual artifacts to mirror the crypto reporting flow.
- Evaluate whether cross-domain ensembling improves consistency.
- Use LOFO results to prune or regroup weak features more systematically.
- Explore regime-switching or architecture-routing strategies.
- Extend the comparison to additional financial domains and longer evaluation windows.
