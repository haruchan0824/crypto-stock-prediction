# Interview Prep Notes: TFT vs LightGBM for Financial Time-Series Prediction

## 1. Project Summary

### What the project is
- This project compares **Temporal Fusion Transformer (TFT)** and **LightGBM** on financial time-series prediction tasks.
- It covers two domains:
  - **Cryptocurrency forecasting**
  - **Stock forecasting**
- The goal is not just to report one average score, but to understand how model behavior changes across folds, domains, and feature sets.

### Why it was done
- In financial ML, a model that looks good on one split can fail under a different regime or asset class.
- I wanted to test whether a sequence model like TFT adds value beyond a strong tabular baseline like LightGBM.
- I also wanted to understand **where** each model performs well, not only **whether** it wins on average.

### What models were compared
- **TFT**: a sequence-aware model designed to capture temporal context and feature interactions over time.
- **LightGBM**: a strong and interpretable gradient boosting baseline for engineered tabular features.

### What domains were covered
- **Crypto**: evaluated with walk-forward validation and LOFO-style interpretation.
- **Stocks**: evaluated under aligned fold-based settings with fold-level and LOFO-based interpretation.

## 2. One-Minute Explanation

> I built a project comparing Temporal Fusion Transformer and LightGBM for financial time-series prediction across crypto and stocks. The reason was to test whether a more complex sequential model actually beats a strong tabular baseline under fair, time-respecting evaluation. In crypto, the result was basically a near tie: TFT had mean AUC 0.601 versus 0.603 for LightGBM, so the main takeaway there was regime sensitivity rather than clear dominance. In stocks, TFT performed better on average, with mean AUC 0.5815 versus 0.5127 for LightGBM, and it won 4 out of 5 folds. I also used LOFO analysis to understand feature dependence. The overall lesson is that model choice depends on domain, regime, and feature interaction, so I focused on interpretation, not just headline performance.

## 3. Three-Minute Explanation

> This project compares TFT and LightGBM across two financial time-series settings: crypto and stocks. I chose those two models because they represent different modeling assumptions. TFT is a sequence model and is meant to capture temporal context and richer interactions over time. LightGBM is a very strong baseline when the feature engineering is already good, so it is a useful benchmark.
>
> For crypto, I used walk-forward validation because time order matters and I wanted to avoid leakage from future data into training. The main metric was AUC, because the task is directional classification and AUC gives a threshold-independent view of ranking quality. In crypto, TFT and LightGBM ended up almost tied on average: TFT mean AUC was 0.601 and LightGBM mean AUC was 0.603. TFT won a little over half of the folds, which suggested that its performance was more regime-sensitive than consistently better.
>
> I then looked beyond the mean score. Fold-by-fold plots, AUC distribution, and win-vs-non-win analysis showed that TFT had stronger upside in some folds but also meaningful downside in others. I also used LOFO analysis to see which feature groups or individual features appeared helpful versus redundant.
>
> In the stock experiment, I kept the comparison aligned at the fold level. There, TFT showed a clearer average advantage: mean AUC 0.5815 versus 0.5127 for LightGBM, and TFT won 4 of 5 folds. The fold-level table showed that the biggest relative gain was in fold 2 and the only losing fold for TFT was fold 3. The LOFO results suggested that TFT benefited more from a richer mix of macro, market, and temporal-context features, while LightGBM behaved more like a strong tabular model tied to summary-style handcrafted features.
>
> So the main conclusion is that the answer is domain-dependent. A more complex model is not automatically better, but architecture and feature interaction can matter a lot depending on the financial setting.

## 4. Five-Minute Explanation

> The project started from a practical question: if I already have a strong tabular baseline like LightGBM with engineered financial features, when does it make sense to move to a more complex sequential model like TFT?
>
> I chose TFT and LightGBM because they are a meaningful contrast. TFT is built to model temporal structure and changing feature importance over time, while LightGBM is often very competitive when the useful signal is already summarized well by handcrafted inputs. In finance, that comparison matters because many improvements disappear once you benchmark against a strong baseline.
>
> I evaluated the models in a time-respecting way. For crypto, I used walk-forward validation. That means each fold trains on historical data available up to a point and evaluates on the next unseen window. I avoided random splits because they can leak future information or create unrealistic train-test overlap. For stocks, I used aligned fold-based evaluation so that TFT and LightGBM were compared under the same fold structure.
>
> I used AUC as the main metric because the task is directional classification and I wanted a threshold-independent measure of ranking quality. Then I looked at more than just the average score. I separated overall fold-level comparison, fold-by-fold behavior, and LOFO-based interpretation.
>
> In crypto, the average result was basically a draw. TFT mean AUC was 0.601 and LightGBM mean AUC was 0.603, with mean delta of -0.0015. TFT still won about 52% of folds, which is why the interpretation is not simply that LightGBM was better. The fold-level and win-vs-non-win analysis suggested that TFT had regime-sensitive behavior: it could outperform meaningfully in some folds and underperform in others. So the crypto result taught me that a sequence model can be competitive without being consistently superior.
>
> I also used LOFO to understand model dependence on features. In crypto, the LOFO interpretation supported the idea that on-chain features contributed meaningful signal. I treat that cautiously because LOFO is better viewed as a sensitivity analysis than as a causal importance measure, but it still helped identify which inputs looked useful versus redundant.
>
> In stocks, the story was different. TFT showed a clearer average advantage: mean AUC 0.5815 versus 0.5127 for LightGBM, standard deviation 0.0511 versus 0.0155, and TFT won 4 of 5 folds. The largest relative improvement was in fold 2, while fold 3 was the one fold where TFT lost. So I would describe the stock result as a clear average advantage for TFT under this setup, but not as proof that TFT is always better.
>
> The stock LOFO results helped explain that difference. TFT's top LOFO features included macro, market, and temporal-style inputs such as `fred_DGS10_ret1`, `ATR_14`, `volume`, `log_volume`, `ret_1_rolling_mean_20`, `trend_60_120`, and `yf_^GSPC_close_ret1`. For LightGBM, the LOFO pattern suggested stronger reliance on more summary-style handcrafted features such as `ma_120`, `ATR_14`, `ret_1_rolling_mean_20`, and `log_volume`. There were also features whose removal improved LightGBM performance, like `log_ret_1`, `mkt_trend_regime_id`, `log_ret_5`, and `ma_120`, which suggests that some features may have been redundant or harmful for that architecture under this setup.
>
> The final takeaway is cross-domain. In crypto, LightGBM remained extremely competitive and TFT looked more regime-sensitive than clearly better. In stocks, TFT had a stronger average edge. So my conclusion is not that one model always wins. It is that model choice depends on domain, regime, and how the feature space interacts with the architecture. That is also why I would extend this work with more domain coverage, better stock-side artifact reporting, and potentially ensembles or regime-switching approaches.

## 5. Experiment Design Notes

### Why TFT and LightGBM were compared
- They are meaningfully different baselines.
- TFT tests whether sequential modeling adds value.
- LightGBM tests whether strong feature engineering is already enough.
- If TFT beats LightGBM, that is informative because LightGBM is a strong benchmark, not a weak baseline.

### Why walk-forward or fold-based evaluation was used
- Financial data are time-ordered.
- Random splits can create unrealistic leakage or overstate generalization.
- Walk-forward validation better reflects deployment because the model is always evaluated on future data.
- The stock setup used aligned folds so the model comparison stayed fair.

### Why AUC was used
- The task is directional classification.
- AUC is threshold-independent.
- It is useful when comparing ranking quality across folds rather than focusing on a single decision threshold.

### How leakage was avoided
- The evaluation respected time order.
- Crypto used walk-forward validation, where training uses past data and testing uses later unseen windows.
- The stock comparison was aligned fold-by-fold rather than using random mixing across time.
- [TODO] If asked, be ready to describe the exact feature-generation timing constraints in the pipeline.

### Why LOFO was used
- LOFO helps test how performance changes when a feature or feature group is removed.
- It is useful for diagnosing which inputs appear helpful, redundant, or potentially harmful.
- It is especially useful when comparing architectures that may use the same features differently.
- It should be described as a **performance sensitivity tool**, not a causal explanation tool.

## 6. Crypto Findings

### Main result
- **TFT mean AUC:** 0.601
- **LightGBM mean AUC:** 0.603
- **Mean delta (TFT - LGBM):** -0.0015
- **TFT win rate:** ~52%

### Key interpretation points
- The crypto result was **very close on average**.
- LightGBM had a slightly higher mean AUC.
- TFT still won a little over half of the folds.
- That combination suggests **regime-sensitive competitiveness**, not consistent superiority.
- Fold-level analysis mattered more than just looking at the mean.
- LOFO interpretation suggested that **on-chain features contributed meaningful signal**.

### Cautions / limitations
- The average difference was very small.
- Win rate alone is not the headline because stronger losses in some folds can offset more frequent wins.
- LOFO should not be overstated as a definitive importance ranking.
- The crypto interpretation is stronger as an analysis of behavior across folds than as a claim of model dominance.

## 7. Stock Findings

### Main full-fold result
- **Total folds analyzed:** 5
- **Mean TFT AUC:** 0.5815
- **Std TFT AUC:** 0.0511
- **Mean LightGBM AUC:** 0.5127
- **Std LightGBM AUC:** 0.0155
- **Mean delta AUC (TFT - LGBM):** 0.0689
- **TFT wins:** 4 out of 5 folds

### Fold-by-fold nuance
- TFT outperformed LightGBM in **4 of 5 folds**.
- **Best fold by delta:** fold 2
- **Worst fold by delta:** fold 3
- Fold details:

| Fold | TFT AUC | LightGBM AUC | Delta AUC |
| --- | ---: | ---: | ---: |
| 0 | 0.585650 | 0.505614 | 0.080036 |
| 1 | 0.584992 | 0.515848 | 0.069143 |
| 2 | 0.613839 | 0.504185 | 0.109654 |
| 3 | 0.486788 | 0.496575 | -0.009787 |
| 4 | 0.636347 | 0.541082 | 0.095265 |

### LOFO interpretation
- TFT top LOFO features included:
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
- LightGBM LOFO results suggested stronger reliance on more static or summary-style features such as:
  - `ma_120`
  - `ATR_14`
  - `ret_1_rolling_mean_20`
  - `log_volume`
  - `fred_term_spread_10y_2y`
  - `fred_DGS10_ret1`
  - `yf_^VIX_close_ret1`
  - `yf_^GSPC_close_ret1`
- Some feature removals improved LightGBM performance, including examples like:
  - `log_ret_1`
  - `mkt_trend_regime_id`
  - `log_ret_5`
  - `ma_120`
- Practical interpretation:
  - TFT appears to benefit more from a richer combination of market, macro, and temporal-context features.
  - LightGBM appears to behave more like a strong tabular baseline whose performance is tied more tightly to how summary-style handcrafted features are defined.

### Cautions / limitations
- The stock result should be described as a **clear average advantage under this setup**, not as a universal win for TFT.
- There were only **5 folds** in the reported comparison.
- Fold 3 shows the advantage was not uniform.
- The stock-side README presentation currently does not include checked-in figure artifacts comparable to the crypto section.

## 8. Cross-Domain Comparison

### Similarities between crypto and stock
- Both experiments compare the same two models.
- Both use fold-based, time-respecting evaluation logic.
- Both show that average metrics alone are not enough.
- Both benefit from LOFO-style interpretation for feature sensitivity.

### Differences between crypto and stock
- In **crypto**, the average result was nearly tied.
- In **stocks**, TFT had a clearer average advantage.
- In crypto, the main interpretation is regime sensitivity.
- In stocks, the main interpretation is stronger average performance for TFT under the evaluated setup.
- In stocks, the LOFO pattern suggests stronger architecture-specific feature interaction differences.

### What this suggests about model choice and feature interaction
- Model choice appears to be **domain-dependent**.
- Feature usefulness may depend on the model architecture.
- A strong tabular baseline can remain highly competitive even when a sequential model is available.
- Richer features do not help every model in the same way.
- It is important to evaluate both **performance** and **feature interaction** rather than treating model selection as architecture-only.

## 9. Key Technical Talking Points

- I compared TFT and LightGBM because they represent sequence modeling versus strong tabular modeling.
- I treated LightGBM as a serious benchmark, not as a weak baseline.
- I used time-respecting evaluation because random splits are risky in financial data.
- In crypto, the headline was not that TFT won; the headline was that the models were nearly tied on average.
- In crypto, TFT looked more regime-sensitive than consistently superior.
- In stocks, TFT showed a clear average advantage under the reported 5-fold setup.
- I focused on fold-level interpretation, not only mean AUC.
- I used AUC because it is appropriate for directional classification and is threshold-independent.
- LOFO was useful to compare feature sensitivity across architectures.
- I describe LOFO carefully as a sensitivity analysis, not a causal importance claim.
- Crypto LOFO suggested on-chain signals mattered.
- Stock LOFO suggested TFT benefited more from richer market, macro, and temporal-context inputs.
- Some features improved LightGBM when removed, which suggests redundancy or feature-architecture mismatch.
- One of the main lessons is that model choice can change by domain.
- Another lesson is that better architecture alone is not enough; evaluation design and feature interaction matter.

## 10. Limitations and Future Work

### What is not solved yet
- The project does not show that one model is best across all financial domains.
- The stock result is based on a limited 5-fold summary.
- The stock-side reporting is less complete visually than the crypto-side reporting.
- LOFO gives sensitivity evidence, but not causal proof.
- [TODO] Add more implementation details if interviewers ask about exact preprocessing and feature-timestamp alignment.

### What the next improvements would be
- Add stock-side visual artifacts similar to the crypto figure set.
- Extend evaluation to more domains and longer time ranges.
- Investigate regime-switching or architecture-routing strategies.
- Explore ensembles of TFT and LightGBM.
- Use LOFO results to prune weak or harmful features more systematically.
