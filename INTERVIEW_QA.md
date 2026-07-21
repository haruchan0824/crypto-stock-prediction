# Mock Interview Q&A: TFT vs LightGBM for Financial Time-Series Prediction

This document is for interview preparation. The answers are intentionally concise, factual, and optimized for speaking.

## 1. Project Overview

### Q1. What is this project about?
**Short answer:**
This project compares **Temporal Fusion Transformer (TFT)** and **LightGBM** on financial time-series prediction across **crypto** and **stocks**. The main goal was to evaluate not only average performance, but also fold-level behavior and feature sensitivity.

**Extended answer:**
I wanted the project to be more than a benchmark table. So I included fold-by-fold evaluation, walk-forward validation for crypto, aligned folds for stocks, and LOFO analysis to understand which features each model seemed to depend on.

**Interview intent:**
Can the candidate explain the project clearly at a high level?

### Q2. What was the main outcome of the project?
**Short answer:**
The outcome was **domain-dependent**. In crypto, TFT and LightGBM were nearly tied on average, while in stocks, TFT showed a clearer average advantage under the reported setup.

**Extended answer:**
That contrast was the main lesson. It suggested that model choice depends on the interaction between domain, evaluation setting, and feature space rather than on model complexity alone.

**Interview intent:**
Can the candidate summarize the result without overselling it?

### Q3. What makes this project different from just training a model and reporting a score?
**Short answer:**
I treated it as an analysis project, not just a leaderboard exercise. I separated overall fold-level comparison, fold-by-fold behavior, and LOFO-based interpretation so I could explain *why* the models behaved differently.

**Extended answer:**
That matters in financial ML because a single average score can hide instability, regime sensitivity, or feature redundancy. I wanted to show model behavior, not only final metrics.

**Interview intent:**
Is the candidate thinking beyond basic benchmarking?

## 2. Motivation and Problem Setting

### Q4. Why did you choose this problem?
**Short answer:**
Financial time-series prediction is a good test case for robustness because the data are noisy, non-stationary, and regime-sensitive. It is also a setting where stronger models do not automatically beat strong baselines.

**Interview intent:**
Does the candidate understand why the problem is technically interesting?

### Q5. Why compare crypto and stocks in the same project?
**Short answer:**
I wanted to see whether the same model comparison would generalize across domains. Using both crypto and stocks helped test whether conclusions were stable or domain-specific.

**Extended answer:**
That turned out to be useful because the answer was not the same in both domains. Crypto looked close overall, while stocks favored TFT more clearly.

**Interview intent:**
Can the candidate justify the multi-domain setup?

### Q6. Why use this task formulation instead of a regression setup?
**Short answer:**
The evaluation was framed as directional classification, which makes AUC a natural primary metric. That let me compare ranking quality without depending on a single threshold.

**Extended answer:**
I would not claim classification is the only valid formulation. But for this project, it made the comparison cleaner and easier to interpret across folds.

**Interview intent:**
Can the candidate explain modeling choices and trade-offs?

## 3. Model Choice

### Q7. Why compare TFT and LightGBM specifically?
**Short answer:**
They represent two strong but different approaches. TFT is a sequence model designed to capture temporal context, while LightGBM is a strong tabular baseline for engineered features.

**Extended answer:**
That makes the comparison meaningful. If TFT only beats weak baselines, the result is less convincing. Comparing it to LightGBM raises the bar.

**Interview intent:**
Does the candidate understand why the comparison is credible?

### Q8. Why is LightGBM a good baseline here?
**Short answer:**
Because LightGBM is strong, fast, interpretable, and often hard to beat when the feature engineering is already good. It is the kind of baseline I would expect a serious applied ML project to include.

**Interview intent:**
Is the candidate using baselines appropriately?

### Q9. What do you expect TFT to capture that LightGBM might miss?
**Short answer:**
TFT is better positioned to capture temporal context and more complex feature interactions across time. That can matter when the relationship between features and outcomes changes over sequences rather than within static snapshots.

**Extended answer:**
I would still be careful here: the project does not prove TFT captures those effects directly. It only shows that under some settings, especially stocks, TFT performed better on average.

**Interview intent:**
Can the candidate discuss inductive bias without making unsupported claims?

### Q10. Why not compare more models?
**Short answer:**
I focused on depth over breadth. TFT and LightGBM already give a strong architecture contrast, and I wanted to spend more effort on evaluation and interpretation than on building a wide benchmark list.

**Extended answer:**
Adding more models would be a valid next step, but for this project I wanted a defensible comparison rather than a shallow model zoo.

**Interview intent:**
Can the candidate defend scope decisions?

## 4. Evaluation Design

### Q11. Why did you use walk-forward validation for crypto?
**Short answer:**
Because financial data are time-ordered, and walk-forward validation better matches deployment. It ensures the model is always evaluated on future unseen data rather than on shuffled splits.

**Interview intent:**
Does the candidate understand time-series evaluation fundamentals?

### Q12. Why did you use aligned folds for stocks?
**Short answer:**
To keep the comparison fair between TFT and LightGBM. The aligned fold structure makes sure both models are evaluated on the same fold boundaries.

**Interview intent:**
Can the candidate explain fairness in experimental design?

### Q13. Why use AUC as the main metric?
**Short answer:**
AUC is appropriate for directional classification and is threshold-independent. It gives a clean view of ranking quality across folds without tying the evaluation to one operating point.

**Extended answer:**
That said, AUC is not the only metric that matters in practice. If I were taking this toward trading decisions, I would also care about calibration, thresholding, and downstream utility metrics.

**Interview intent:**
Does the candidate understand metric selection and its limits?

### Q14. How did you avoid leakage?
**Short answer:**
The main safeguard was using time-respecting evaluation. In crypto, training always used past data and testing used later unseen windows; in stocks, the fold comparison was aligned instead of randomly mixing samples across time.

**Extended answer:**
I would be careful not to overclaim beyond that. If an interviewer asks for exact preprocessing or timestamp alignment details, the right answer is [TODO] unless I have the pipeline in front of me.

**Interview intent:**
Can the candidate explain leakage prevention honestly?

### Q15. Why is fold-level analysis important here?
**Short answer:**
Because averages can hide instability. Two models can have similar mean AUC but very different behavior across folds, which matters a lot in non-stationary domains like finance.

**Interview intent:**
Does the candidate understand variability, not just mean performance?

### Q16. Why did performance vary by fold?
**Short answer:**
The most likely explanation is that the data-generating conditions changed across folds. In financial time series, shifts in volatility, trend, macro context, or market activity can change which model and features work better.

**Extended answer:**
I would phrase that as interpretation, not proof. The fold-level variation is evidence of instability or regime sensitivity, but not direct causal attribution.

**Interview intent:**
Can the candidate interpret variability without overreaching?

## 5. Results Interpretation

### Q17. What happened in the crypto experiment?
**Short answer:**
Crypto was essentially a near tie on average. TFT mean AUC was **0.601**, LightGBM mean AUC was **0.603**, the mean delta was **-0.0015**, and TFT won about **52%** of folds.

**Extended answer:**
The important nuance is that win rate and average AUC were telling slightly different stories. That suggests TFT had meaningful wins in some folds but also meaningful losses in others.

**Interview intent:**
Can the candidate state the crypto result accurately and with nuance?

### Q18. What happened in the stock experiment?
**Short answer:**
In stocks, TFT performed better on average under the reported 5-fold setup. TFT mean AUC was **0.5815** versus **0.5127** for LightGBM, with TFT winning **4 out of 5** folds.

**Extended answer:**
I would frame that as a clear average advantage under this setup, not as a universal conclusion. The fold-level variation still mattered, especially because TFT lost one fold.

**Interview intent:**
Can the candidate state the stock result without overstating it?

### Q19. Why do you say the crypto result is regime-sensitive?
**Short answer:**
Because TFT won in some folds and lost in others even though the average result was very close overall. That pattern is consistent with a model that is competitive but more condition-dependent.

**Interview intent:**
Can the candidate explain “regime-sensitive” in practical terms?

### Q20. Why do you think TFT did better in stocks?
**Short answer:**
The cautious answer is that TFT appeared to benefit more from the richer combination of market, macro, and temporal-context features in that setting. I would treat that as an interpretation supported by the LOFO pattern, not as proof of mechanism.

**Interview intent:**
Can the candidate offer a reasonable hypothesis without inventing certainty?

### Q21. How would you describe the stock fold-by-fold nuance?
**Short answer:**
TFT won **4 of 5 folds**, with the best delta in **fold 2** and the weakest result in **fold 3**. So the stock result was strong on average, but not uniform across every fold.

**Interview intent:**
Does the candidate remember and communicate the fold-level nuance?

## 6. LOFO / Feature Importance

### Q22. Why did you use LOFO instead of simpler feature importance?
**Short answer:**
LOFO is more directly tied to model performance because it measures what happens when a feature or feature group is removed. That makes it useful for comparing feature sensitivity across architectures.

**Extended answer:**
I would not say LOFO is universally better than all other importance methods. I used it because it aligned well with the practical question: which inputs seem helpful, redundant, or harmful under this setup?

**Interview intent:**
Does the candidate understand why LOFO was chosen?

### Q23. How do you interpret features whose removal improves performance?
**Short answer:**
I interpret that as evidence that some features may be redundant, noisy, or poorly matched to the model under the current setup. It is a useful sign that more features are not always better.

**Interview intent:**
Can the candidate reason about negative importance sensibly?

### Q24. What did LOFO suggest in crypto?
**Short answer:**
The crypto LOFO interpretation suggested that **on-chain features contributed meaningful signal**. I would still present that carefully, because LOFO is a sensitivity analysis rather than a causal explanation.

**Interview intent:**
Can the candidate connect feature analysis to the crypto result?

### Q25. What did LOFO suggest in stocks?
**Short answer:**
It suggested that TFT benefited more from a richer mix of market, macro, and temporal-context features, while LightGBM behaved more like a strong tabular model tied to summary-style handcrafted features. It also showed that some features improved LightGBM when removed.

**Extended answer:**
Examples of features whose removal improved LightGBM included `log_ret_1`, `mkt_trend_regime_id`, `log_ret_5`, and `ma_120`. That is useful because it points to possible redundancy or feature-architecture mismatch.

**Interview intent:**
Can the candidate give a nuanced feature interpretation?

## 7. Crypto vs Stock Comparison

### Q26. What did you learn from comparing crypto and stock results directly?
**Short answer:**
I learned that the same architecture comparison can produce different conclusions across domains. Crypto was nearly tied overall, while stocks gave stronger evidence for TFT under the evaluated setup.

**Interview intent:**
Can the candidate extract a cross-domain lesson?

### Q27. What does this suggest about model choice?
**Short answer:**
It suggests model choice should be domain-aware rather than ideology-driven. A more complex model is not automatically better, and feature interaction seems to matter differently by domain.

**Interview intent:**
Does the candidate think pragmatically about model selection?

### Q28. If an interviewer says “so TFT is better,” how would you respond?
**Short answer:**
I would say that is too broad. TFT had a clear average advantage in the reported stock setup, but in crypto the models were essentially tied on average, so the more accurate conclusion is that performance was domain-dependent.

**Interview intent:**
Can the candidate resist overgeneralization?

## 8. Limitations and Future Work

### Q29. What are the main weaknesses of the project?
**Short answer:**
The main limitations are that the stock result is based on a relatively small **5-fold** summary, stock-side artifact reporting is less complete than crypto-side reporting, and LOFO does not provide causal feature importance. I would also avoid claiming implementation details that are not fully documented in front of me.

**Interview intent:**
Can the candidate talk honestly about limitations?

### Q30. What would you improve next?
**Short answer:**
I would add stock-side visual artifacts, extend the evaluation across more domains and longer time windows, and use LOFO findings to prune weaker features more systematically. I would also explore regime-switching or ensemble approaches.

**Interview intent:**
Can the candidate propose realistic next steps?

### Q31. How would you productionize this project?
**Short answer:**
I would start with a reproducible offline pipeline, fixed train-validation schedules, and monitoring for data drift and fold-level degradation. Then I would add model versioning, scheduled retraining, and business-facing evaluation beyond AUC.

**Extended answer:**
I would be careful not to pretend the current repository is already production-ready. A production version would also need stronger feature-timestamp guarantees, automated backfills, and decision-threshold monitoring. Some implementation specifics are [TODO].

**Interview intent:**
Can the candidate connect research work to production thinking?

## 9. Engineering / Reproducibility / Repository Design

### Q32. How is the repository organized to support reproducibility?
**Short answer:**
The repository includes scripts, configs, source modules, figures, and documentation that separate modeling logic from reporting. The README also documents expected artifacts such as `tft_vs_lgbm_compare.csv`, `tft_vs_lgbm_summary.json`, `lofo_group_agg.csv`, and `lofo_feature_agg.csv`.

**Interview intent:**
Does the candidate think about reproducibility, not just modeling?

### Q33. What would you say if an interviewer asks for implementation details that are not documented here?
**Short answer:**
I would answer with what is documented and clearly label anything missing as [TODO]. In an interview, I would rather be precise and incomplete than confident and inaccurate.

**Interview intent:**
Does the candidate handle uncertainty well?

### Q34. How do the README and interview-prep documents complement each other?
**Short answer:**
The README is public-facing and structured like a project report, while the interview-prep materials are optimized for speaking. That split helps me keep the public documentation polished while still preparing concise, defensible answers.

**Interview intent:**
Can the candidate explain documentation strategy?

### Q35. What is the most defensible one-sentence takeaway from the whole project?
**Short answer:**
A strong tabular baseline and a sequence model can behave very differently across financial domains, so model selection should be based on time-respecting evaluation, fold-level analysis, and feature interaction rather than on architecture alone.

**Interview intent:**
Can the candidate close with a precise, defensible summary?
