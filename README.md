# Reproducible ETH 1-hour LightGBM pipeline

This repository contains a leakage-aware, locally reproducible baseline for a
financial time-series classification task. The canonical workflow downloads and
validates ETH/USD hourly OHLCV data, builds causal features and exact six-hour
forward labels, and evaluates LightGBM with rolling walk-forward splits.

## Task definition

At each hourly timestamp, predict whether the ETH/USD return at exactly six
hours ahead is in the top 20% of the current fold's training returns. This is a
binary ranking and probability-estimation task; it is not a trading strategy or
investment recommendation.

## Data

The representative dataset is CryptoCompare CCCAGG ETH/USD 1-hour OHLCV from
2021-04-01 00:00 UTC through 2024-12-31 23:00 UTC (32,904 rows). Raw data stays
under `data/raw/` and is Git-ignored.

The source contains 91 known OHLC consistency anomalies. Validation records
them as warnings and preserves every original value; the pipeline neither
repairs nor drops those rows automatically. High/low-derived features are
excluded from this initial baseline.

## Repository structure

```text
configs/                    Canonical acquisition, prepare, and training configs
scripts/fetch_eth_ohlcv.py  CryptoCompare downloader
scripts/run_pipeline.py     Local validation and preparation
scripts/train_lgbm.py       One-fold and walk-forward LightGBM entrypoint
src/local_data.py           OHLCV loading, validation, and profiling
src/lgbm_pipeline.py        Causal features, labels, folds, and purge logic
src/lgbm_training.py        Deterministic training, metrics, and artifacts
legacy/                     Preserved Colab/notebook experiments and dependencies
reports/raw/                Generated run artifacts (Git-ignored)
```

Other `src/` modules, historical figures, and interview notes support the
preserved earlier TFT-versus-LightGBM research and are not imported by the
canonical pipeline.

## Environment setup

The verified environment is CPython 3.11.9. Direct canonical dependencies are
pinned in `requirements-canonical.txt`; the working LightGBM version is 4.7.0.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-canonical.txt
```

The broad historical notebook dependencies are preserved separately in
`legacy/requirements-legacy.txt` and are not required for the local pipeline.

## Data acquisition and preparation

CryptoCompare may require `CRYPTOCOMPARE_API_KEY` in the environment. From a
fresh checkout without the raw file:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_eth_ohlcv.py --config configs\fetch_eth_ohlcv.yaml
```

Validate the local data and exercise the one-fold preparation stage:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\smoke_local.yaml --run-id prepare_check
```

Generated profiles record the input fingerprint, row and timestamp checks,
irregular intervals, the 91 OHLC warnings, feature causality, split boundaries,
label thresholds, and purged rows.

## Training and representative execution

Run the reproducibility-focused one-fold baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_lgbm.py --config configs\lgbm_one_fold.yaml --run-id one_fold
```

Run the representative walk-forward experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_lgbm.py --config configs\lgbm_representative.yaml --run-id representative
```

Each run saves models, fold and aggregate metrics, predictions, feature
importance, the resolved configuration, data/feature profiles, training
profiles, and run metadata beneath `reports/raw/`.

## Evaluation methodology

The representative experiment uses 36 fixed rolling folds. Each fold contains
263 days of training, seven immediately following days of validation, and 30
immediately following days of test data, then advances 30 days. This reproduces
the meaning of the legacy 270-day fitting window whose final seven days were
held out for validation.

LightGBM uses seed 42, one CPU thread, up to 2,000 boosting rounds, and early
stopping after 100 rounds without validation improvement. The classification
decision threshold is fixed at 0.5. A constant-probability baseline uses only
the training positive rate.

Reported metrics are ROC-AUC, PR-AUC, LogLoss, Accuracy, Precision, Recall, F1,
and MCC. Ranking and probability metrics are more informative here than the
fixed-threshold classification metrics because positives are deliberately
unbalanced.

## Representative results

Verified on CPython 3.11.9 and LightGBM 4.7.0:

| Test metric | LightGBM mean | Fold std | Constant baseline mean |
|---|---:|---:|---:|
| ROC-AUC | 0.5581 | 0.0457 | 0.5000 |
| PR-AUC | 0.2402 | 0.0749 | 0.1948 |
| LogLoss | 0.4952 | 0.1041 | 0.4932 |
| Accuracy | 0.8002 | 0.0717 | 0.8052 |
| Precision | 0.1084 | 0.2047 | 0.0000 |
| Recall | 0.0192 | 0.0397 | 0.0000 |
| F1 | 0.0283 | 0.0532 | 0.0000 |
| MCC | 0.0187 | 0.0405 | 0.0000 |

Two independent representative executions produced identical predictions and
all 36 identical model files. The one-fold reproducibility check likewise
produced identical models, metrics, predictions, and feature importance.

The model improves ranking metrics over the constant baseline, but its mean
LogLoss and fixed-threshold classification results do not improve on that
baseline. These are baseline results, not evidence of production readiness.

## Leakage prevention

The canonical pipeline enforces the following:

* features use right-aligned past-and-present windows only;
* no centered rolling windows, negative feature shifts, backfill, or global
  fold transformations are used;
* forward returns match the exact timestamp at `t + 6h`, not a row offset;
* train and validation boundary rows are purged when their label timestamp
  crosses the split boundary;
* each label threshold is fitted only on that fold's training returns;
* LightGBM fits on train and early-stops only on validation;
* test is accessed only after training/model selection and never tunes
  parameters, labels, features, or decision thresholds.

## Limitations

This baseline uses only causal OHLCV-derived features and one asset/source.
Known source anomalies are preserved, test performance varies materially by
fold, and the fixed 0.5 decision threshold produces few positive predictions.
No costs, slippage, calibration study, portfolio construction, or live trading
evaluation is included. The current result should be treated as a reproducible
research baseline.

## Legacy/reference material

`legacy/scripts/final.py` and `legacy/notebooks/final.ipynb` preserve the earlier
Colab-oriented TFT/LightGBM research. They may contain Google Drive, GPU, broad
dependency, and notebook-state assumptions. They are reference material only:
do not import them into or treat them as the executable canonical pipeline.
