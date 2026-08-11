# 再現可能な ETH 1時間足 LightGBM パイプライン

このリポジトリのcanonical workflowは、ETH/USDの1時間足OHLCVを取得・検証し、
因果的特徴量と厳密な6時間先ラベルを作成して、LightGBMをrolling walk-forwardで
評価するローカル再現可能な時系列分類baselineです。

## タスクとデータ

各時刻について、正確に6時間後のETH/USDリターンが、そのfoldのtrain期間の
上位20%に入るかを予測します。代表データはCryptoCompare CCCAGGの
2021-04-01 00:00 UTC〜2024-12-31 23:00 UTC、32,904行です。
`data/raw/`はGit管理外です。

ソースには91行の既知OHLC整合性異常があります。値は修正・自動削除せず、
metadataへwarningとして記録します。初期baselineではhigh/low由来特徴量を除外します。

## セットアップ

検証済み環境はCPython 3.11.9、LightGBM 4.7.0です。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-canonical.txt
```

旧notebook専用依存は`legacy/requirements-legacy.txt`に分離されています。

## 実行

データ取得（必要な場合は環境変数`CRYPTOCOMPARE_API_KEY`を設定）:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_eth_ohlcv.py --config configs\fetch_eth_ohlcv.yaml
```

validationとprepare:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\smoke_local.yaml --run-id prepare_check
```

1fold再現性baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_lgbm.py --config configs\lgbm_one_fold.yaml --run-id one_fold
```

代表walk-forward:

```powershell
.\.venv\Scripts\python.exe scripts\train_lgbm.py --config configs\lgbm_representative.yaml --run-id representative
```

model、fold別・集約metrics、predictions、feature importance、実行config、各種profile、
run metadataはGit管理外の`reports/raw/`へ保存されます。

## 評価条件と結果

代表実験は、train 263日、validation 7日、test 30日の36foldで、30日ずつ移動します。
これは旧代表実験の「270日窓の末尾7日をvalidationにする」意味を再構築したものです。
seed 42、1 thread、最大2,000 rounds、validation early stopping 100、判定閾値0.5です。

| Test metric | LightGBM平均 | fold標準偏差 | 定数baseline平均 |
|---|---:|---:|---:|
| ROC-AUC | 0.5581 | 0.0461 | 0.5000 |
| PR-AUC | 0.2402 | 0.0750 | 0.1947 |
| LogLoss | 0.4950 | 0.1042 | 0.4931 |
| Accuracy | 0.8003 | 0.0717 | 0.8053 |
| Precision | 0.1084 | 0.2047 | 0.0000 |
| Recall | 0.0192 | 0.0397 | 0.0000 |
| F1 | 0.0283 | 0.0532 | 0.0000 |
| MCC | 0.0187 | 0.0405 | 0.0000 |

代表実験の独立2回実行ではpredictionsと36個すべてのmodelが一致しました。
ranking metricsは定数baselineを上回りますが、LogLossと固定閾値の分類指標は
baselineを上回っておらず、production readyな結果ではありません。

## リーク対策

特徴量はright-alignedな過去・現在情報だけを使い、centered rolling、未来shift、
`bfill`、globalなfold変換を使いません。6時間先returnは行shiftではなく正確なtimestampで
照合します。half-open split境界に到達または越えるlabel timestampはpurgeし、label閾値はfoldごとのtrainのみで
fitします。学習はtrainのみ、early stoppingはvalidationのみ、testは学習完了後の最終評価に
だけ使用します。

## 構成と制約

canonicalコードは`scripts/fetch_eth_ohlcv.py`、`scripts/run_pipeline.py`、
`scripts/train_lgbm.py`と、`src/local_data.py`、`src/lgbm_pipeline.py`、
`src/lgbm_training.py`です。

このbaselineは単一asset/sourceと因果的OHLCV特徴量のみを使用します。取引コスト、
slippage、calibration、portfolio構築、live trading評価は含みません。

旧Colab/TFT実験は`legacy/`に履歴として保存されています。
`legacy/scripts/final.py`はreferenceであり、canonical pipelineへimportしたり、
実行可能なcanonical entrypointとして扱ったりしないでください。
