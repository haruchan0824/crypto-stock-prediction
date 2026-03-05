# crypto-stock-prediction

## PhaseA について

PhaseA の目的は、**学習なし・依存最小（pandasのみ）**で成果物仕様を先に確定し、
`make_figures` や README 整備を先に進められる状態を作ることです。

- notebook 由来コード（`src/dataset.py`, `src/filtering.py`, `ccxt`, `get_ipython`, `google.colab`）は
  PhaseA のデフォルト実行では import しません。
- `scripts/run_final.py` のデフォルトは `python -m src.pipeline_phaseA` です。
- 旧実装を使う場合のみ `--legacy` を付けます。

## 実行方法

- PhaseA（デフォルト）
  - `python scripts/run_final.py`
- 旧pipeline（legacy）
  - `python scripts/run_final.py --legacy`
- dry-run（新旧どちらも可）
  - `python scripts/run_final.py --dry_run`
  - `python scripts/run_final.py --legacy --dry_run`

## RUN_OUTPUT_DIR に生成される成果物（4点）

`reports/raw/run_YYYYMMDD_HHMMSS`（=`RUN_OUTPUT_DIR`）配下に次の4ファイルを生成します。

- `tft_vs_lgbm_compare.csv`  
  - foldごとの比較プレースホルダ（`fold`, `lgbm_auc`, `tft_auc`, `delta_auc` など）
- `tft_vs_lgbm_summary.json`  
  - `status`, `reason`, `created_at`, `config_path` を含むサマリ
- `lofo_group_agg.csv`  
  - グループLOFO用プレースホルダ（`group`, `delta_auc` など）
- `lofo_feature_agg.csv`  
  - 特徴量LOFO用プレースホルダ（`feature`, `delta_auc` など）
