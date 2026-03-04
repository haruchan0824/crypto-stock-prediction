# crypto-stock-prediction

## 実行方法

`run_final` は **新pipeline (`src/pipeline.py`) がデフォルト** です。  
旧 `scripts/final.py` を使う場合だけ `--legacy` を付けます。

- 新pipeline（デフォルト）
  - `python scripts/run_final.py`
- 旧pipeline（legacy）
  - `python scripts/run_final.py --legacy`
- dry-run（新旧どちらも可）
  - `python scripts/run_final.py --dry_run`
  - `python scripts/run_final.py --legacy --dry_run`

## RUN_OUTPUT_DIR に生成される必須成果物

`reports/raw/run_YYYYMMDD_HHMMSS`（=`RUN_OUTPUT_DIR`）配下に、次の4ファイルを出力します。

- `tft_vs_lgbm_compare.csv`
- `tft_vs_lgbm_summary.json`
- `lofo_group_agg.csv`
- `lofo_feature_agg.csv`
