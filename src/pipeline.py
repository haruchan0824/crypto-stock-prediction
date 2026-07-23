#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

COMPARE_OUTPUT = "tft_vs_lgbm_compare.csv"
SUMMARY_OUTPUT = "tft_vs_lgbm_summary.json"


class PipelineInputError(RuntimeError):
    pass


def resolve_run_output_dir(run_output_dir: Path | None = None) -> Path:
    env_dir = os.environ.get("RUN_OUTPUT_DIR")
    resolved = Path(env_dir) if env_dir else (run_output_dir or Path("reports/raw/manual_run"))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PipelineInputError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataframe(path: Path):
    import pandas as pd

    if not path.exists():
        raise PipelineInputError(f"Input dataframe not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise PipelineInputError(f"Unsupported dataframe extension: {suffix} ({path})")


def _print_columns(df, label: str) -> None:
    print(f"[INFO] Available columns ({label}): {list(df.columns)}")


def _require_columns(df, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _print_columns(df, label)
        raise PipelineInputError(f"Missing required columns for {label}: {missing}")


def load_config(config_path: str) -> dict[str, Any]:
    cfg = _read_json(Path(config_path))
    required_keys = [
        "df_diag_path",
        "base_config_json",
        "cont_features_json",
        "cat_features_json",
        "regime_col",
        "date_col",
        "valid_col",
        "label_col",
        "train_span_days",
        "test_span_days",
        "min_train_rows",
        "min_test_rows",
        "num_epochs",
        "decoder_len",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise PipelineInputError(f"Missing required config keys: {missing}")
    return cfg


def main(config_path: str = "configs/pipeline_config.json", run_output_dir: Path | None = None) -> None:
    cfg = load_config(config_path)
    out_dir = resolve_run_output_dir(run_output_dir)

    from src import legacy_core

    df_diag = _load_dataframe(Path(cfg["df_diag_path"]))
    date_col = str(cfg["date_col"])
    valid_col = str(cfg["valid_col"])
    label_col = str(cfg["label_col"])
    _require_columns(df_diag, [date_col, valid_col, label_col], "df_diag")

    base_config = _read_json(Path(cfg["base_config_json"]))
    cont_features = _read_json(Path(cfg["cont_features_json"]))
    cat_features = _read_json(Path(cfg["cat_features_json"]))
    if not isinstance(base_config, dict):
        raise PipelineInputError("base_config_json must contain a JSON object")
    if not isinstance(cont_features, list) or not isinstance(cat_features, list):
        raise PipelineInputError("cont_features_json/cat_features_json must contain JSON lists")

    # scripts/final.py の関数が参照するグローバルを注入（ロジック変更なし）
    legacy_core.df_diag_v3 = df_diag
    legacy_core.CONT_FEATURES_ALL = cont_features
    legacy_core.CAT_FEATURES_ALL = cat_features
    legacy_core.CONT_FEATURES_FOR_AUC = cont_features
    legacy_core.CAT_FEATURES_FOR_AUC = cat_features

    # scripts/final.py の実行例（5324付近）と同名引数で呼び出し
    tft_results_df, _diag_df, _summary_df, fold_meta_df = legacy_core.regime_aware_wfa_with_tft(
        df_diag=df_diag,
        base_config=base_config,
        regime_col=str(cfg["regime_col"]),
        date_col=date_col,
        valid_col=valid_col,
        strategy_type=cfg.get("strategy_type"),
        eval_strategy_type=cfg.get("eval_strategy_type"),
        cont_features=cont_features,
        cat_features=cat_features,
        train_span_days=int(cfg["train_span_days"]),
        test_span_days=int(cfg["test_span_days"]),
        min_train_rows=int(cfg["min_train_rows"]),
        min_test_rows=int(cfg["min_test_rows"]),
        num_epochs=int(cfg["num_epochs"]),
        decoder_len=int(cfg["decoder_len"]),
        label_col=label_col,
    )

    lgbm_results_df = legacy_core.run_lgbm_wfa_with_diagnostics(
        df_diag=df_diag,
        fold_meta_df=fold_meta_df,
        cont_features=cont_features,
        cat_features_base=cat_features,
        label_col=label_col,
        regime_col=str(cfg["regime_col"]),
        eval_strategy_type=cfg.get("eval_strategy_type"),
    )

    comp_df, summary = legacy_core.fold_by_fold_tft_vs_lgbm(
        tft_results_df=tft_results_df,
        lgbm_results_df=lgbm_results_df,
        fold_meta_df=fold_meta_df,
    )

    comp_df.to_csv(out_dir / COMPARE_OUTPUT, index=False)
    (out_dir / SUMMARY_OUTPUT).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Saved: {out_dir / COMPARE_OUTPUT}")
    print(f"[INFO] Saved: {out_dir / SUMMARY_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase-A TFT vs LGBM artifacts")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.json")
    parser.add_argument("--run_output_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, run_output_dir=args.run_output_dir)
