#!/usr/bin/env python3
"""Canonical leakage-safe LightGBM walk-forward entrypoint."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import pandas as pd
import sklearn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lgbm_pipeline import add_forward_return, build_basic_features, make_timestamp_folds, prepare_fold_data
from src.lgbm_training import train_fold
from src.local_data import load_local_ohlcv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lgbm_one_fold.yaml"))
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = REPO_ROOT / config["output"]["root"]
    run_dir = output_root / f"run_{args.run_id}_eth_1h_lgbm"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    clock = time.monotonic()

    data_config = config["data"]
    data, data_profile = load_local_ohlcv(
        REPO_ROOT / data_config["input_path"],
        max_rows=int(data_config["max_rows"]),
        timeframe=data_config["timeframe"],
        timestamp_unit=data_config.get("timestamp_unit"),
        strict_time_grid=bool(data_config.get("strict_time_grid", False)),
        ohlc_consistency=config["validation"]["ohlc_consistency"],
    )
    featured, feature_columns, feature_profile = build_basic_features(data)
    horizon = int(config["label"]["horizon_hours"])
    labeled = add_forward_return(featured, horizon_hours=horizon)
    folds = make_timestamp_folds(
        labeled["date"],
        train_days=int(config["split"]["train_days"]),
        validation_days=int(config["split"]["validation_days"]),
        test_days=int(config["split"]["test_days"]),
        max_folds=int(config["split"]["max_folds"]),
    )

    training = config["training"]
    seed = int(config["run"]["seed"])
    parameters = {
        **training["parameters"],
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "verbosity": -1,
        "num_threads": int(training["num_threads"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }

    fold_metrics: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for fold in folds.to_dict(orient="records"):
        prepared = prepare_fold_data(
            labeled,
            fold,
            feature_columns=feature_columns,
            horizon_hours=horizon,
            positive_rate=float(config["label"]["positive_rate"]),
        )
        result = train_fold(
            prepared,
            feature_columns=feature_columns,
            fold_number=int(fold["fold"]),
            parameters=parameters,
            num_boost_round=int(training["num_boost_round"]),
            early_stopping_rounds=int(training["early_stopping_rounds"]),
            decision_threshold=float(training["decision_threshold"]),
        )
        result.booster.save_model(str(run_dir / f"model_fold_{int(fold['fold']):02d}.txt"))
        fold_metrics.append(result.metrics)
        profiles.append(result.profile)
        predictions.append(result.predictions)
        importances.append(result.feature_importance)

    test_metric_names = list(fold_metrics[0]["model"]["test"])
    aggregate = {
        name: {
            "mean": float(pd.Series([row["model"]["test"][name] for row in fold_metrics]).mean()),
            "std": float(pd.Series([row["model"]["test"][name] for row in fold_metrics]).std(ddof=0)),
        }
        for name in test_metric_names
    }
    write_json(run_dir / "metrics.json", {"folds": fold_metrics, "aggregate_test": aggregate})
    pd.concat(predictions, ignore_index=True).to_parquet(run_dir / "predictions.parquet", index=False)
    pd.concat(importances, ignore_index=True).to_csv(run_dir / "feature_importance.csv", index=False)
    write_json(run_dir / "training_profile.json", {"folds": profiles})
    write_json(run_dir / "data_profile.json", data_profile)
    write_json(run_dir / "feature_profile.json", feature_profile)
    folds.to_csv(run_dir / "folds.csv", index=False)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - clock, 6),
            "python_version": platform.python_version(),
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(git_value("status", "--porcelain")),
            "fold_count": len(fold_metrics),
        },
    )
    print(json.dumps({"run_dir": str(run_dir), "aggregate_test": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
