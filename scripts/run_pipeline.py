#!/usr/bin/env python3
"""Local reproducible pipeline entrypoint (data loading and validation phase)."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.local_data import load_local_ohlcv
from src.lgbm_pipeline import (
    add_forward_return,
    build_basic_features,
    make_prepared_sample,
    make_timestamp_folds,
    prepare_fold_data,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_metadata() -> dict[str, Any]:
    status = _git_value("status", "--porcelain")
    return {
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and validate local ETH 1-hour OHLCV data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_local.yaml"),
        help="YAML configuration path relative to the repository root.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Local CSV/Parquet.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--strict-time-grid", action="store_true", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def _resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config_path = _resolve_from_root(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    loaded["_config_path"] = str(config_path)
    return loaded


def apply_cli_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    merged = {
        **config,
        "pipeline": dict(config.get("pipeline") or {}),
        "data": dict(config.get("data") or {}),
        "features": dict(config.get("features") or {}),
        "label": dict(config.get("label") or {}),
        "split": dict(config.get("split") or {}),
        "validation": dict(config.get("validation") or {}),
        "output": dict(config.get("output") or {}),
        "run": dict(config.get("run") or {}),
    }
    if args.input is not None:
        merged["data"]["input_path"] = str(args.input)
    if args.max_rows is not None:
        merged["data"]["max_rows"] = args.max_rows
    if args.timestamp_unit is not None:
        merged["data"]["timestamp_unit"] = args.timestamp_unit
    if args.strict_time_grid is not None:
        merged["data"]["strict_time_grid"] = args.strict_time_grid
    if args.output_root is not None:
        merged["output"]["root"] = str(args.output_root)
    if args.run_id is not None:
        merged["run"]["run_id"] = args.run_id
    return merged


def validate_config(config: dict[str, Any]) -> None:
    for section in ("pipeline", "data", "validation", "output", "run"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")

    required_data = ("input_path", "asset", "timeframe", "max_rows")
    missing = [key for key in required_data if key not in config["data"]]
    if missing:
        raise ValueError(f"Missing data configuration values: {missing}")
    if str(config["data"]["asset"]).upper() != "ETH":
        raise ValueError("This pipeline phase supports asset='ETH' only.")
    if config["data"]["timeframe"] != "1h":
        raise ValueError("This pipeline phase supports timeframe='1h' only.")
    if int(config["data"]["max_rows"]) <= 0:
        raise ValueError("data.max_rows must be positive.")
    ohlc_consistency = config["validation"].get("ohlc_consistency", "error")
    if ohlc_consistency not in {"error", "warn"}:
        raise ValueError(
            "validation.ohlc_consistency must be either 'error' or 'warn'."
        )
    if not config["output"].get("root"):
        raise ValueError("output.root must not be empty.")
    stage = config["pipeline"].get("stage", "validate")
    if stage not in {"validate", "prepare"}:
        raise ValueError("pipeline.stage must be either 'validate' or 'prepare'.")
    if stage == "prepare":
        for section in ("features", "label", "split"):
            if not isinstance(config.get(section), dict):
                raise ValueError(
                    f"Missing or invalid configuration section for prepare: {section}"
                )
        if config["features"].get("implementation") != "pandas_v1":
            raise ValueError("features.implementation must be 'pandas_v1'.")
        horizon = int(config["label"].get("horizon_hours", 0))
        if horizon != 6:
            raise ValueError("label.horizon_hours must be 6 in this phase.")
        positive_rate = float(config["label"].get("positive_rate", 0))
        if not 0 < positive_rate < 1:
            raise ValueError("label.positive_rate must be between 0 and 1.")
        for key in ("train_days", "validation_days", "test_days", "max_folds"):
            if int(config["split"].get(key, 0)) <= 0:
                raise ValueError(f"split.{key} must be positive.")


def create_run_directory(config: dict[str, Any]) -> tuple[str, Path]:
    asset = str(config["data"]["asset"]).lower()
    timeframe = str(config["data"]["timeframe"]).lower()
    requested = config["run"].get("run_id")
    token = str(requested) if requested else _utc_now().strftime("%Y%m%dT%H%M%SZ")
    if not token or any(character in token for character in '<>:"/\\|?*'):
        raise ValueError("run_id is empty or contains invalid path characters.")

    run_id = f"run_{token}_{asset}_{timeframe}_pipeline"
    output_root = _resolve_from_root(Path(config["output"]["root"])).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("local_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _metadata_base(
    *,
    run_id: str,
    run_dir: Path,
    config: dict[str, Any],
    input_path: Path,
    started_at: datetime,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "started",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "entrypoint": str(Path(__file__).resolve()),
        **_git_metadata(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "asset": config["data"]["asset"],
        "timeframe": config["data"]["timeframe"],
        "input_path": str(input_path),
        "output_directory": str(run_dir),
        "error_type": None,
        "error_message": None,
    }


def _save_config(path: Path, config: dict[str, Any]) -> None:
    serializable = {key: value for key, value in config.items() if key != "_config_path"}
    path.write_text(
        yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _log_profile(logger: logging.Logger, profile: dict[str, Any]) -> None:
    logger.info("Input format: %s", profile["input_file"]["format"])
    logger.info("Original rows: %s", profile["rows"]["original"])
    logger.info("Final rows: %s", profile["rows"]["after_max_rows"])
    logger.info(
        "Data period: %s to %s",
        profile["time"]["min_timestamp"],
        profile["time"]["max_timestamp"],
    )
    logger.info("Column renames: %s", profile["columns"]["renamed"])
    logger.info(
        "Timestamp interpretation: unit=%s inferred=%s assumed_timezone=%s",
        profile["time"]["timestamp_unit"],
        profile["time"]["timestamp_unit_inferred"],
        profile["time"]["assumed_timezone"],
    )
    logger.info(
        "Time grid: irregular=%s estimated_missing_hours=%s min=%s max=%s",
        profile["time"]["irregular_interval_count"],
        profile["time"]["estimated_missing_hours"],
        profile["time"]["minimum_interval"],
        profile["time"]["maximum_interval"],
    )
    logger.info("Input fingerprint: %s", profile["input_file"])
    quality = profile["quality"]
    if (
        quality["ohlc_consistency_policy"] == "warn"
        and quality["invalid_ohlc_rows"]
    ):
        logger.warning(
            "%d OHLC consistency violations found. Raw values are preserved; "
            "no rows were dropped or corrected. min_timestamp=%s max_timestamp=%s "
            "violation_counts=%s sample_first_20=%s",
            quality["invalid_ohlc_rows"],
            quality["invalid_ohlc_min_timestamp"],
            quality["invalid_ohlc_max_timestamp"],
            quality["invalid_ohlc_violation_counts"],
            quality["invalid_ohlc_rows_sample"],
        )


def _run_prepare_stage(
    data: pd.DataFrame,
    *,
    config: dict[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    featured, feature_columns, feature_profile = build_basic_features(data)
    horizon_hours = int(config["label"]["horizon_hours"])
    prepared = add_forward_return(featured, horizon_hours=horizon_hours)
    folds = make_timestamp_folds(
        prepared["date"],
        train_days=int(config["split"]["train_days"]),
        validation_days=int(config["split"]["validation_days"]),
        test_days=int(config["split"]["test_days"]),
        max_folds=int(config["split"]["max_folds"]),
    )

    fold_rows: list[dict[str, Any]] = []
    split_profiles: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    for fold in folds.to_dict(orient="records"):
        result = prepare_fold_data(
            prepared,
            fold,
            feature_columns=feature_columns,
            horizon_hours=horizon_hours,
            positive_rate=float(config["label"]["positive_rate"]),
        )
        counts = result.profile["counts"]
        splits = result.profile["splits"]
        fold_rows.append(
            {
                **{key: fold[key] for key in folds.columns},
                "n_train_before_purge": counts["n_train_before_purge"],
                "n_train": counts["n_train"],
                "n_validation_before_purge": counts[
                    "n_validation_before_purge"
                ],
                "n_validation": counts["n_validation"],
                "n_test_before_filter": counts["n_test_before_filter"],
                "n_test": counts["n_test"],
                "purged_train_rows": counts["purged_train_rows"],
                "purged_validation_rows": counts["purged_validation_rows"],
                "label_threshold": result.label_threshold,
                "positive_rate_train": splits["train"]["positive_rate"],
                "positive_rate_validation": splits["validation"]["positive_rate"],
                "positive_rate_test": splits["test"]["positive_rate"],
                "status": "prepared",
            }
        )
        split_profiles.append(result.profile)
        sample = make_prepared_sample(result, feature_columns=feature_columns)
        sample.insert(0, "fold", int(fold["fold"]))
        samples.append(sample)

    _write_json(run_dir / "feature_profile.json", feature_profile)
    _write_json(run_dir / "feature_columns.json", feature_columns)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    _write_json(
        run_dir / "split_profile.json",
        {
            "horizon_hours": horizon_hours,
            "positive_rate_target": float(config["label"]["positive_rate"]),
            "folds": split_profiles,
        },
    )
    pd.concat(samples, ignore_index=True).to_parquet(
        run_dir / "prepared_sample.parquet", index=False
    )
    logger.info(
        "Prepare stage completed: features=%d rows_after_warmup=%d folds=%d",
        len(feature_columns),
        len(featured),
        len(folds),
    )


def main() -> int:
    args = parse_args()
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    run_dir: Path | None = None
    metadata: dict[str, Any] | None = None
    logger: logging.Logger | None = None

    try:
        config = apply_cli_overrides(load_config(args.config), args)
        validate_config(config)
        input_path = _resolve_from_root(Path(config["data"]["input_path"])).resolve()
        run_id, run_dir = create_run_directory(config)
        logger = configure_logging(run_dir)
        metadata = _metadata_base(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            input_path=input_path,
            started_at=started_at,
        )
        _write_json(run_dir / "run_metadata.json", metadata)
        _save_config(run_dir / "config_used.yaml", config)

        logger.info("Configuration file: %s", config["_config_path"])
        logger.info("Input file: %s", input_path)
        logger.info("Run directory: %s", run_dir)

        data, profile = load_local_ohlcv(
            input_path,
            max_rows=int(config["data"]["max_rows"]),
            timeframe=str(config["data"]["timeframe"]),
            timestamp_unit=config["data"].get("timestamp_unit"),
            strict_time_grid=bool(config["data"].get("strict_time_grid", False)),
            ohlc_consistency=str(
                config["validation"].get("ohlc_consistency", "error")
            ),
        )
        _write_json(run_dir / "data_profile.json", profile)
        _log_profile(logger, profile)
        if profile["time"]["irregular_interval_count"]:
            logger.warning("Irregular 1-hour intervals were found; no rows were filled.")
        if config["pipeline"].get("stage", "validate") == "prepare":
            _run_prepare_stage(
                data,
                config=config,
                run_dir=run_dir,
                logger=logger,
            )

        finished_at = _utc_now()
        metadata.update(
            status="completed",
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=round(time.monotonic() - monotonic_start, 6),
        )
        _write_json(run_dir / "run_metadata.json", metadata)
        logger.info("Pipeline completed successfully after data validation.")
        return 0
    except Exception as exc:
        finished_at = _utc_now()
        if metadata is not None and run_dir is not None:
            metadata.update(
                status="failed",
                finished_at_utc=finished_at.isoformat(),
                duration_seconds=round(time.monotonic() - monotonic_start, 6),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            _write_json(run_dir / "run_metadata.json", metadata)
        if logger is not None:
            logger.exception("Pipeline failed: %s", exc)
        else:
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
