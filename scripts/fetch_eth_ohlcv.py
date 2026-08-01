#!/usr/bin/env python3
"""Fetch reproducible ETH hourly OHLCV data independently of model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow
import requests
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.market_data import (  # noqa: E402
    CRYPTOCOMPARE_HISTOHOUR_URL,
    atomic_save_frame,
    fetch_cryptocompare_hourly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ETH/USD 1-hour OHLCV from CryptoCompare."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fetch_eth_ohlcv.yaml"),
        help="YAML configuration path relative to the repository root.",
    )
    parser.add_argument("--start", help="Inclusive start datetime; timezone-naive means UTC.")
    parser.add_argument("--end", help="Inclusive end datetime; timezone-naive means UTC.")
    parser.add_argument("--output", type=Path, help="Destination .parquet or .csv path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of an existing output and metadata file.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    for section in ("data", "request", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' is required.")
    return config


def _parse_utc(value: str | None, *, name: str) -> tuple[pd.Timestamp, bool]:
    if not value:
        raise ValueError(f"{name} must be provided by CLI or configuration.")
    timestamp = pd.Timestamp(value)
    assumed_utc = timestamp.tzinfo is None
    if assumed_utc:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp, assumed_utc


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fetch_eth_ohlcv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = __import__("time").gmtime
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata already exists: {path}. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    try:
        config = _load_config(args.config)
        data_config = config["data"]
        request_config = config["request"]
        output_config = config["output"]
        validation_config = config.get("validation", {})
        if not isinstance(validation_config, dict):
            raise ValueError("Configuration section 'validation' must be a mapping.")

        asset = str(data_config.get("asset", "")).upper()
        quote = str(data_config.get("quote", "")).upper()
        timeframe = str(data_config.get("timeframe", ""))
        if (asset, quote, timeframe) != ("ETH", "USD", "1h"):
            raise ValueError("This entrypoint supports only ETH/USD with timeframe 1h.")

        start_utc, start_assumed = _parse_utc(
            args.start or data_config.get("start"), name="start"
        )
        end_utc, end_assumed = _parse_utc(
            args.end or data_config.get("end"), name="end"
        )
        output_path = _resolve_path(
            args.output or Path(str(data_config.get("output_path", "")))
        )
        metadata_path = _resolve_path(
            Path(
                str(
                    output_config.get(
                        "metadata_path",
                        output_path.with_suffix(".metadata.json"),
                    )
                )
            )
        )
        log_path = _resolve_path(
            Path(str(output_config.get("log_path", "data/raw/eth_1h.fetch.log")))
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        if metadata_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Metadata already exists: {metadata_path}. Use --overwrite to replace it."
            )

        logger = _setup_logger(log_path)
        api_key_env = str(request_config.get("api_key_env", "CRYPTOCOMPARE_API_KEY"))
        api_key = os.environ.get(api_key_env)
        logger.info(
            "Starting CryptoCompare ETH/USD 1h fetch: start=%s end=%s end_inclusive=true",
            start_utc.isoformat(),
            end_utc.isoformat(),
        )

        result = fetch_cryptocompare_hourly(
            asset=asset,
            quote=quote,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=int(request_config.get("limit", 2000)),
            timeout_seconds=float(request_config.get("timeout_seconds", 30)),
            max_retries=int(request_config.get("max_retries", 3)),
            retry_backoff_seconds=float(
                request_config.get("retry_backoff_seconds", 2)
            ),
            page_delay_seconds=float(
                request_config.get("page_delay_seconds", 3)
            ),
            ohlc_consistency_policy=str(
                validation_config.get("ohlc_consistency", "error")
            ).lower(),
            api_key=api_key,
            logger=logger,
        )
        atomic_save_frame(result.frame, output_path, overwrite=args.overwrite)

        status_output = _git_value("status", "--porcelain")
        metadata = {
            "source": "CryptoCompare",
            "api_endpoint": CRYPTOCOMPARE_HISTOHOUR_URL,
            "asset": asset,
            "quote": quote,
            "exchange": "CCCAGG aggregate (explicit e=CCCAGG)",
            "timeframe": timeframe,
            "requested_start": start_utc.isoformat(),
            "requested_end": end_utc.isoformat(),
            "end_inclusive": True,
            "timezone_naive_start_assumed_utc": start_assumed,
            "timezone_naive_end_assumed_utc": end_assumed,
            "effective_start": result.frame["date"].min().isoformat(),
            "effective_end": result.frame["date"].max().isoformat(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(result.frame),
            "min_timestamp": result.frame["date"].min().isoformat(),
            "max_timestamp": result.frame["date"].max().isoformat(),
            "duplicate_rows_removed": result.duplicate_rows_removed,
            "irregular_interval_count": result.irregular_interval_count,
            "estimated_missing_hours": result.estimated_missing_hours,
            "ohlc_consistency_policy": result.ohlc_consistency_policy,
            "invalid_ohlc_row_count": result.invalid_ohlc_row_count,
            "invalid_ohlc_min_timestamp": result.invalid_ohlc_min_timestamp,
            "invalid_ohlc_max_timestamp": result.invalid_ohlc_max_timestamp,
            "invalid_ohlc_violation_counts": result.invalid_ohlc_violation_counts,
            "invalid_ohlc_rows_sample": result.invalid_ohlc_rows_sample,
            "volume_definition": "CryptoCompare volumefrom (ETH base-asset volume)",
            "additional_columns": {
                "volume_quote": "CryptoCompare volumeto (USD quote-asset volume)"
            },
            "output_path": str(output_path),
            "output_format": output_path.suffix.lower().lstrip("."),
            "sha256": _sha256(output_path),
            "request_count": result.request_count,
            "api_key_used": bool(api_key),
            "script_entrypoint": "scripts/fetch_eth_ohlcv.py",
            "git_branch": _git_value("branch", "--show-current"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(status_output),
            "python_version": platform.python_version(),
            "package_versions": {
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "PyYAML": yaml.__version__,
                "requests": requests.__version__,
            },
        }
        _atomic_write_json(metadata_path, metadata, overwrite=args.overwrite)
        logger.info(
            "Completed: rows=%d range=%s..%s output=%s metadata=%s",
            len(result.frame),
            result.frame["date"].min().isoformat(),
            result.frame["date"].max().isoformat(),
            output_path,
            metadata_path,
        )
        return 0
    except Exception as exc:
        logging.getLogger("fetch_eth_ohlcv").error(
            "%s: %s", type(exc).__name__, exc
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
