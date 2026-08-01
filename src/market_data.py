"""CryptoCompare OHLCV retrieval and validation.

This module is intentionally independent from the training pipeline and the
Notebook-derived modules.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


CRYPTOCOMPARE_HISTOHOUR_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
REQUIRED_API_FIELDS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volumefrom",
    "volumeto",
)


class MarketDataError(RuntimeError):
    """Raised when retrieval or validation cannot safely continue."""


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    request_count: int
    duplicate_rows_removed: int
    irregular_interval_count: int
    estimated_missing_hours: int
    ohlc_consistency_policy: str
    invalid_ohlc_row_count: int
    invalid_ohlc_min_timestamp: str | None
    invalid_ohlc_max_timestamp: str | None
    invalid_ohlc_violation_counts: dict[str, int]
    invalid_ohlc_rows_sample: list[dict[str, Any]]


def _request_page(
    session: requests.Session,
    *,
    params: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    request_number: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                CRYPTOCOMPARE_HISTOHOUR_URL,
                params=params,
                timeout=timeout_seconds,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= max_retries:
                raise MarketDataError(
                    f"CryptoCompare request failed after {attempt + 1} attempts: {exc}"
                ) from exc
            time.sleep(retry_backoff_seconds * (2**attempt))
            continue

        if response.status_code in retryable_statuses:
            if attempt >= max_retries:
                raise MarketDataError(
                    f"CryptoCompare returned retryable HTTP {response.status_code} "
                    f"after {attempt + 1} attempts."
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else retry_backoff_seconds * (2**attempt)
            except ValueError:
                delay = retry_backoff_seconds * (2**attempt)
            if response.status_code == 429:
                logger.warning(
                    "request=%d retry_attempt=%d wait_seconds=%s error_type=rate_limit_http_429",
                    request_number,
                    attempt + 1,
                    delay,
                )
            time.sleep(max(0.0, delay))
            continue

        if response.status_code in {400, 401, 403}:
            raise MarketDataError(
                f"CryptoCompare rejected the request with HTTP {response.status_code}."
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise MarketDataError(
                f"CryptoCompare returned HTTP {response.status_code}."
            ) from exc

        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise MarketDataError("CryptoCompare returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise MarketDataError("CryptoCompare returned a non-object JSON response.")
        if payload.get("Response") != "Success":
            message = payload.get("Message", "unspecified API error")
            is_rate_limit = (
                payload.get("Response") == "Error"
                and isinstance(message, str)
                and "rate limit" in message.casefold()
            )
            if is_rate_limit:
                if attempt >= max_retries:
                    raise MarketDataError(
                        "CryptoCompare JSON rate limit persisted after "
                        f"{attempt + 1} attempts."
                    )
                delay = retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "request=%d retry_attempt=%d wait_seconds=%s error_type=rate_limit_json",
                    request_number,
                    attempt + 1,
                    delay,
                )
                time.sleep(max(0.0, delay))
                continue
            raise MarketDataError(f"CryptoCompare API error: {message}")
        return payload

    raise AssertionError("retry loop exited unexpectedly")


def _normalize_and_validate(
    rows: list[dict[str, Any]],
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    ohlc_consistency_policy: str = "error",
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, int, int, int, dict[str, Any]]:
    if ohlc_consistency_policy not in {"error", "warn"}:
        raise ValueError(
            "ohlc_consistency_policy must be either 'error' or 'warn'."
        )
    if not rows:
        raise MarketDataError("CryptoCompare returned no OHLCV rows.")

    raw = pd.DataFrame(rows)
    missing = [column for column in REQUIRED_API_FIELDS if column not in raw.columns]
    if missing:
        raise MarketDataError(f"CryptoCompare response is missing fields: {missing}")

    raw = raw.loc[:, REQUIRED_API_FIELDS].copy()
    raw["date"] = pd.to_datetime(raw["time"], unit="s", utc=True, errors="coerce")
    if raw["date"].isna().any():
        raise MarketDataError("One or more CryptoCompare timestamps are invalid.")

    numeric_columns = ["open", "high", "low", "close", "volumefrom", "volumeto"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    values = raw[numeric_columns].to_numpy(dtype=float)
    if raw[numeric_columns].isna().any().any() or not np.isfinite(values).all():
        raise MarketDataError("OHLCV data contains NaN, non-numeric, or infinite values.")

    duplicate_mask = raw.duplicated(subset=["date"], keep=False)
    duplicate_rows_removed = 0
    if duplicate_mask.any():
        for timestamp, group in raw.loc[duplicate_mask].groupby("date", sort=False):
            if len(group[numeric_columns].drop_duplicates()) != 1:
                raise MarketDataError(
                    f"Conflicting OHLCV values found for duplicate timestamp {timestamp.isoformat()}."
                )
        before = len(raw)
        raw = raw.drop_duplicates(subset=["date"], keep="first")
        duplicate_rows_removed = before - len(raw)

    raw = raw.loc[(raw["date"] >= start_utc) & (raw["date"] <= end_utc)].copy()
    if raw.empty:
        raise MarketDataError("No OHLCV rows remain inside the requested UTC range.")

    raw = raw.sort_values("date").reset_index(drop=True)
    if raw["date"].min() < start_utc or raw["date"].max() > end_utc:
        raise MarketDataError("OHLCV timestamps remain outside the requested UTC range.")
    if raw["date"].duplicated().any():
        raise MarketDataError("Duplicate timestamps remain after deduplication.")
    if not raw["date"].is_monotonic_increasing:
        raise MarketDataError("OHLCV timestamps are not sorted in ascending order.")
    if raw[list(REQUIRED_API_FIELDS) + ["date"]].isna().any().any():
        raise MarketDataError("Required OHLCV fields contain missing values after filtering.")

    price_columns = ["open", "high", "low", "close"]
    if (raw[price_columns] <= 0).any().any():
        raise MarketDataError("OHLC prices must be strictly positive.")
    if (raw[["volumefrom", "volumeto"]] < 0).any().any():
        raise MarketDataError("Volume values must be non-negative.")

    violation_labels = {
        "high_lt_open": "high < open",
        "high_lt_close": "high < close",
        "low_gt_open": "low > open",
        "low_gt_close": "low > close",
        "high_lt_low": "high < low",
    }
    violations = pd.DataFrame(
        {
            "high_lt_open": raw["high"] < raw["open"],
            "high_lt_close": raw["high"] < raw["close"],
            "low_gt_open": raw["low"] > raw["open"],
            "low_gt_close": raw["low"] > raw["close"],
            "high_lt_low": raw["high"] < raw["low"],
        },
        index=raw.index,
    )
    invalid_mask = violations.any(axis=1)
    violation_counts = {
        column: int(violations[column].sum()) for column in violations.columns
    }
    ohlc_report: dict[str, Any] = {
        "policy": ohlc_consistency_policy,
        "row_count": 0,
        "min_timestamp": None,
        "max_timestamp": None,
        "violation_counts": violation_counts,
        "rows_sample": [],
    }
    if invalid_mask.any():
        invalid = raw.loc[invalid_mask, ["date", "open", "high", "low", "close"]].copy()
        invalid["violations"] = violations.loc[invalid_mask].apply(
            lambda row: ", ".join(
                violation_labels[column] for column in row.index[row]
            ),
            axis=1,
        )
        sample = [
            {
                "date": record.date.isoformat(),
                "open": float(record.open),
                "high": float(record.high),
                "low": float(record.low),
                "close": float(record.close),
                "violations": record.violations,
            }
            for record in invalid.head(20).itertuples(index=False)
        ]
        ohlc_report.update(
            {
                "row_count": len(invalid),
                "min_timestamp": invalid["date"].min().isoformat(),
                "max_timestamp": invalid["date"].max().isoformat(),
                "rows_sample": sample,
            }
        )
        preview = invalid.head(20).to_string(index=False)
        if ohlc_consistency_policy == "error":
            raise MarketDataError(
                "OHLC high/low consistency validation failed. "
                f"invalid_row_count={len(invalid)}; "
                f"invalid_min_utc={invalid['date'].min().isoformat()}; "
                f"invalid_max_utc={invalid['date'].max().isoformat()}; "
                "invalid_rows_first_20:\n"
                f"{preview}"
            )
        (logger or logging.getLogger(__name__)).warning(
            "%d OHLC consistency violations found in CryptoCompare CCCAGG data. "
            "Raw source values are preserved; no rows were dropped or corrected. "
            "violation_counts=%s sample_first_20=\n%s",
            len(invalid),
            violation_counts,
            preview,
        )

    intervals = raw["date"].diff().dropna()
    irregular = intervals[intervals != pd.Timedelta(hours=1)]
    estimated_missing_hours = sum(
        max(0, math.floor(delta / pd.Timedelta(hours=1)) - 1)
        for delta in irregular
        if delta > pd.Timedelta(hours=1)
    )

    output = raw.rename(
        columns={"volumefrom": "volume", "volumeto": "volume_quote"}
    )[["date", "open", "high", "low", "close", "volume", "volume_quote"]]
    output[["open", "high", "low", "close", "volume", "volume_quote"]] = output[
        ["open", "high", "low", "close", "volume", "volume_quote"]
    ].astype(float)
    return (
        output,
        duplicate_rows_removed,
        len(irregular),
        estimated_missing_hours,
        ohlc_report,
    )


def fetch_cryptocompare_hourly(
    *,
    asset: str,
    quote: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    limit: int = 2000,
    timeout_seconds: float = 30,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2,
    page_delay_seconds: float = 1,
    ohlc_consistency_policy: str = "error",
    api_key: str | None = None,
    logger: logging.Logger | None = None,
) -> FetchResult:
    """Fetch an inclusive UTC interval using bounded backward pagination."""
    if end_utc <= start_utc:
        raise ValueError("end must be later than start.")
    if not 1 <= limit <= 2000:
        raise ValueError("CryptoCompare histohour limit must be between 1 and 2000.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if ohlc_consistency_policy not in {"error", "warn"}:
        raise ValueError(
            "ohlc_consistency_policy must be either 'error' or 'warn'."
        )

    log = logger or logging.getLogger(__name__)
    expected_hours = math.ceil((end_utc - start_utc) / pd.Timedelta(hours=1)) + 1
    max_requests = math.ceil(expected_hours / limit) + 2
    all_rows: list[dict[str, Any]] = []
    to_timestamp = int(end_utc.timestamp())
    previous_oldest: int | None = None
    request_count = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "crypto-stock-prediction/ohlcv-fetcher"})
        while to_timestamp >= int(start_utc.timestamp()):
            if request_count >= max_requests:
                raise MarketDataError(
                    f"Pagination exceeded the safety limit of {max_requests} requests."
                )
            request_count += 1
            params: dict[str, Any] = {
                "fsym": asset.upper(),
                "tsym": quote.upper(),
                "e": "CCCAGG",
                "aggregate": 1,
                "UTCHourDiff": 0,
                "limit": limit,
                "toTs": to_timestamp,
                "extraParams": "crypto-stock-prediction",
            }
            if api_key:
                params["api_key"] = api_key

            payload = _request_page(
                session,
                params=params,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                request_number=request_count,
                logger=log,
            )
            batch = payload.get("Data", {}).get("Data")
            if not isinstance(batch, list):
                raise MarketDataError("CryptoCompare response has an invalid Data.Data field.")
            if not batch:
                raise MarketDataError(
                    f"CryptoCompare returned an empty page on request {request_count}."
                )
            if any(not isinstance(row, dict) or "time" not in row for row in batch):
                raise MarketDataError("CryptoCompare page contains rows without timestamps.")

            timestamps = [int(row["time"]) for row in batch]
            oldest = min(timestamps)
            newest = max(timestamps)
            if previous_oldest is not None and oldest >= previous_oldest:
                raise MarketDataError(
                    "CryptoCompare pagination made no backward progress."
                )
            previous_oldest = oldest
            all_rows.extend(batch)
            unique_count = len({int(row["time"]) for row in all_rows})
            log.info(
                "request=%d requested_to_utc=%s received_rows=%d "
                "response_min_utc=%s response_max_utc=%s accumulated_unique_rows=%d",
                request_count,
                pd.Timestamp(to_timestamp, unit="s", tz="UTC").isoformat(),
                len(batch),
                pd.Timestamp(oldest, unit="s", tz="UTC").isoformat(),
                pd.Timestamp(newest, unit="s", tz="UTC").isoformat(),
                unique_count,
            )

            if oldest <= int(start_utc.timestamp()):
                break
            next_to_timestamp = oldest - 1
            if next_to_timestamp >= to_timestamp:
                raise MarketDataError("CryptoCompare pagination cursor did not decrease.")
            to_timestamp = next_to_timestamp
            if page_delay_seconds > 0:
                time.sleep(page_delay_seconds)

    frame, duplicates, irregular, missing_hours, ohlc_report = _normalize_and_validate(
        all_rows,
        start_utc=start_utc,
        end_utc=end_utc,
        ohlc_consistency_policy=ohlc_consistency_policy,
        logger=log,
    )
    return FetchResult(
        frame=frame,
        request_count=request_count,
        duplicate_rows_removed=duplicates,
        irregular_interval_count=irregular,
        estimated_missing_hours=missing_hours,
        ohlc_consistency_policy=ohlc_report["policy"],
        invalid_ohlc_row_count=ohlc_report["row_count"],
        invalid_ohlc_min_timestamp=ohlc_report["min_timestamp"],
        invalid_ohlc_max_timestamp=ohlc_report["max_timestamp"],
        invalid_ohlc_violation_counts=ohlc_report["violation_counts"],
        invalid_ohlc_rows_sample=ohlc_report["rows_sample"],
    )


def atomic_save_frame(frame: pd.DataFrame, output_path: Path, *, overwrite: bool) -> None:
    """Write CSV or Parquet, verify it, then atomically replace the final path."""
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    suffix = output_path.suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        raise ValueError("Output must use .parquet or .csv.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp{suffix}")
    try:
        if suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
            verified = pd.read_parquet(temporary)
        else:
            frame.to_csv(temporary, index=False)
            verified = pd.read_csv(temporary)
        if len(verified) != len(frame) or list(verified.columns) != list(frame.columns):
            raise MarketDataError("Saved OHLCV verification failed.")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
