"""Local OHLCV loading and validation for the reproducible pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
VALUE_COLUMNS = (*PRICE_COLUMNS, "volume")

_COLUMN_ALIASES = {
    "date": ("date", "datetime", "timestamp", "time", "open_time"),
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "volume": ("volume", "volume_eth", "base_volume", "volumefrom"),
}

_TIMESTAMP_UNIT_LIMITS = {
    "s": (10**8, 10**11),
    "ms": (10**11, 10**14),
    "us": (10**14, 10**17),
    "ns": (10**17, 10**20),
}


def _read_table(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), "csv"
    if suffix == ".parquet":
        return pd.read_parquet(path), "parquet"
    raise ValueError(
        f"Unsupported input format '{path.suffix}'. Expected .csv or .parquet."
    )


def _normalize_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    original_columns = [str(column) for column in df.columns]
    lowered: dict[str, list[str]] = {}
    for column in original_columns:
        lowered.setdefault(column.casefold(), []).append(column)

    rename_map: dict[str, str] = {}
    source_columns: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        matches: list[str] = []
        for alias in aliases:
            matches.extend(lowered.get(alias.casefold(), []))
        matches = list(dict.fromkeys(matches))
        if not matches:
            raise ValueError(
                f"Missing required OHLCV column '{canonical}'. "
                f"Accepted names: {', '.join(aliases)}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous columns for '{canonical}': {matches}. "
                "Remove or explicitly rename one of them."
            )
        source = matches[0]
        source_columns[canonical] = source
        if source != canonical:
            rename_map[source] = canonical

    normalized = df.rename(columns=rename_map).loc[:, list(CANONICAL_COLUMNS)].copy()
    return normalized, rename_map, source_columns["date"]


def _infer_timestamp_unit(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="raise")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        raise ValueError("Cannot infer timestamp unit from an empty numeric column.")

    magnitude = float(np.median(np.abs(finite.to_numpy(dtype=np.float64))))
    matches = [
        unit
        for unit, (lower, upper) in _TIMESTAMP_UNIT_LIMITS.items()
        if lower <= magnitude < upper
    ]
    if len(matches) != 1:
        raise ValueError(
            "Timestamp unit is ambiguous. Specify --timestamp-unit "
            "with one of: s, ms, us, ns."
        )
    return matches[0]


def _strings_have_explicit_timezone(values: pd.Series) -> bool:
    samples = values.dropna().astype(str).str.strip()
    if samples.empty:
        return False
    timezone_pattern = r"(?:Z|[+-]\d{2}:?\d{2})$"
    return bool(samples.str.contains(timezone_pattern, regex=True).all())


def _parse_datetime_column(
    values: pd.Series,
    *,
    timestamp_unit: str | None,
) -> tuple[pd.Series, dict[str, Any]]:
    is_numeric = pd.api.types.is_numeric_dtype(values)
    resolved_unit: str | None = None
    unit_was_inferred = False
    assumed_timezone: str | None = None

    if timestamp_unit is not None:
        if timestamp_unit not in _TIMESTAMP_UNIT_LIMITS:
            raise ValueError("timestamp_unit must be one of: s, ms, us, ns.")
        numeric = pd.to_numeric(values, errors="raise")
        parsed = pd.to_datetime(numeric, unit=timestamp_unit, utc=True, errors="raise")
        resolved_unit = timestamp_unit
    elif is_numeric:
        resolved_unit = _infer_timestamp_unit(values)
        unit_was_inferred = True
        parsed = pd.to_datetime(
            pd.to_numeric(values, errors="raise"),
            unit=resolved_unit,
            utc=True,
            errors="raise",
        )
    else:
        if not _strings_have_explicit_timezone(values):
            assumed_timezone = "UTC"
        parsed = pd.to_datetime(values, utc=True, errors="raise")

    parsed_series = pd.Series(parsed, index=values.index, name="date")
    if parsed_series.isna().any():
        raise ValueError("The timestamp column contains missing or unparseable values.")

    details = {
        "timestamp_unit": resolved_unit,
        "timestamp_unit_inferred": unit_was_inferred,
        "assumed_timezone": assumed_timezone,
    }
    return parsed_series, details


def _compute_file_fingerprint(path: Path, file_format: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = path.stat()
    modified = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
    return {
        "absolute_path": str(path.resolve()),
        "format": file_format,
        "size_bytes": int(stat.st_size),
        "modified_at_utc": modified.isoformat(),
        "sha256": digest.hexdigest(),
    }


def _time_grid_profile(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    intervals = df["date"].diff().dropna()
    minimum = intervals.min() if not intervals.empty else None
    maximum = intervals.max() if not intervals.empty else None

    irregular_count = 0
    estimated_missing = 0
    if timeframe == "1h" and not intervals.empty:
        expected = pd.Timedelta(hours=1)
        irregular = intervals != expected
        irregular_count = int(irregular.sum())
        positive_gaps = intervals[intervals > expected]
        estimated_missing = int(
            sum(max(int(gap / expected) - 1, 0) for gap in positive_gaps)
        )

    return {
        "irregular_interval_count": irregular_count,
        "estimated_missing_hours": estimated_missing,
        "minimum_interval": str(minimum) if minimum is not None else None,
        "maximum_interval": str(maximum) if maximum is not None else None,
    }


def _validate_ohlcv(df: pd.DataFrame) -> dict[str, Any]:
    converted = df.copy()
    for column in VALUE_COLUMNS:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")

    missing = {
        column: int(converted[column].isna().sum())
        for column in CANONICAL_COLUMNS
    }
    infinite = {
        column: int(np.isinf(converted[column].to_numpy(dtype=np.float64)).sum())
        for column in VALUE_COLUMNS
    }

    non_positive_price = (converted[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    negative_volume = converted["volume"] < 0
    invalid_ohlc = (
        (converted["high"] < converted["open"])
        | (converted["high"] < converted["close"])
        | (converted["high"] < converted["low"])
        | (converted["low"] > converted["open"])
        | (converted["low"] > converted["close"])
    )

    quality = {
        "missing_values_by_column": missing,
        "infinite_values_by_column": infinite,
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "non_positive_price_rows": int(non_positive_price.sum()),
        "negative_volume_rows": int(negative_volume.sum()),
    }

    problems = []
    if any(missing.values()):
        problems.append(f"missing values={missing}")
    if any(infinite.values()):
        problems.append(f"infinite values={infinite}")
    if quality["invalid_ohlc_rows"]:
        problems.append(f"invalid OHLC rows={quality['invalid_ohlc_rows']}")
    if quality["non_positive_price_rows"]:
        problems.append(
            f"non-positive price rows={quality['non_positive_price_rows']}"
        )
    if quality["negative_volume_rows"]:
        problems.append(f"negative volume rows={quality['negative_volume_rows']}")
    if problems:
        raise ValueError("OHLCV validation failed: " + "; ".join(problems))

    for column in VALUE_COLUMNS:
        converted[column] = converted[column].astype(float)
    return {"data": converted, "quality": quality}


def load_local_ohlcv(
    path: Path,
    *,
    max_rows: int | None,
    timeframe: str,
    timestamp_unit: str | None = None,
    strict_time_grid: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and strictly validate a local CSV or Parquet OHLCV file."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Local OHLCV input file not found: {input_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer or null.")
    if timeframe != "1h":
        raise ValueError("This pipeline phase currently supports timeframe='1h' only.")

    raw_df, file_format = _read_table(input_path)
    if raw_df.empty:
        raise ValueError(f"Input file contains no rows: {input_path}")

    original_columns = [str(column) for column in raw_df.columns]
    original_rows = int(len(raw_df))
    normalized, rename_map, timestamp_source = _normalize_column_names(raw_df)
    parsed_dates, datetime_details = _parse_datetime_column(
        normalized["date"],
        timestamp_unit=timestamp_unit,
    )
    normalized["date"] = parsed_dates

    was_sorted = bool(normalized["date"].is_monotonic_increasing)
    normalized = normalized.sort_values("date", kind="mergesort").reset_index(drop=True)
    duplicate_count = int(normalized["date"].duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"Duplicate timestamps detected ({duplicate_count} rows). "
            "Automatic aggregation is disabled."
        )

    validation = _validate_ohlcv(normalized)
    validated = validation["data"]
    after_validation = int(len(validated))
    if max_rows is not None:
        validated = validated.tail(max_rows).reset_index(drop=True)
    after_max_rows = int(len(validated))

    grid = _time_grid_profile(validated, timeframe)
    if grid["irregular_interval_count"] and strict_time_grid:
        raise ValueError(
            "Irregular 1-hour timestamp intervals detected: "
            f"{grid['irregular_interval_count']}."
        )

    profile = {
        "input_file": _compute_file_fingerprint(input_path, file_format),
        "rows": {
            "original": original_rows,
            "after_validation": after_validation,
            "after_max_rows": after_max_rows,
        },
        "columns": {
            "original": original_columns,
            "renamed": rename_map,
            "final": list(validated.columns),
        },
        "time": {
            "timestamp_source_column": timestamp_source,
            **datetime_details,
            "was_sorted": was_sorted,
            "min_timestamp": validated["date"].min().isoformat(),
            "max_timestamp": validated["date"].max().isoformat(),
            "duplicate_count": duplicate_count,
            **grid,
        },
        "quality": validation["quality"],
    }
    return validated, profile
