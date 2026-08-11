"""Leakage-aware preparation for the local LightGBM experiment.

This phase deliberately stops before importing or training LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


FEATURE_IMPLEMENTATION = "pandas_v1"
FORWARD_RETURN_COLUMN = "forward_return_6h"

FEATURE_COLUMNS = [
    "open",
    "close",
    "volume",
    "ret_1h",
    "ret_6h",
    "ret_12h",
    "log_ret_1h",
    "ma_6h",
    "ma_24h",
    "ma_72h",
    "ma_168h",
    "close_to_ma_6h",
    "close_to_ma_24h",
    "close_to_ma_72h",
    "close_to_ma_168h",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "return_std_6h",
    "return_std_24h",
    "return_std_72h",
    "volume_pchg_1h",
    "volume_pchg_6h",
    "volume_mean_6h",
    "volume_mean_24h",
    "volume_ratio_6h",
    "volume_ratio_24h",
]


@dataclass(frozen=True)
class FoldPreparationResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label_threshold: float
    profile: dict[str, Any]


def build_basic_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build right-aligned, past-and-present-only pandas features."""
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature generation: {missing}")
    if df.empty:
        raise ValueError("Cannot generate features from an empty DataFrame.")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Feature input timestamps must be sorted ascending.")
    if df["date"].duplicated().any():
        raise ValueError("Feature input timestamps must be unique.")

    featured = df.copy()
    close = featured["close"].astype(float)
    volume = featured["volume"].astype(float)

    featured["ret_1h"] = close.pct_change(1, fill_method=None)
    featured["ret_6h"] = close.pct_change(6, fill_method=None)
    featured["ret_12h"] = close.pct_change(12, fill_method=None)
    featured["log_ret_1h"] = np.log(close).diff(1)

    for hours in (6, 24, 72, 168):
        moving_average = close.rolling(window=hours, min_periods=hours).mean()
        featured[f"ma_{hours}h"] = moving_average
        featured[f"close_to_ma_{hours}h"] = close / moving_average - 1.0

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss
    featured["rsi_14"] = 100.0 - (100.0 / (1.0 + relative_strength))
    featured.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100.0
    featured.loc[(average_loss == 0) & (average_gain == 0), "rsi_14"] = 50.0

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    featured["macd"] = ema_12 - ema_26
    featured["macd_signal"] = featured["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    featured["macd_hist"] = featured["macd"] - featured["macd_signal"]

    for hours in (6, 24, 72):
        featured[f"return_std_{hours}h"] = featured["ret_1h"].rolling(
            window=hours, min_periods=hours
        ).std()

    featured["volume_pchg_1h"] = volume.pct_change(1, fill_method=None)
    featured["volume_pchg_6h"] = volume.pct_change(6, fill_method=None)
    for hours in (6, 24):
        volume_mean = volume.rolling(window=hours, min_periods=hours).mean()
        featured[f"volume_mean_{hours}h"] = volume_mean
        featured[f"volume_ratio_{hours}h"] = volume / volume_mean

    original_rows = len(featured)
    featured[list(FEATURE_COLUMNS)] = featured[list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    nan_before_drop = {
        column: int(featured[column].isna().sum()) for column in FEATURE_COLUMNS
    }
    featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    finite = np.isfinite(featured[FEATURE_COLUMNS].to_numpy(dtype=float))
    if not finite.all():
        raise ValueError("Non-finite feature values remain after warm-up filtering.")

    profile = {
        "feature_implementation": FEATURE_IMPLEMENTATION,
        "rsi_implementation": "Wilder-style EMA (alpha=1/14, adjust=False)",
        "macd_implementation": "EMA12 - EMA26; signal EMA9; adjust=False",
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "original_rows": original_rows,
        "rows_after_warmup": len(featured),
        "dropped_rows": original_rows - len(featured),
        "nan_counts_before_warmup_drop": nan_before_drop,
        "nan_count_after_warmup_drop": int(
            featured[FEATURE_COLUMNS].isna().sum().sum()
        ),
        "inf_count_after_warmup_drop": int((~finite).sum()),
        "maximum_lookback_hours": 168,
        "raw_high_low_preserved": True,
        "high_low_features_excluded_reason": (
            "CryptoCompare CCCAGG contains known raw high/low consistency "
            "violations; raw columns are preserved but excluded from the initial "
            "training feature set. ATR and other high/low-derived features are omitted."
        ),
        "volume_quote_excluded_reason": (
            "The canonical local loader currently returns base volume only; no "
            "quote-volume feature is used in this initial preparation stage."
        ),
        "causality": {
            "rolling_alignment": "right",
            "centered_rolling": False,
            "feature_negative_shifts": False,
            "backfill": False,
            "global_scaling": False,
            "global_quantiles": False,
            "winsorization": False,
        },
    }
    return featured, list(FEATURE_COLUMNS), profile


def add_forward_return(
    df: pd.DataFrame,
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    """Match close at exactly t + horizon using timestamps, never row offsets."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")
    if df["date"].duplicated().any():
        raise ValueError("Forward-return input timestamps must be unique.")
    result = df.copy()
    result["label_timestamp"] = result["date"] + pd.Timedelta(hours=horizon_hours)
    close_by_timestamp = result.set_index("date")["close"]
    future_close = result["label_timestamp"].map(close_by_timestamp)
    column = f"forward_return_{horizon_hours}h"
    result[column] = future_close.to_numpy(dtype=float) / result["close"].to_numpy(
        dtype=float
    ) - 1.0
    return result


def make_timestamp_folds(
    timestamps: pd.Series,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    max_folds: int,
) -> pd.DataFrame:
    """Create fixed rolling half-open folds from timestamps only."""
    if min(train_days, validation_days, test_days, max_folds) <= 0:
        raise ValueError("Fold durations and max_folds must be positive.")
    parsed = pd.Series(pd.to_datetime(timestamps, utc=True)).dropna().sort_values()
    if parsed.empty:
        raise ValueError("Cannot create folds from empty timestamps.")
    if parsed.duplicated().any():
        raise ValueError("Fold timestamps must be unique.")

    first = parsed.iloc[0]
    available_end = parsed.iloc[-1] + pd.Timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    for fold_number in range(max_folds):
        train_start = first + pd.Timedelta(days=fold_number * test_days)
        train_end = train_start + pd.Timedelta(days=train_days)
        validation_start = train_end
        validation_end = validation_start + pd.Timedelta(days=validation_days)
        test_start = validation_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > available_end:
            break
        rows.append(
            {
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    if not rows:
        required_days = train_days + validation_days + test_days
        raise ValueError(
            "Insufficient timestamp coverage for one fold: "
            f"required_days={required_days}, available_start={first.isoformat()}, "
            f"available_end={available_end.isoformat()}."
        )
    return pd.DataFrame(rows)


def fit_positive_threshold(
    train_forward_returns: pd.Series,
    *,
    positive_rate: float,
) -> float:
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be between 0 and 1.")
    values = pd.to_numeric(train_forward_returns, errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=float))]
    if values.empty:
        raise ValueError("No finite train forward returns are available.")
    return float(values.quantile(1.0 - positive_rate))


def _split_summary(frame: pd.DataFrame, *, purged_rows: int) -> dict[str, Any]:
    returns = frame[FORWARD_RETURN_COLUMN]
    labels = frame["label"]
    features = frame.drop(
        columns=[
            column
            for column in ("date", "label_timestamp", FORWARD_RETURN_COLUMN, "label")
            if column in frame.columns
        ]
    )
    finite = np.isfinite(features.to_numpy(dtype=float))
    return {
        "start": frame["date"].min().isoformat(),
        "end": frame["date"].max().isoformat(),
        "rows": len(frame),
        "positive_rows": int(labels.sum()),
        "negative_rows": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()),
        "forward_return": {
            "min": float(returns.min()),
            "max": float(returns.max()),
            "mean": float(returns.mean()),
            "median": float(returns.median()),
            "std": float(returns.std()),
        },
        "purged_rows": purged_rows,
        "label_timestamp_min": frame["label_timestamp"].min().isoformat(),
        "label_timestamp_max": frame["label_timestamp"].max().isoformat(),
        "feature_nan_count": int(features.isna().sum().sum()),
        "feature_inf_count": int((~finite).sum()),
    }


def prepare_fold_data(
    df: pd.DataFrame,
    fold: Mapping[str, Any],
    *,
    feature_columns: list[str],
    horizon_hours: int,
    positive_rate: float,
) -> FoldPreparationResult:
    forward_column = f"forward_return_{horizon_hours}h"
    if forward_column != FORWARD_RETURN_COLUMN:
        raise ValueError("This preparation phase currently supports horizon_hours=6 only.")
    forbidden = {forward_column, "label", "label_timestamp"}.intersection(
        feature_columns
    )
    if forbidden:
        raise ValueError(f"Future or label columns found in features: {sorted(forbidden)}")
    missing_features = sorted(set(feature_columns).difference(df.columns))
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    boundaries = {
        key: pd.Timestamp(fold[key])
        for key in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        )
    }
    if not (
        boundaries["train_start"] < boundaries["train_end"]
        == boundaries["validation_start"] < boundaries["validation_end"]
        == boundaries["test_start"] < boundaries["test_end"]
    ):
        raise ValueError("Fold boundaries are not ordered, contiguous half-open intervals.")

    train_before = df.loc[
        (df["date"] >= boundaries["train_start"])
        & (df["date"] < boundaries["train_end"])
    ].copy()
    validation_before = df.loc[
        (df["date"] >= boundaries["validation_start"])
        & (df["date"] < boundaries["validation_end"])
    ].copy()
    test_before = df.loc[
        (df["date"] >= boundaries["test_start"])
        & (df["date"] < boundaries["test_end"])
    ].copy()

    train = train_before.loc[
        train_before[forward_column].notna()
        & (train_before["label_timestamp"] < boundaries["train_end"])
    ].copy()
    validation = validation_before.loc[
        validation_before[forward_column].notna()
        & (validation_before["label_timestamp"] < boundaries["validation_end"])
    ].copy()
    test = test_before.loc[
        test_before[forward_column].notna()
        & (test_before["label_timestamp"] < boundaries["test_end"])
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more prepared splits are empty.")
    threshold = fit_positive_threshold(
        train[forward_column], positive_rate=positive_rate
    )
    for split in (train, validation, test):
        split["label"] = (split[forward_column] >= threshold).astype("int8")

    split_frames = {"train": train, "validation": validation, "test": test}
    for name, split in split_frames.items():
        if not split["date"].is_monotonic_increasing:
            raise ValueError(f"{name} timestamps are not sorted ascending.")
        if split[feature_columns].isna().any().any():
            raise ValueError(f"{name} features contain NaN values.")
        if not np.isfinite(split[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} features contain infinite values.")
        if split["label"].isna().any():
            raise ValueError(f"{name} labels contain NaN values.")
        if set(split["label"].unique()) != {0, 1}:
            raise ValueError(f"{name} does not contain both label classes.")

    if train["label_timestamp"].max() >= boundaries["train_end"]:
        raise ValueError("Train purge failed at train_end.")
    if validation["label_timestamp"].max() >= boundaries["validation_end"]:
        raise ValueError("Validation purge failed at validation_end.")
    if test["label_timestamp"].max() >= boundaries["test_end"]:
        raise ValueError("Test purge failed at test_end.")
    if set(train["date"]).intersection(validation["date"]):
        raise ValueError("Train and validation timestamps overlap.")
    if set(validation["date"]).intersection(test["date"]):
        raise ValueError("Validation and test timestamps overlap.")

    purged_train = len(train_before) - len(train)
    purged_validation = len(validation_before) - len(validation)
    filtered_test = len(test_before) - len(test)
    profile = {
        "fold": int(fold["fold"]),
        "label_threshold": threshold,
        "positive_rate_target": positive_rate,
        "counts": {
            "n_train_before_purge": len(train_before),
            "n_train": len(train),
            "n_validation_before_purge": len(validation_before),
            "n_validation": len(validation),
            "n_test_before_filter": len(test_before),
            "n_test": len(test),
            "purged_train_rows": purged_train,
            "purged_validation_rows": purged_validation,
            "filtered_test_rows": filtered_test,
        },
        "splits": {
            "train": _split_summary(train[["date", "label_timestamp", forward_column, "label", *feature_columns]], purged_rows=purged_train),
            "validation": _split_summary(validation[["date", "label_timestamp", forward_column, "label", *feature_columns]], purged_rows=purged_validation),
            "test": _split_summary(test[["date", "label_timestamp", forward_column, "label", *feature_columns]], purged_rows=filtered_test),
        },
    }

    return FoldPreparationResult(
        X_train=train[feature_columns].copy(),
        y_train=train["label"].copy(),
        X_validation=validation[feature_columns].copy(),
        y_validation=validation["label"].copy(),
        X_test=test[feature_columns].copy(),
        y_test=test["label"].copy(),
        train=train,
        validation=validation,
        test=test,
        label_threshold=threshold,
        profile=profile,
    )


def make_prepared_sample(
    result: FoldPreparationResult,
    *,
    feature_columns: list[str],
    rows_per_edge: int = 50,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = [
        "date",
        "label_timestamp",
        "close",
        FORWARD_RETURN_COLUMN,
        "label",
        *feature_columns,
    ]
    for name, frame in (
        ("train", result.train),
        ("validation", result.validation),
        ("test", result.test),
    ):
        sample = pd.concat(
            [frame.head(rows_per_edge), frame.tail(rows_per_edge)],
            ignore_index=True,
        ).drop_duplicates(subset=["date"], keep="first")
        sample = sample.loc[:, list(dict.fromkeys(columns))].copy()
        sample.insert(0, "split", name)
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)
