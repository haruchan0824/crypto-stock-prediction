"""Deterministic LightGBM training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class TrainedFold:
    booster: lgb.Booster
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    profile: dict[str, Any]


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    decision_threshold: float,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= decision_threshold).astype(np.int8)
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
    }


def train_fold(
    prepared: Any,
    *,
    feature_columns: list[str],
    fold_number: int,
    parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    decision_threshold: float,
) -> TrainedFold:
    """Fit on train, early-stop on validation, then evaluate test once."""
    train_set = lgb.Dataset(
        prepared.X_train,
        label=prepared.y_train,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        prepared.X_validation,
        label=prepared.y_validation,
        reference=train_set,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = int(booster.best_iteration)
    train_probability = booster.predict(prepared.X_train, num_iteration=best_iteration)
    validation_probability = booster.predict(
        prepared.X_validation, num_iteration=best_iteration
    )
    # Test is only accessed after fitting and model selection are complete.
    test_probability = booster.predict(prepared.X_test, num_iteration=best_iteration)
    baseline_probability = float(prepared.y_train.mean())
    test_baseline = np.full(len(prepared.y_test), baseline_probability, dtype=float)

    metrics = {
        "fold": fold_number,
        "best_iteration": best_iteration,
        "decision_threshold": decision_threshold,
        "label_threshold": float(prepared.label_threshold),
        "model": {
            "train": classification_metrics(
                prepared.y_train,
                train_probability,
                decision_threshold=decision_threshold,
            ),
            "validation": classification_metrics(
                prepared.y_validation,
                validation_probability,
                decision_threshold=decision_threshold,
            ),
            "test": classification_metrics(
                prepared.y_test,
                test_probability,
                decision_threshold=decision_threshold,
            ),
        },
        "constant_probability_baseline": {
            "probability": baseline_probability,
            "source": "train_positive_rate",
            "test": classification_metrics(
                prepared.y_test,
                test_baseline,
                decision_threshold=decision_threshold,
            ),
        },
    }

    prediction_parts: list[pd.DataFrame] = []
    for split, frame, probability in (
        ("train", prepared.train, train_probability),
        ("validation", prepared.validation, validation_probability),
        ("test", prepared.test, test_probability),
    ):
        part = frame[["date", "label_timestamp", "forward_return_6h", "label"]].copy()
        part.insert(0, "fold", fold_number)
        part.insert(1, "split", split)
        part["probability"] = probability
        part["prediction"] = (probability >= decision_threshold).astype(np.int8)
        part["baseline_probability"] = baseline_probability
        prediction_parts.append(part)

    importance = pd.DataFrame(
        {
            "fold": fold_number,
            "feature": feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "feature"], ascending=[False, True])

    profile = {
        "fold": fold_number,
        "best_iteration": best_iteration,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "early_stopping_dataset": "validation",
        "test_role": "final_evaluation_only",
        "decision_threshold": decision_threshold,
        "decision_threshold_source": "fixed_configuration",
        "label_threshold_source": "train_forward_returns_only",
        "parameters": parameters,
        "counts": prepared.profile["counts"],
    }
    return TrainedFold(
        booster=booster,
        metrics=metrics,
        predictions=pd.concat(prediction_parts, ignore_index=True),
        feature_importance=importance,
        profile=profile,
    )
