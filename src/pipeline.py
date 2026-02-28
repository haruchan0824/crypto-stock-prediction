#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


COMPARE_OUTPUT = "tft_vs_lgbm_compare.csv"
SUMMARY_OUTPUT = "tft_vs_lgbm_summary.json"
LOFO_GROUP_OUTPUT = "lofo_group_agg.csv"
LOFO_FEATURE_OUTPUT = "lofo_feature_agg.csv"


class PipelineInputError(RuntimeError):
    pass


@dataclass
class PipelineInputs:
    compare_source_csv: Path | None
    lofo_group_source_csv: Path | None
    lofo_feature_source_csv: Path | None


def read_simple_yaml(path: Path) -> dict[str, Any]:
    """Minimal YAML reader for existing configs/final.yaml style."""
    data: dict[str, Any] = {}
    current_section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and ":" not in line[:-1]:
            current_section = line[:-1]
            data[current_section] = {}
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            if current_section:
                section = data.setdefault(current_section, {})
                section[k] = v
            else:
                data[k] = v
    return data


def resolve_run_output_dir(run_output_dir: Path | None = None) -> Path:
    env_dir = os.environ.get("RUN_OUTPUT_DIR")
    resolved = Path(env_dir) if env_dir else (run_output_dir or Path("reports/raw/manual_run"))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _print_columns(df: pd.DataFrame, label: str) -> None:
    print(f"[INFO] Available columns ({label}): {list(df.columns)}")


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    _print_columns(df, label)
    raise PipelineInputError(f"Missing required columns {candidates} for {label}")


def _build_compare_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    fold_col = _pick_col(raw_df, ["fold", "fold_id", "cv_fold"], "compare")
    lgbm_col = _pick_col(raw_df, ["lgbm_auc", "auc_lgbm"], "compare")
    tft_col = _pick_col(raw_df, ["tft_auc", "auc_tft"], "compare")

    df = raw_df[[fold_col, lgbm_col, tft_col]].copy()
    df = df.rename(columns={fold_col: "fold", lgbm_col: "lgbm_auc", tft_col: "tft_auc"})
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["lgbm_auc"] = pd.to_numeric(df["lgbm_auc"], errors="coerce")
    df["tft_auc"] = pd.to_numeric(df["tft_auc"], errors="coerce")
    df = df.sort_values("fold").reset_index(drop=True)
    return df


def _build_summary(compare_df: pd.DataFrame) -> dict[str, Any]:
    valid = compare_df.dropna(subset=["lgbm_auc", "tft_auc"]).copy()
    delta = valid["tft_auc"] - valid["lgbm_auc"]
    return {
        "num_folds": int(len(valid)),
        "lgbm_auc_mean": float(valid["lgbm_auc"].mean()) if not valid.empty else None,
        "lgbm_auc_std": float(valid["lgbm_auc"].std(ddof=1)) if len(valid) > 1 else 0.0,
        "tft_auc_mean": float(valid["tft_auc"].mean()) if not valid.empty else None,
        "tft_auc_std": float(valid["tft_auc"].std(ddof=1)) if len(valid) > 1 else 0.0,
        "delta_auc_mean": float(delta.mean()) if not valid.empty else None,
        "delta_auc_std": float(delta.std(ddof=1)) if len(valid) > 1 else 0.0,
        "win_folds": int((delta > 0).sum()) if not valid.empty else 0,
        "loss_folds": int((delta <= 0).sum()) if not valid.empty else 0,
    }


def _build_lofo_agg(raw_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    name_candidates = ["group", "group_name"] if kind == "group" else ["feature", "feature_name"]
    name_col = _pick_col(raw_df, name_candidates, f"lofo_{kind}")
    delta_col = _pick_col(raw_df, ["delta_auc", "auc_delta", "diff_auc"], f"lofo_{kind}")

    df = raw_df[[name_col, delta_col]].copy()
    df = df.rename(columns={name_col: kind, delta_col: "delta_auc"})
    df["delta_auc"] = pd.to_numeric(df["delta_auc"], errors="coerce")
    agg = (
        df.groupby(kind, as_index=False)["delta_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "delta_auc_mean", "std": "delta_auc_std", "count": "n"})
    )
    return agg.sort_values("delta_auc_mean", ascending=False)


def _discover_inputs(cfg: dict[str, Any], run_output_dir: Path) -> PipelineInputs:
    # TODO: configs/final.yaml に pipeline.compare_source_csv / pipeline.lofo_*_source_csv を追加して明示化する
    pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline", {}), dict) else {}

    def _path_or_none(key: str, fallback: str) -> Path | None:
        val = pipeline_cfg.get(key)
        candidate = Path(val) if val else (run_output_dir / fallback)
        return candidate if candidate.exists() else None

    return PipelineInputs(
        compare_source_csv=_path_or_none("compare_source_csv", COMPARE_OUTPUT),
        lofo_group_source_csv=_path_or_none("lofo_group_source_csv", LOFO_GROUP_OUTPUT),
        lofo_feature_source_csv=_path_or_none("lofo_feature_source_csv", LOFO_FEATURE_OUTPUT),
    )


def _write_placeholders(run_output_dir: Path, reason: str) -> None:
    print(f"[ERROR] {reason}")
    pd.DataFrame(columns=["fold", "lgbm_auc", "tft_auc", "status"]).assign(status="TODO").to_csv(
        run_output_dir / COMPARE_OUTPUT,
        index=False,
    )
    (run_output_dir / SUMMARY_OUTPUT).write_text(
        json.dumps({"status": "TODO", "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(columns=["group", "delta_auc_mean", "delta_auc_std", "n", "status"]).assign(
        status="TODO"
    ).to_csv(run_output_dir / LOFO_GROUP_OUTPUT, index=False)
    pd.DataFrame(columns=["feature", "delta_auc_mean", "delta_auc_std", "n", "status"]).assign(
        status="TODO"
    ).to_csv(run_output_dir / LOFO_FEATURE_OUTPUT, index=False)


def main(config_path: str = "configs/final.yaml", run_output_dir: Path | None = None) -> None:
    cfg_path = Path(config_path)
    out_dir = resolve_run_output_dir(run_output_dir)
    if not cfg_path.exists():
        _write_placeholders(out_dir, f"Config not found: {cfg_path}")
        raise SystemExit(1)

    cfg = read_simple_yaml(cfg_path)
    inputs = _discover_inputs(cfg, out_dir)

    try:
        if inputs.compare_source_csv is None:
            raise PipelineInputError(
                "compare source CSV not found. TODO: set pipeline.compare_source_csv in configs/final.yaml"
            )
        if inputs.lofo_group_source_csv is None:
            raise PipelineInputError(
                "lofo-group source CSV not found. TODO: set pipeline.lofo_group_source_csv in configs/final.yaml"
            )
        if inputs.lofo_feature_source_csv is None:
            raise PipelineInputError(
                "lofo-feature source CSV not found. TODO: set pipeline.lofo_feature_source_csv in configs/final.yaml"
            )

        compare_raw = pd.read_csv(inputs.compare_source_csv)
        group_raw = pd.read_csv(inputs.lofo_group_source_csv)
        feature_raw = pd.read_csv(inputs.lofo_feature_source_csv)

        compare_df = _build_compare_df(compare_raw)
        summary = _build_summary(compare_df)
        lofo_group_agg = _build_lofo_agg(group_raw, "group")
        lofo_feature_agg = _build_lofo_agg(feature_raw, "feature")

        compare_df.to_csv(out_dir / COMPARE_OUTPUT, index=False)
        (out_dir / SUMMARY_OUTPUT).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lofo_group_agg.to_csv(out_dir / LOFO_GROUP_OUTPUT, index=False)
        lofo_feature_agg.to_csv(out_dir / LOFO_FEATURE_OUTPUT, index=False)

        print(f"[INFO] Saved: {out_dir / COMPARE_OUTPUT}")
        print(f"[INFO] Saved: {out_dir / SUMMARY_OUTPUT}")
        print(f"[INFO] Saved: {out_dir / LOFO_GROUP_OUTPUT}")
        print(f"[INFO] Saved: {out_dir / LOFO_FEATURE_OUTPUT}")
    except (PipelineInputError, KeyError) as e:
        _write_placeholders(out_dir, str(e))
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final pipeline artifacts")
    parser.add_argument("--config", type=str, default="configs/final.yaml")
    parser.add_argument("--run_output_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, run_output_dir=args.run_output_dir)
