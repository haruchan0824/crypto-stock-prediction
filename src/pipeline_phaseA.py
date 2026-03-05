#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COMPARE_OUTPUT = "tft_vs_lgbm_compare.csv"
SUMMARY_OUTPUT = "tft_vs_lgbm_summary.json"
LOFO_GROUP_OUTPUT = "lofo_group_agg.csv"
LOFO_FEATURE_OUTPUT = "lofo_feature_agg.csv"


def resolve_run_output_dir(run_output_dir: Path | None = None) -> Path:
    env_dir = os.environ.get("RUN_OUTPUT_DIR")
    resolved = Path(env_dir) if env_dir else (run_output_dir or Path("reports/raw/manual_run"))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main(config_path: str = "configs/pipeline_config.json", run_output_dir: Path | None = None) -> None:
    out_dir = resolve_run_output_dir(run_output_dir)

    compare_df = pd.DataFrame(
        [
            {
                "fold": 0,
                "lgbm_auc": 0.0,
                "tft_auc": 0.0,
                "delta_auc": 0.0,
                "tft_win": False,
                "status": "TODO",
            }
        ]
    )
    lofo_group_df = pd.DataFrame(
        [
            {
                "group": "TODO_GROUP",
                "delta_auc": 0.0,
                "delta_auc_mean": 0.0,
                "status": "TODO",
            }
        ]
    )
    lofo_feature_df = pd.DataFrame(
        [
            {
                "feature": "TODO_FEATURE",
                "delta_auc": 0.0,
                "delta_auc_mean": 0.0,
                "status": "TODO",
            }
        ]
    )
    summary = {
        "status": "TODO",
        "reason": "PhaseA placeholder artifacts generated (no training/import of notebook-derived modules).",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path)),
        "run_output_dir": str(out_dir),
    }

    compare_df.to_csv(out_dir / COMPARE_OUTPUT, index=False)
    (out_dir / SUMMARY_OUTPUT).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lofo_group_df.to_csv(out_dir / LOFO_GROUP_OUTPUT, index=False)
    lofo_feature_df.to_csv(out_dir / LOFO_FEATURE_OUTPUT, index=False)

    print(f"[INFO] Saved: {out_dir / COMPARE_OUTPUT}")
    print(f"[INFO] Saved: {out_dir / SUMMARY_OUTPUT}")
    print(f"[INFO] Saved: {out_dir / LOFO_GROUP_OUTPUT}")
    print(f"[INFO] Saved: {out_dir / LOFO_FEATURE_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PhaseA placeholder artifacts")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.json")
    parser.add_argument("--run_output_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, run_output_dir=args.run_output_dir)
