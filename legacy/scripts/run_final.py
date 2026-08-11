#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final training/evaluation pipeline and manage run output directory")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("reports/raw"),
        help="Base directory where run_* artifacts are stored.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id suffix. If omitted, current timestamp is used.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="python scripts/final.py",
        help="Pipeline command to execute.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only create/print run directory and command without executing.",
    )
    return parser.parse_args()


def make_run_dir(output_root: Path, run_id: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, args.run_id)

    command = shlex.split(args.command)
    env = os.environ.copy()
    env["RUN_OUTPUT_DIR"] = str(run_dir)

    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] command: {' '.join(command)}")
    print("[INFO] env RUN_OUTPUT_DIR exported for downstream scripts.")

    if args.dry_run:
        print("[INFO] dry-run mode enabled. Command execution skipped.")
        return

    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print(f"[INFO] Completed successfully. Artifacts should be available under: {run_dir}")


if __name__ == "__main__":
    main()
