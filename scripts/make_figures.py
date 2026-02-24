#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


FILE_CANDIDATES = {
    "compare_csv": [
        "tft_vs_lgbm_compare.csv",
        "compare_tft_lgbm.csv",
        "auc_compare.csv",
    ],
    "summary_json": [
        "tft_vs_lgbm_summary.json",
        "compare_summary.json",
        "summary.json",
    ],
    "lofo_group_csv": [
        "lofo_group_agg.csv",
        "lofo_group.csv",
        "group_lofo.csv",
    ],
    "lofo_feature_csv": [
        "lofo_feature_agg.csv",
        "lofo_feature.csv",
        "feature_lofo.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README figures from run_final artifacts")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing run artifacts. If omitted, latest reports/raw/run_* is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to write generated PNG files.",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N features for fig06")
    return parser.parse_args()


def pick_latest_run_dir(raw_dir: Path) -> Path:
    run_dirs = [p for p in raw_dir.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under: {raw_dir}")
    latest = max(run_dirs, key=lambda p: p.name)
    print(f"[INFO] Auto-selected latest run directory: {latest}")
    return latest


def find_file(input_dir: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p
    return None


def load_csv_with_required_columns(path: Path, required_any: list[list[str]], label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {label}: {path.name}")
    print(f"[INFO] Available columns ({label}): {list(df.columns)}")

    missing_groups = [group for group in required_any if not any(col in df.columns for col in group)]
    if missing_groups:
        missing_text = " OR ".join(["{" + ", ".join(g) + "}" for g in missing_groups])
        raise ValueError(
            f"Missing required columns for {label}: {missing_text}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def load_summary_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded summary JSON: {path.name}")
    print(f"[INFO] Top-level keys (summary_json): {list(data.keys())}")
    return data


def pick_col(df: pd.DataFrame, exact_candidates: list[str], contains_all: list[str] | None = None) -> str:
    for c in exact_candidates:
        if c in df.columns:
            return c
    if contains_all:
        for c in df.columns:
            lowered = c.lower()
            if all(token in lowered for token in contains_all):
                return c
    raise ValueError(f"Could not detect column from candidates={exact_candidates}, contains_all={contains_all}")


def plot_fig03_fold_auc(compare_df: pd.DataFrame, output_path: Path) -> None:
    fold_col = pick_col(compare_df, ["fold", "fold_id", "cv_fold"], contains_all=["fold"])
    lgbm_auc_col = pick_col(compare_df, ["lgbm_auc", "auc_lgbm"], contains_all=["lgbm", "auc"])
    tft_auc_col = pick_col(compare_df, ["tft_auc", "auc_tft"], contains_all=["tft", "auc"])

    d = compare_df[[fold_col, lgbm_auc_col, tft_auc_col]].copy().sort_values(fold_col)

    plt.figure(figsize=(10, 5.5))
    plt.plot(d[fold_col], d[lgbm_auc_col], marker="o", label="LightGBM")
    plt.plot(d[fold_col], d[tft_auc_col], marker="o", label="TFT")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.title("Fold-by-fold AUC: LightGBM vs TFT")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig04_auc_dist(compare_df: pd.DataFrame, output_path: Path) -> None:
    lgbm_auc_col = pick_col(compare_df, ["lgbm_auc", "auc_lgbm"], contains_all=["lgbm", "auc"])
    tft_auc_col = pick_col(compare_df, ["tft_auc", "auc_tft"], contains_all=["tft", "auc"])

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot([compare_df[lgbm_auc_col].dropna(), compare_df[tft_auc_col].dropna()], labels=["LightGBM", "TFT"])
    plt.ylabel("AUC")
    plt.title("AUC Distribution by Model")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lofo_bar(df: pd.DataFrame, output_path: Path, title: str, top_n: int | None = None) -> None:
    name_col = pick_col(df, ["group", "group_name", "feature", "feature_name"], None)
    delta_col = pick_col(df, ["delta_auc", "auc_delta", "diff_auc"], contains_all=["auc", "delta"])

    d = df[[name_col, delta_col]].dropna().copy().sort_values(delta_col, ascending=False)
    if top_n is not None:
        d = d.head(top_n)

    fig_h = max(5.5, 0.35 * len(d))
    plt.figure(figsize=(10, fig_h))
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in d[delta_col]]
    plt.barh(d[name_col].astype(str), d[delta_col], color=colors)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("ΔAUC")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig07_winfold_contrib(compare_df: pd.DataFrame | None, output_path: Path) -> None:
    if compare_df is None:
        _save_todo_figure(output_path, "TODO: compare CSV not found, cannot compute win-fold contribution.")
        return

    try:
        lgbm_auc_col = pick_col(compare_df, ["lgbm_auc", "auc_lgbm"], contains_all=["lgbm", "auc"])
        tft_auc_col = pick_col(compare_df, ["tft_auc", "auc_tft"], contains_all=["tft", "auc"])
    except ValueError as e:
        _save_todo_figure(output_path, f"TODO: {e}")
        return

    d = compare_df[[lgbm_auc_col, tft_auc_col]].dropna().copy()
    if d.empty:
        _save_todo_figure(output_path, "TODO: No non-null fold AUC pairs found.")
        return

    d["delta"] = d[tft_auc_col] - d[lgbm_auc_col]
    win = d[d["delta"] > 0]["delta"]
    non_win = d[d["delta"] <= 0]["delta"]

    if win.empty or non_win.empty:
        _save_todo_figure(output_path, "TODO: Need both win and non-win folds to compare contribution.")
        return

    plt.figure(figsize=(7.5, 5.5))
    values = [win.mean(), non_win.mean()]
    plt.bar(["Win folds", "Non-win folds"], values, color=["#2ca02c", "#d62728"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Mean (TFT AUC - LGBM AUC)")
    plt.title("Win-fold vs Non-win-fold Contribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_todo_figure(output_path: Path, message: str) -> None:
    print(f"[WARN] {message}")
    plt.figure(figsize=(9, 3.5))
    plt.text(0.02, 0.5, message, fontsize=11, va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if input_dir is None:
        input_dir = pick_latest_run_dir(Path("reports/raw"))
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_path = find_file(input_dir, FILE_CANDIDATES["compare_csv"])
    summary_path = find_file(input_dir, FILE_CANDIDATES["summary_json"])
    lofo_group_path = find_file(input_dir, FILE_CANDIDATES["lofo_group_csv"])
    lofo_feature_path = find_file(input_dir, FILE_CANDIDATES["lofo_feature_csv"])

    print("[INFO] File detection results:")
    print(f"  compare_csv: {compare_path}")
    print(f"  summary_json: {summary_path}")
    print(f"  lofo_group_csv: {lofo_group_path}")
    print(f"  lofo_feature_csv: {lofo_feature_path}")

    compare_df = None
    if compare_path:
        compare_df = load_csv_with_required_columns(
            compare_path,
            required_any=[["fold", "fold_id", "cv_fold"], ["lgbm_auc", "auc_lgbm"], ["tft_auc", "auc_tft"]],
            label="compare_csv",
        )
    if summary_path:
        _ = load_summary_json(summary_path)
    if lofo_group_path:
        lofo_group_df = load_csv_with_required_columns(
            lofo_group_path,
            required_any=[["group", "group_name"], ["delta_auc", "auc_delta", "diff_auc"]],
            label="lofo_group_csv",
        )
    else:
        lofo_group_df = None

    if lofo_feature_path:
        lofo_feature_df = load_csv_with_required_columns(
            lofo_feature_path,
            required_any=[["feature", "feature_name"], ["delta_auc", "auc_delta", "diff_auc"]],
            label="lofo_feature_csv",
        )
    else:
        lofo_feature_df = None

    if compare_df is None:
        raise FileNotFoundError("compare CSV is required to generate fig03 and fig04 but was not found.")

    plot_fig03_fold_auc(compare_df, output_dir / "fig03_fold_auc.png")
    plot_fig04_auc_dist(compare_df, output_dir / "fig04_auc_dist.png")

    if lofo_group_df is not None:
        plot_lofo_bar(lofo_group_df, output_dir / "fig05_lofo_group_delta_auc.png", "Group LOFO: ΔAUC by Group")
    else:
        _save_todo_figure(output_dir / "fig05_lofo_group_delta_auc.png", "TODO: lofo_group CSV not found.")

    if lofo_feature_df is not None:
        plot_lofo_bar(
            lofo_feature_df,
            output_dir / "fig06_lofo_feature_topN_delta_auc.png",
            f"Feature LOFO Top {args.top_n}: ΔAUC Ranking",
            top_n=args.top_n,
        )
    else:
        _save_todo_figure(output_dir / "fig06_lofo_feature_topN_delta_auc.png", "TODO: lofo_feature CSV not found.")

    plot_fig07_winfold_contrib(compare_df, output_dir / "fig07_winfold_contrib.png")

    print(f"[INFO] Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
