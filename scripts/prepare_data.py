#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def read_simple_yaml(path: Path) -> dict:
    """
    Minimal YAML reader for our simple config (key: value, nested one level).
    If you already use PyYAML/Hydra, replace this with yaml.safe_load.
    """
    data = {}
    current_section = None
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
            # booleans
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            # attach
            if current_section:
                data[current_section][k] = v
            else:
                data[k] = v
    return data


@dataclass
class SyncSpec:
    src_glob: str
    dst_dir: Path
    incremental: bool
    mirror: bool


def list_sources(src_glob: str) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(src_glob))]


def same_file(src: Path, dst: Path) -> bool:
    """Heuristic: consider same if size matches and dst mtime >= src mtime."""
    if not dst.exists():
        return False
    try:
        return (src.stat().st_size == dst.stat().st_size) and (dst.stat().st_mtime >= src.stat().st_mtime)
    except FileNotFoundError:
        return False


def copy_files(spec: SyncSpec) -> Tuple[int, int, int]:
    """
    Returns (copied, skipped, missing_src)
    """
    src_files = list_sources(spec.src_glob)
    if not src_files:
        return (0, 0, 1)

    spec.dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src in src_files:
        dst = spec.dst_dir / src.name
        if spec.incremental and same_file(src, dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    if spec.mirror:
        src_names = {p.name for p in src_files}
        for dst in spec.dst_dir.glob("*"):
            if dst.is_file() and dst.name not in src_names:
                dst.unlink()

    return (copied, skipped, 0)


def main():
    parser = argparse.ArgumentParser(description="Sync Drive datasets into local cache (data/raw).")
    parser.add_argument("--config", type=str, default="configs/final.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = read_simple_yaml(cfg_path)

    # Validate essentials
    for section in ("data", "sync"):
        if section not in cfg:
            raise ValueError(f"Missing section '{section}' in {cfg_path}")

    dcfg = cfg["data"]
    scfg = cfg["sync"]

    trades_spec = SyncSpec(
        src_glob=str(dcfg["drive_trades_glob"]),
        dst_dir=Path(str(dcfg["local_trades_dir"])),
        incremental=bool(scfg.get("incremental", True)),
        mirror=bool(scfg.get("mirror", False)),
    )
    onchain_spec = SyncSpec(
        src_glob=str(dcfg["drive_onchain_glob"]),
        dst_dir=Path(str(dcfg["local_onchain_dir"])),
        incremental=bool(scfg.get("incremental", True)),
        mirror=bool(scfg.get("mirror", False)),
    )

    print("=== prepare_data ===")
    print(f"[trades]  src: {trades_spec.src_glob}")
    print(f"[trades]  dst: {trades_spec.dst_dir}")
    c, s, miss = copy_files(trades_spec)
    if miss:
        print("[trades]  ERROR: no source files matched. Is Google Drive mounted?")
        print("          Hint: in Colab run: from google.colab import drive; drive.mount('/content/drive')")
        print("          Also check the path/glob in configs/final.yaml")
        raise SystemExit(1)
    print(f"[trades]  copied={c}, skipped={s}")

    print(f"[onchain] src: {onchain_spec.src_glob}")
    print(f"[onchain] dst: {onchain_spec.dst_dir}")
    c, s, miss = copy_files(onchain_spec)
    if miss:
        print("[onchain] ERROR: no source files matched. Check path/glob and Drive mount.")
        raise SystemExit(1)
    print(f"[onchain] copied={c}, skipped={s}")

    # Quick summary
    print("\n=== summary ===")
    print(f"trades_parquet files: {len(list(trades_spec.dst_dir.glob('*.parquet')))}")
    print(f"onchain_csv files:    {len(list(onchain_spec.dst_dir.glob('*.csv')))}")
    print("Done.")


if __name__ == "__main__":
    main()
