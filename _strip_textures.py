#!/usr/bin/env python3
"""Helper: for each asset under datasets/eval_assets_e2e/, strip textures from
<uid>.glb and write _input_mesh.glb next to it. Skips assets that already have
an _input_mesh.glb. Reports failures, never aborts the batch."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.baselines._common import strip_textures_glb

SRC_DIR = REPO_ROOT / "datasets" / "eval_assets_e2e"


def main() -> int:
    n_done = n_skip = n_fail = 0
    for asset_dir in sorted(SRC_DIR.iterdir()):
        if not asset_dir.is_dir():
            continue
        uid = asset_dir.name
        src = asset_dir / f"{uid}.glb"
        dst = asset_dir / "_input_mesh.glb"
        if not src.exists():
            print(f"  [skip] no {src.name}: {asset_dir}")
            n_skip += 1
            continue
        if dst.exists():
            print(f"  [skip] already exists: {dst}")
            n_skip += 1
            continue
        try:
            strip_textures_glb(src, dst)
            sz = dst.stat().st_size / 1024
            print(f"  OK {uid}  ({sz:.1f} KB)")
            n_done += 1
        except Exception as e:
            print(f"  FAIL {uid}: {e}")
            n_fail += 1
    print(f"\n[done] {n_done} stripped, {n_skip} skipped, {n_fail} failed")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
