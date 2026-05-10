"""Pre-compute features once and cache to parquet for re-use across experiments.

Usage:
    PYTHONUNBUFFERED=1 uv run python -m rogii.build_features

Outputs:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

from .data_io import (
    TEST_DIR,
    TRAIN_DIR,
    load_split_horizontals,
)
from .features import FEATURE_COLS, add_features

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build(split: str, src_dir: Path) -> None:
    print(f"\n==> [{split}] loading horizontals", flush=True)
    t0 = time.perf_counter()
    h = load_split_horizontals(src_dir)
    print(
        f"   loaded {len(h):,} rows / {h['well'].nunique()} wells in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    print(f"==> [{split}] feature engineering", flush=True)
    t0 = time.perf_counter()
    h = add_features(h)
    print(f"   features built in {time.perf_counter()-t0:.1f}s", flush=True)

    keep_cols = ["well", "row_idx", "TVT_input", *FEATURE_COLS]
    if "TVT" in h.columns:
        keep_cols.append("TVT")

    out_path = OUT_DIR / f"{split}_features.parquet"
    h[keep_cols].to_parquet(out_path, index=False, compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"==> [{split}] wrote {out_path} ({size_mb:.1f} MB)", flush=True)


def main() -> None:
    _build("train", TRAIN_DIR)
    _build("test", TEST_DIR)
    print("\n✓ build_features done", flush=True)


if __name__ == "__main__":
    main()
