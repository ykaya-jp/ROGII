"""Pre-compute features once with FormationPlaneKNN imputation.

Run:
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
from .imputers import FormationPlaneKNN

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("==> building FormationPlaneKNN imputer (centroid K=10, 6 formations)", flush=True)
    t0 = time.perf_counter()
    imputer = FormationPlaneKNN(TRAIN_DIR, k=10)
    print(
        f"   {len(imputer.df)} centroids in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print("\n==> [train] loading horizontals", flush=True)
    t0 = time.perf_counter()
    h_train = load_split_horizontals(TRAIN_DIR)
    print(
        f"   {len(h_train):,} rows / {h_train['well'].nunique()} wells in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print("==> [train] feature engineering (leave-one-out imputation)", flush=True)
    t0 = time.perf_counter()
    train = add_features(h_train, imputer=imputer, exclude_self=True)
    print(f"   features built in {time.perf_counter() - t0:.1f}s", flush=True)

    keep_train = ["well", "row_idx", "TVT_input", *FEATURE_COLS]
    if "TVT" in train.columns:
        keep_train.append("TVT")
    train_path = OUT_DIR / "train_features.parquet"
    train[keep_train].to_parquet(train_path, index=False, compression="zstd")
    print(f"==> [train] wrote {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    print("\n==> [test] loading horizontals", flush=True)
    t0 = time.perf_counter()
    h_test = load_split_horizontals(TEST_DIR)
    print(
        f"   {len(h_test):,} rows / {h_test['well'].nunique()} wells in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print("==> [test] feature engineering", flush=True)
    t0 = time.perf_counter()
    test = add_features(h_test, imputer=imputer, exclude_self=False)
    print(f"   features built in {time.perf_counter() - t0:.1f}s", flush=True)

    test_path = OUT_DIR / "test_features.parquet"
    test[["well", "row_idx", "TVT_input", *FEATURE_COLS]].to_parquet(
        test_path, index=False, compression="zstd"
    )
    print(f"==> [test] wrote {test_path} ({test_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    print("\n✓ build_features done", flush=True)


if __name__ == "__main__":
    main()
