"""Build features for exp003: formation imputation + typewell signals.

Adds Beam/Self-NCC/tw_diff/xcorr/affine_cal/gr_detrend on top of exp002.

Run:
    PYTHONUNBUFFERED=1 uv run python -m rogii.build_features_v3

Outputs:
    data/processed/train_features_v3.parquet
    data/processed/test_features_v3.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .data_io import (
    TEST_DIR,
    TRAIN_DIR,
    list_wells,
    load_split_horizontals,
)
from .features import FEATURE_COLS_V3, add_features
from .imputers import FormationPlaneKNN

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_typewells_lookup(split_dir: Path) -> dict[str, tuple]:
    lookup: dict[str, tuple] = {}
    for wid in list_wells(split_dir):
        try:
            tw = pd.read_csv(split_dir / f"{wid}__typewell.csv").sort_values("TVT")
            lookup[wid] = (tw["TVT"].to_numpy(), tw["GR"].to_numpy())
        except Exception:
            continue
    return lookup


def main() -> None:
    print("==> building FormationPlaneKNN imputer (centroid K=10, 6 formations)", flush=True)
    t0 = time.perf_counter()
    imputer = FormationPlaneKNN(TRAIN_DIR, k=10)
    print(f"   {len(imputer.df)} centroids in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n==> [train] loading typewells", flush=True)
    t0 = time.perf_counter()
    train_tw = _build_typewells_lookup(TRAIN_DIR)
    print(f"   {len(train_tw)} typewells in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n==> [train] loading horizontals", flush=True)
    t0 = time.perf_counter()
    h_train = load_split_horizontals(TRAIN_DIR)
    print(
        f"   {len(h_train):,} rows / {h_train['well'].nunique()} wells in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print("==> [train] feature engineering (formation + tysig, leave-one-out impute)", flush=True)
    t0 = time.perf_counter()
    train = add_features(h_train, imputer=imputer, typewells=train_tw, exclude_self=True)
    print(f"   features built in {time.perf_counter() - t0:.1f}s", flush=True)

    keep_train = ["well", "row_idx", "TVT_input", *FEATURE_COLS_V3]
    if "TVT" in train.columns:
        keep_train.append("TVT")
    keep_train = [c for c in keep_train if c in train.columns]
    train_path = OUT_DIR / "train_features_v3.parquet"
    train[keep_train].to_parquet(train_path, index=False, compression="zstd")
    print(f"==> [train] wrote {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    print("\n==> [test] loading typewells", flush=True)
    test_tw = _build_typewells_lookup(TEST_DIR)
    print(f"   {len(test_tw)} typewells", flush=True)

    print("==> [test] loading horizontals", flush=True)
    h_test = load_split_horizontals(TEST_DIR)
    print(f"   {len(h_test):,} rows / {h_test['well'].nunique()} wells", flush=True)
    print("==> [test] feature engineering", flush=True)
    t0 = time.perf_counter()
    test = add_features(h_test, imputer=imputer, typewells=test_tw, exclude_self=False)
    print(f"   features built in {time.perf_counter() - t0:.1f}s", flush=True)

    keep_test = ["well", "row_idx", "TVT_input", *FEATURE_COLS_V3]
    keep_test = [c for c in keep_test if c in test.columns]
    test_path = OUT_DIR / "test_features_v3.parquet"
    test[keep_test].to_parquet(test_path, index=False, compression="zstd")
    print(f"==> [test] wrote {test_path} ({test_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    print(f"\n✓ build_features_v3 done. feature count: {len(FEATURE_COLS_V3)}", flush=True)


if __name__ == "__main__":
    main()
