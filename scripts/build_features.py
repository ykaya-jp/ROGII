"""Build full train + test feature parquet for exp004 (Approach A)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src layout works when invoked as `python scripts/build_features.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rogii.beam import warmup as beam_warmup  # noqa: E402
from rogii.features_d1 import build_cluster_map  # noqa: E402
from rogii.train_v3 import (  # noqa: E402
    CLUSTER_PARQUET,
    _resolve_paths,
    build_features,
)


def main() -> int:
    beam_warmup()
    cluster_map = build_cluster_map(CLUSTER_PARQUET)
    train_dir, test_dir, _ = _resolve_paths()

    out_dir = Path("outputs/exp004")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_train = out_dir / "features.parquet"
    cache_test = out_dir / "features_test.parquet"

    t0 = time.perf_counter()
    train_df, test_df = build_features(
        train_dir,
        test_dir,
        cluster_map,
        n_jobs=4,
        quiet=False,
        n_wells_limit=None,
        use_self_exclusion=True,
    )
    train_df.to_parquet(cache_train)
    test_df.to_parquet(cache_test)
    print(
        f"[features] saved {cache_train} ({len(train_df)} rows) "
        f"and {cache_test} ({len(test_df)} rows) "
        f"in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
