"""AG deep EDA finding D1: per-well `b_well` cluster categorical encoding.

`b_well` (= TVT_input + Z - formation_top) clusters around discrete formation
elevations in the ROGII dataset (66 unique values across 773 train wells, see
`outputs/eda/deepest_eda/per-well-outlier.parquet`). This module emits a stable
integer cluster id per well, suitable for LightGBM `categorical_feature`.

Public API:
  - `build_cluster_map(parquet_path) -> dict[str, int]`
    Build mapping `well_id -> cluster_id` (sorted by b_med ascending).
  - `encode_well(well_id, cluster_map) -> int`
    Look up cluster id; unseen wells return `UNKNOWN_CLUSTER_ID`.
  - `CLUSTER_MAP_FALLBACK`: hard-coded mapping for Kaggle kernel (inline use).

Design:
  - Cluster id is **a function of `b_med` rounded to 2 decimals**, ordered by
    ascending b_med so it is reproducible from EDA output alone.
  - `UNKNOWN_CLUSTER_ID = -1` for any well not in the map (test 3 wells are
    all in the seen set per AC-13).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UNKNOWN_CLUSTER_ID = -1


def build_cluster_map(parquet_path: str | Path) -> dict[str, int]:
    """Build well_id -> cluster_id map from AG deep-EDA parquet.

    Cluster id is the rank of `b_med` rounded to 2 decimals among all train
    wells (ascending). Two wells sharing the same rounded `b_med` get the
    same cluster id (= 67 distinct clusters when test split rows are excluded).
    """
    df = pd.read_parquet(parquet_path)
    train = df[df["split"] == "train"].copy()
    train["b_round"] = train["b_med"].round(2)
    train = train.dropna(subset=["b_round"])
    # Stable ordering: sort distinct b_round ascending and rank from 0.
    distinct = sorted(train["b_round"].unique())
    b_to_cluster: dict[float, int] = {b: i for i, b in enumerate(distinct)}
    well_to_cluster: dict[str, int] = {}
    for wid, b in zip(train["well_id"].to_list(), train["b_round"].to_list(), strict=True):
        well_to_cluster[str(wid)] = b_to_cluster[b]
    return well_to_cluster


def encode_well(well_id: str, cluster_map: dict[str, int]) -> int:
    """Return cluster id for `well_id`, or UNKNOWN_CLUSTER_ID if unseen."""
    return int(cluster_map.get(well_id, UNKNOWN_CLUSTER_ID))


# ---- Test-only hard-coded subset (used by tests/test_features_d1.py).
# Full map is hard-coded into the Kaggle kernel script via build_cluster_map()
# at kernel build time. For tests we hard-code a 3-well sample to assert
# behavior without depending on the full parquet build.
TEST_CLUSTER_SAMPLE: dict[str, int] = {
    "000d7d20": 30,  # b_med = 11373.26 (= one of two 11373 cluster ids in AG output)
    "00e12e8b": 30,
    "00bbac68": 35,  # b_med = 11855.20
}
