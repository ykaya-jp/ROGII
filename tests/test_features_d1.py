"""Test for src/rogii/features_d1.py — D1 cluster encoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from rogii.features_d1 import (
    UNKNOWN_CLUSTER_ID,
    build_cluster_map,
    encode_well,
)

PARQUET = Path(
    "/home/yusuke_kaya/projects/kaggle/ROGII/outputs/eda/deepest_eda/per-well-outlier.parquet"
)


@pytest.fixture(scope="module")
def cluster_map() -> dict[str, int]:
    if not PARQUET.exists():
        pytest.skip(f"per-well-outlier.parquet missing at {PARQUET}")
    return build_cluster_map(PARQUET)


def test_build_cluster_map_total_clusters(cluster_map: dict[str, int]) -> None:
    """66 distinct b_med (rounded 2 decimals) clusters in train split."""
    n_unique = len(set(cluster_map.values()))
    # 66 = total unique b_med round 2 in train (AG output).
    # We accept [60, 75] as a tolerant band in case parquet schema drifts.
    assert 60 <= n_unique <= 75


def test_test_three_wells_seen(cluster_map: dict[str, int]) -> None:
    """AC-13: test 3 wells must be seen clusters (= deterministic encoding)."""
    for wid in ("000d7d20", "00bbac68", "00e12e8b"):
        cid = encode_well(wid, cluster_map)
        assert cid != UNKNOWN_CLUSTER_ID, f"{wid} unexpectedly unseen"
        assert cid >= 0


def test_two_test_wells_share_cluster(cluster_map: dict[str, int]) -> None:
    """000d7d20 + 00e12e8b both have b_med = 11373.26 -> same cluster id."""
    cid_a = encode_well("000d7d20", cluster_map)
    cid_b = encode_well("00e12e8b", cluster_map)
    assert cid_a == cid_b


def test_third_test_well_distinct_cluster(cluster_map: dict[str, int]) -> None:
    """00bbac68 has b_med = 11855.20 -> different cluster than the 11373 pair."""
    cid_a = encode_well("000d7d20", cluster_map)
    cid_c = encode_well("00bbac68", cluster_map)
    assert cid_a != cid_c


def test_unknown_well_returns_default() -> None:
    """Unseen well -> UNKNOWN_CLUSTER_ID (= -1)."""
    cid = encode_well("ZZZZZZZZ", {})
    assert cid == UNKNOWN_CLUSTER_ID
    assert cid == -1


def test_cluster_ids_dense_zero_indexed(cluster_map: dict[str, int]) -> None:
    """Cluster ids are dense [0, n_unique - 1]."""
    ids = set(cluster_map.values())
    assert min(ids) == 0
    assert max(ids) == len(ids) - 1
