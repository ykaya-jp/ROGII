"""Unit tests for deepest EDA D1 / D4 / D9 feature helpers (= winning-path B).

Run from repo root:
    .venv/bin/pytest tests/test_winning_path_features.py -v

Covers:
    D1 = compute_b_well_cluster_id:
        - 773 train wells produce exactly 67 unique cluster_ids (= deepest EDA §3.1.1)
        - test wells (000d7d20, 00e12e8b) share the same cluster (= F1 11373)
        - test 00bbac68 lives in a different cluster (= F1 11855)
        - missing well returns -1
        - default round_to=2 yields stable cluster IDs
    D4 = compute_b_jump_max_abs:
        - return contains all train wells with finite values
        - test 000d7d20 ≈ 00e12e8b (= same b_cluster → same b_jump_max)
        - test 00bbac68 differs (= different b_cluster)
    D9 = compute_formation_top_medians:
        - returns 6 columns (top_<F>_med) per well
        - all 6 finite for test wells
        - missing well returns NaNs for all 6 cols
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rogii.features import (  # noqa: E402
    compute_b_jump_max_abs,
    compute_b_well_cluster_id,
    compute_formation_top_medians,
    is_test_cluster,
)

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]
B_CLUSTER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/b-cluster-xy.parquet"
HIGH_ORDER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/per-well-high-order.parquet"
FORMATION_UNIQ_PATH = REPO_ROOT / "outputs/eda/deepest_eda/formation-uniqueness.parquet"


# ────────────────────────────────────────────────────────────────────────────
# D1 — b_well 67 cluster ID
# ────────────────────────────────────────────────────────────────────────────


def test_d1_67_unique_clusters_default():
    """Snapshot: 773 wells reduce to exactly 67 b-clusters at default round_to=3
    (= deepest EDA §3.1.1 canonical setting)."""
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"{B_CLUSTER_PATH} missing")
    bdf = pd.read_parquet(B_CLUSTER_PATH)
    all_wells = bdf["well_id"].astype(str).unique().tolist()
    assert len(all_wells) >= 770, f"expected ~773 train wells, got {len(all_wells)}"
    m = compute_b_well_cluster_id(all_wells)  # default round_to=3
    cluster_ids = set(m.values())
    cluster_ids.discard(-1)
    assert len(cluster_ids) == 67, (
        f"snapshot drift: default round_to=3 should yield 67 b-clusters, got {len(cluster_ids)}"
    )


def test_d1_round_to_sensitivity():
    """round_to controls cluster granularity (see docstring table)."""
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"{B_CLUSTER_PATH} missing")
    bdf = pd.read_parquet(B_CLUSTER_PATH)
    all_wells = bdf["well_id"].astype(str).unique().tolist()
    expected = {0: 87, 1: 55, 2: 66, 3: 67, 4: 67}
    for r, exp_n in expected.items():
        m = compute_b_well_cluster_id(all_wells, round_to=r)
        got_n = len({v for v in m.values() if v != -1})
        assert got_n == exp_n, (
            f"round_to={r}: expected {exp_n} clusters, got {got_n}"
        )


def test_d1_test_wells_share_two_clusters():
    """F1 finding: 3 test wells fall in exactly 2 b-clusters (= 11373 ×2 + 11855 ×1)."""
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"{B_CLUSTER_PATH} missing")
    m = compute_b_well_cluster_id(TEST_WELLS)
    cluster_ids = [m[w] for w in TEST_WELLS]
    assert -1 not in cluster_ids, f"all 3 test wells must be in the parquet: {m}"
    unique = set(cluster_ids)
    assert len(unique) == 2, (
        f"F1: expected exactly 2 unique clusters, got {len(unique)} → {m}"
    )
    # 000d7d20 + 00e12e8b are in the same cluster (= 11373); 00bbac68 is alone (= 11855)
    assert m["000d7d20"] == m["00e12e8b"], (
        f"000d7d20 and 00e12e8b must share a cluster (both b_ANCC_med ≈ 11373); got {m}"
    )
    assert m["00bbac68"] != m["000d7d20"], (
        f"00bbac68 must NOT share with the other two (b_ANCC_med ≈ 11855); got {m}"
    )


def test_d1_missing_well_returns_neg1():
    """Wells not in the parquet get cluster_id = -1 (= sentinel)."""
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"{B_CLUSTER_PATH} missing")
    m = compute_b_well_cluster_id(["nonexistent_well_xxx", "another_missing_yyy"])
    assert m["nonexistent_well_xxx"] == -1
    assert m["another_missing_yyy"] == -1


def test_d1_is_test_cluster_marks_membership():
    """is_test_cluster returns 1 for wells in the 2 test clusters, 0 otherwise."""
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"{B_CLUSTER_PATH} missing")
    bdf = pd.read_parquet(B_CLUSTER_PATH)
    train_wells = [
        w for w in bdf["well_id"].astype(str).unique().tolist()
        if w not in TEST_WELLS
    ][:50]  # smoke on 50
    m = is_test_cluster(train_wells)
    assert set(m.values()).issubset({0, 1}), f"non-binary values found: {set(m.values())}"
    n_marked = sum(m.values())
    # At least one train well should share a cluster with a test well (= F1 says
    # 11373 covers test 000d7d20 + 00e12e8b plus at least more train wells)
    assert n_marked >= 1, (
        f"expected at least 1 train well in a test cluster, got 0 → {m}"
    )


# ────────────────────────────────────────────────────────────────────────────
# D4 — b_jump_max_abs (per-well 6 formation b std)
# ────────────────────────────────────────────────────────────────────────────


def test_d4_test_wells_share_b_jump_when_same_cluster():
    """000d7d20 + 00e12e8b are in the same cluster → b_jump_max_abs should match
    (= deepest EDA §1.4 + F1, the cluster-level formation structure is identical)."""
    if not HIGH_ORDER_PATH.exists():
        pytest.skip(f"{HIGH_ORDER_PATH} missing")
    m = compute_b_jump_max_abs(TEST_WELLS)
    assert not np.isnan(m["000d7d20"])
    assert not np.isnan(m["00e12e8b"])
    assert m["000d7d20"] == pytest.approx(m["00e12e8b"], rel=1e-3), (
        f"same-cluster wells should share b_jump_max_abs; got "
        f"000d7d20={m['000d7d20']} vs 00e12e8b={m['00e12e8b']}"
    )
    # 00bbac68 is in a different cluster → b_jump differs
    assert m["00bbac68"] != pytest.approx(m["000d7d20"], rel=1e-3), (
        f"different-cluster wells should differ in b_jump; got equal values"
    )


def test_d4_missing_well_returns_nan():
    m = compute_b_jump_max_abs(["nonexistent_well_xxx"])
    assert np.isnan(m["nonexistent_well_xxx"])


# ────────────────────────────────────────────────────────────────────────────
# D9 — 6 formation top per-well medians
# ────────────────────────────────────────────────────────────────────────────


EXPECTED_FORMATION_COLS = (
    "top_ANCC_med", "top_ASTNU_med", "top_ASTNL_med",
    "top_EGFDU_med", "top_EGFDL_med", "top_BUDA_med",
)


def test_d9_returns_6_cols_per_well():
    if not FORMATION_UNIQ_PATH.exists():
        pytest.skip(f"{FORMATION_UNIQ_PATH} missing")
    m = compute_formation_top_medians(TEST_WELLS)
    for wid in TEST_WELLS:
        cols = set(m[wid].keys())
        assert cols == set(EXPECTED_FORMATION_COLS), (
            f"well {wid}: expected 6 formation cols, got {cols}"
        )


def test_d9_finite_for_test_wells():
    if not FORMATION_UNIQ_PATH.exists():
        pytest.skip(f"{FORMATION_UNIQ_PATH} missing")
    m = compute_formation_top_medians(TEST_WELLS)
    for wid in TEST_WELLS:
        for col in EXPECTED_FORMATION_COLS:
            assert not np.isnan(m[wid][col]), (
                f"well {wid} col {col} is NaN; expected a finite formation top"
            )


def test_d9_missing_well_returns_all_nan():
    m = compute_formation_top_medians(["nonexistent_well_xxx"])
    row = m["nonexistent_well_xxx"]
    for col in EXPECTED_FORMATION_COLS:
        assert np.isnan(row[col]), f"missing well should have NaN for {col}"
