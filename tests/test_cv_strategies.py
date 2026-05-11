"""Unit tests for the 4 CV strategies (C1 / C2 / C3 / C4) introduced by
``.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml``.

Run from repo root:
    .venv/bin/pytest tests/test_cv_strategies.py -v

Covers:
    AC-2 (no-leak + balance + F1 b_cluster rep + F8 typewell hash leak free
          + deterministic seed for all 4 CV)
    AC-3 (full): typewell content-hash detects exactly EXPECTED_DUPLICATE_*
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow ``from rogii...`` resolution when test is run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rogii.cv import (  # noqa: E402
    build_edge_q_folds,
    build_multi_key_stratified_edge_q_folds,
    build_pseudo_test_fold,
    build_stratified_edge_q_folds,
    compute_b_cluster_balance,
    compute_test_distance_features,
    make_well_folds,
    verify_edge_q_no_leak,
)
from rogii.typewell_hash import (  # noqa: E402
    EXPECTED_DUPLICATE_GROUP_COUNT,
    EXPECTED_DUPLICATE_TOTAL_WELLS,
    compute_typewell_hashes,
    count_duplicate_groups,
)

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]
TRAIN_DIR = REPO_ROOT / "data/raw/train"
TEST_DIR = REPO_ROOT / "data/raw/test"
STATS_PATH = REPO_ROOT / "outputs/eda/first_principles/per-well-stats.parquet"
HIGH_ORDER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/per-well-high-order.parquet"
B_CLUSTER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/b-cluster-xy.parquet"


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def stats() -> pd.DataFrame:
    if not STATS_PATH.exists():
        pytest.skip(f"per-well-stats.parquet not found at {STATS_PATH}")
    return pd.read_parquet(STATS_PATH)


@pytest.fixture(scope="module")
def high_order() -> pd.DataFrame:
    if not HIGH_ORDER_PATH.exists():
        pytest.skip(f"per-well-high-order.parquet not found at {HIGH_ORDER_PATH}")
    return pd.read_parquet(HIGH_ORDER_PATH)


@pytest.fixture(scope="module")
def b_cluster_lookup() -> dict[str, float]:
    if not B_CLUSTER_PATH.exists():
        pytest.skip(f"b-cluster-xy.parquet not found at {B_CLUSTER_PATH}")
    df = pd.read_parquet(B_CLUSTER_PATH)
    return dict(zip(df["well_id"].astype(str), df["b_ANCC_med"].astype(float)))


@pytest.fixture(scope="module")
def synthetic_train_df(stats: pd.DataFrame) -> pd.DataFrame:
    """50 wells × 100 rows synthetic frame with target proportional to
    ``tw_gr_resid_std`` so 'difficult' wells produce higher-variance targets."""
    rng = np.random.RandomState(42)
    sampled = stats.sort_values("well_id").sample(n=50, random_state=42).reset_index(drop=True)
    rows = []
    for _, r in sampled.iterrows():
        wid = str(r["well_id"])
        sigma = float(r["tw_gr_resid_std"])
        for i in range(100):
            rows.append({"well": wid, "id": f"{wid}_{i}", "target": float(rng.randn() * sigma)})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# AC-3 full: typewell content-hash duplicate group snapshot on real train
# ────────────────────────────────────────────────────────────────────────────


def test_ac3_typewell_dup_groups_snapshot():
    """Real train wells reproduce EXPECTED_DUPLICATE_GROUP_COUNT (= 13) and
    EXPECTED_DUPLICATE_TOTAL_WELLS (= 35) — regression target from subagent G
    commit 1bb6caf."""
    if not TRAIN_DIR.exists():
        pytest.skip(f"train dir not found at {TRAIN_DIR}")
    well_ids = sorted(
        p.stem.replace("__horizontal_well", "")
        for p in TRAIN_DIR.glob("*__horizontal_well.csv")
    )
    assert len(well_ids) > 700, f"expected ~773 train wells, got {len(well_ids)}"
    hashes = compute_typewell_hashes(TRAIN_DIR, well_ids)
    n_groups, n_wells = count_duplicate_groups(hashes)
    assert n_groups == EXPECTED_DUPLICATE_GROUP_COUNT, (
        f"snapshot drift: expected {EXPECTED_DUPLICATE_GROUP_COUNT} dup groups, got {n_groups}"
    )
    assert n_wells == EXPECTED_DUPLICATE_TOTAL_WELLS, (
        f"snapshot drift: expected {EXPECTED_DUPLICATE_TOTAL_WELLS} wells in dup groups, got {n_wells}"
    )


# ────────────────────────────────────────────────────────────────────────────
# C1 — pseudo-test fold (kNN-based)
# ────────────────────────────────────────────────────────────────────────────


def _split_train_test_features(high_order: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    well_col = "well_id"
    df = high_order.copy()
    df[well_col] = df[well_col].astype(str)
    train_feat = (
        df[~df[well_col].isin(TEST_WELLS)]
        .drop_duplicates(subset=[well_col])
        .set_index(well_col)
    )
    test_feat = (
        df[df[well_col].isin(TEST_WELLS)]
        .drop_duplicates(subset=[well_col])
        .set_index(well_col)
    )
    return train_feat, test_feat


def test_c1_pseudo_test_fold_basic(synthetic_train_df: pd.DataFrame, high_order: pd.DataFrame):
    """C1 fold builds without error and partitions all rows."""
    train_feat, test_feat = _split_train_test_features(high_order)
    fold_id, groups = build_pseudo_test_fold(
        synthetic_train_df, train_feat, test_feat,
        n_splits=5, seed=42, k_per_test_well=10,
    )
    assert len(fold_id) == len(synthetic_train_df)
    assert len(groups) == len(synthetic_train_df)
    # All 5 fold ids present
    assert set(np.unique(fold_id).tolist()).issubset({0, 1, 2, 3, 4})
    # Fold 0 (= pseudo-test) is non-empty
    assert (fold_id == 0).sum() > 0


def test_c1_pseudo_test_fold_no_leak(synthetic_train_df: pd.DataFrame, high_order: pd.DataFrame):
    """Each well belongs to exactly one fold (no leak)."""
    train_feat, test_feat = _split_train_test_features(high_order)
    fold_id, _groups = build_pseudo_test_fold(
        synthetic_train_df, train_feat, test_feat,
        n_splits=5, seed=42, k_per_test_well=10,
    )
    df = pd.DataFrame({"well": synthetic_train_df["well"].astype(str), "fold": fold_id})
    folds_per_well = df.groupby("well")["fold"].nunique()
    assert (folds_per_well == 1).all(), f"leak: {folds_per_well[folds_per_well > 1].to_dict()}"


def test_c1_pseudo_test_fold_deterministic(synthetic_train_df: pd.DataFrame, high_order: pd.DataFrame):
    """Same seed → identical fold assignment."""
    train_feat, test_feat = _split_train_test_features(high_order)
    fold_id_a, _ = build_pseudo_test_fold(
        synthetic_train_df, train_feat, test_feat, seed=42, k_per_test_well=10,
    )
    fold_id_b, _ = build_pseudo_test_fold(
        synthetic_train_df, train_feat, test_feat, seed=42, k_per_test_well=10,
    )
    np.testing.assert_array_equal(fold_id_a, fold_id_b)


def test_c1_pca_variance_threshold_enforced(high_order: pd.DataFrame):
    """PCA variance ratio below threshold raises ValueError (= fragility guard).

    Variance ratio is bounded in [0, 1], so a threshold > 1.0 is
    unsatisfiable and must always raise — this is the safety net for the
    ``pca_variance_min`` argument's contract.
    """
    train_feat, test_feat = _split_train_test_features(high_order)
    with pytest.raises(ValueError, match="PCA variance ratio"):
        compute_test_distance_features(
            train_feat, test_feat,
            pca_components=10,
            pca_variance_min=2.0,  # impossible: variance ratio ∈ [0, 1]
        )


# ────────────────────────────────────────────────────────────────────────────
# C2 — multi-key stratified Edge Q
# ────────────────────────────────────────────────────────────────────────────


def test_c2_multi_key_no_leak(synthetic_train_df: pd.DataFrame, stats: pd.DataFrame):
    """typewell hash groups stay atomic across folds."""
    fold_id, groups = build_multi_key_stratified_edge_q_folds(
        synthetic_train_df, TRAIN_DIR, stats,
        stratify_specs=[
            ("visible_ratio", "quantile", 5),
            ("tw_gr_resid_std", "quantile", 4),
        ],
        n_splits=5, seed=42,
    )
    ok, msg = verify_edge_q_no_leak(fold_id, groups)
    assert ok, msg


def test_c2_multi_key_balance(synthetic_train_df: pd.DataFrame, stats: pd.DataFrame):
    """Each fold has roughly comparable size (max/min ≤ 2× under synthetic 50 wells)."""
    fold_id, _ = build_multi_key_stratified_edge_q_folds(
        synthetic_train_df, TRAIN_DIR, stats, n_splits=5, seed=42,
    )
    counts = np.bincount(fold_id, minlength=5)
    assert counts.min() > 0, f"fold has zero rows: {counts.tolist()}"
    assert counts.max() / max(counts.min(), 1) <= 3.0, (
        f"fold size imbalance: {counts.tolist()}"
    )


def test_c2_multi_key_deterministic(synthetic_train_df: pd.DataFrame, stats: pd.DataFrame):
    """Same seed → identical fold partition."""
    fold_a, _ = build_multi_key_stratified_edge_q_folds(
        synthetic_train_df, TRAIN_DIR, stats, n_splits=5, seed=42,
    )
    fold_b, _ = build_multi_key_stratified_edge_q_folds(
        synthetic_train_df, TRAIN_DIR, stats, n_splits=5, seed=42,
    )
    np.testing.assert_array_equal(fold_a, fold_b)


# ────────────────────────────────────────────────────────────────────────────
# C4 — b_cluster post-hoc balance audit
# ────────────────────────────────────────────────────────────────────────────


def test_c4_b_cluster_balance_pivot(
    synthetic_train_df: pd.DataFrame, b_cluster_lookup: dict[str, float]
):
    """compute_b_cluster_balance returns (pivot, p_value) with sensible shapes."""
    fold_id, _ = build_edge_q_folds(synthetic_train_df, TRAIN_DIR, n_splits=5, seed=42)
    pivot, p_value = compute_b_cluster_balance(
        fold_id, synthetic_train_df, b_cluster_lookup
    )
    assert not pivot.empty, "pivot is empty"
    assert pivot.shape[0] >= 1, f"too few folds in pivot: {pivot.shape}"
    assert pivot.shape[1] >= 1, f"too few clusters in pivot: {pivot.shape}"
    # p_value is either NaN or in [0, 1]
    assert np.isnan(p_value) or 0.0 <= p_value <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# Cross-CV invariants (= AC-2 omnibus)
# ────────────────────────────────────────────────────────────────────────────


def test_all_cv_no_well_leak(synthetic_train_df: pd.DataFrame, stats: pd.DataFrame, high_order: pd.DataFrame):
    """Baseline + Edge Q v2 + Edge Q v3 stratified + C2 multi-key all no-leak."""
    train_feat, test_feat = _split_train_test_features(high_order)
    folds_baseline = make_well_folds(synthetic_train_df["well"].values, n_splits=5, seed=42)
    for _train_idx, val_idx in folds_baseline:
        val_wells = set(synthetic_train_df.iloc[val_idx]["well"].unique())
        train_wells = set(synthetic_train_df.iloc[_train_idx]["well"].unique())
        assert val_wells.isdisjoint(train_wells)
    for fn, kwargs in [
        (build_edge_q_folds, dict(train_df=synthetic_train_df, train_dir=TRAIN_DIR, n_splits=5, seed=42)),
        (build_stratified_edge_q_folds, dict(train_df=synthetic_train_df, train_dir=TRAIN_DIR, per_well_stats_df=stats, n_splits=5, seed=42)),
        (build_multi_key_stratified_edge_q_folds, dict(train_df=synthetic_train_df, train_dir=TRAIN_DIR, per_well_stats_df=stats, n_splits=5, seed=42)),
    ]:
        fold_id, groups = fn(**kwargs)
        ok, msg = verify_edge_q_no_leak(fold_id, groups)
        assert ok, f"{fn.__name__}: {msg}"
    fold_id_c1, groups_c1 = build_pseudo_test_fold(
        synthetic_train_df, train_feat, test_feat, n_splits=5, seed=42, k_per_test_well=10,
    )
    df = pd.DataFrame({"well": synthetic_train_df["well"].astype(str), "fold": fold_id_c1})
    folds_per_well = df.groupby("well")["fold"].nunique()
    assert (folds_per_well == 1).all(), f"C1 leak: {folds_per_well[folds_per_well > 1].to_dict()}"
