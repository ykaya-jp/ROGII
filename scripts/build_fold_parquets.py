"""Pre-generate fold parquets for the 5 CV strategies (baseline + C1-C4).

Each output parquet has columns ``well`` + ``fold`` and lists every train
well exactly once. Kernels that train ``exp005..exp010`` pick up these
parquets via ``FOLD_OVERRIDE_PARQUET`` env var (see kernel patches added
in the same task) so the same kernel binary can be re-run under each CV
strategy, isolating the CV-strategy effect from model / feature changes.

Outputs:
    outputs/folds/baseline.parquet  # make_well_folds (= plain GroupKFold)
    outputs/folds/C1.parquet        # build_pseudo_test_fold (= kNN)
    outputs/folds/C2.parquet        # build_multi_key_stratified_edge_q_folds
    outputs/folds/C3.parquet        # baseline + adversarial drop
    outputs/folds/C4.parquet        # build_edge_q_folds (= typewell hash GKF)

Run:
    .venv/bin/python scripts/build_fold_parquets.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rogii.adversarial import compute_adversarial_classifier_oof, apply_adversarial_drop_fold
from rogii.cv import (
    build_edge_q_folds,
    build_multi_key_stratified_edge_q_folds,
    build_pseudo_test_fold,
    compute_b_cluster_balance,
    make_well_folds,
    verify_edge_q_no_leak,
)


TRAIN_DIR = REPO_ROOT / "data/raw/train"
TEST_DIR = REPO_ROOT / "data/raw/test"
STATS_PATH = REPO_ROOT / "outputs/eda/first_principles/per-well-stats.parquet"
HIGH_ORDER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/per-well-high-order.parquet"
B_CLUSTER_PATH = REPO_ROOT / "outputs/eda/deepest_eda/b-cluster-xy.parquet"


def _save_fold(well_ids: np.ndarray, fold_id: np.ndarray, path: Path, cv_name: str) -> dict:
    """Write the (well, fold) parquet and return a sanity-summary dict."""
    df = pd.DataFrame({"well": np.asarray(well_ids).astype(str), "fold": fold_id.astype(np.int8)})
    # Drop any wells assigned fold = -1 (= adversarial drop sentinel) from the
    # summary numbers, but keep them in the parquet so the kernel can decide
    # whether to honor the drop.
    counts = (
        df[df["fold"] >= 0]
        .groupby("fold")
        .size()
        .sort_index()
        .to_dict()
    )
    dropped = int((df["fold"] == -1).sum())
    df.to_parquet(path)
    return {
        "cv": cv_name,
        "path": str(path.relative_to(REPO_ROOT)),
        "n_wells": int(len(df)),
        "n_dropped": dropped,
        "fold_sizes": counts,
        "fold_size_min": int(min(counts.values())) if counts else 0,
        "fold_size_max": int(max(counts.values())) if counts else 0,
        "fold_size_max_over_min": (
            max(counts.values()) / max(min(counts.values()), 1) if counts else None
        ),
    }


def main(out_dir: Path, seed: int, n_splits: int) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== build_fold_parquets ==  seed={seed} n_splits={n_splits} out={out_dir}")

    # Enumerate train + test well_ids
    train_wells = sorted(
        p.stem.replace("__horizontal_well", "")
        for p in TRAIN_DIR.glob("*__horizontal_well.csv")
    )
    test_wells = sorted(
        p.stem.replace("__horizontal_well", "")
        for p in TEST_DIR.glob("*__horizontal_well.csv")
    )
    print(f"  train wells: {len(train_wells)}  test wells: {len(test_wells)}")

    # Synthetic 1-row-per-well frame; folds depend on well_id only, not rows
    train_df = pd.DataFrame({"well": train_wells, "target": 0.0})

    # Load per-well stats (used for C2 stratification + C3 features)
    if not STATS_PATH.exists():
        raise FileNotFoundError(STATS_PATH)
    stats = pd.read_parquet(STATS_PATH)

    summary: list[dict] = []

    # ────────── baseline (plain GroupKFold by well) ──────────
    folds_baseline = make_well_folds(train_df["well"].values, n_splits=n_splits, seed=seed)
    fold_id_baseline = np.full(len(train_df), -1, dtype=np.int8)
    for k, (_tr, va) in enumerate(folds_baseline):
        fold_id_baseline[va] = k
    summary.append(_save_fold(train_df["well"].values, fold_id_baseline, out_dir / "baseline.parquet", "baseline"))

    # ────────── C1 — pseudo-test fold via kNN ──────────
    if HIGH_ORDER_PATH.exists():
        ho = pd.read_parquet(HIGH_ORDER_PATH)
        ho["well_id"] = ho["well_id"].astype(str)
        train_feat = (
            ho[~ho["well_id"].isin(test_wells)]
            .drop_duplicates(subset=["well_id"])
            .set_index("well_id")
        )
        test_feat = (
            ho[ho["well_id"].isin(test_wells)]
            .drop_duplicates(subset=["well_id"])
            .set_index("well_id")
        )
        fold_id_c1, _ = build_pseudo_test_fold(
            train_df, train_feat, test_feat,
            n_splits=n_splits, seed=seed, k_per_test_well=50,
        )
        summary.append(_save_fold(train_df["well"].values, fold_id_c1, out_dir / "C1.parquet", "C1"))
    else:
        print(f"  [C1] high-order parquet not found at {HIGH_ORDER_PATH} — skipping C1")

    # ────────── C2 — multi-key stratified Edge Q ──────────
    fold_id_c2, groups_c2 = build_multi_key_stratified_edge_q_folds(
        train_df, TRAIN_DIR, stats,
        stratify_specs=[
            ("visible_ratio",   "quantile", 5),
            ("tw_gr_resid_std", "quantile", 4),
        ],
        n_splits=n_splits, seed=seed,
    )
    ok, msg = verify_edge_q_no_leak(fold_id_c2, groups_c2)
    print(f"  [C2] {msg}")
    assert ok, f"C2 leak: {msg}"
    summary.append(_save_fold(train_df["well"].values, fold_id_c2, out_dir / "C2.parquet", "C2"))

    # ────────── C3 — adversarial validation drop on top of baseline ──────────
    print(f"  [C3] fitting adversarial classifier...")
    t_av = time.perf_counter()
    try:
        auc, train_likelihood = compute_adversarial_classifier_oof(
            TRAIN_DIR, TEST_DIR, train_wells, test_wells,
            n_estimators=200, early_stopping=20, seed=seed,
        )
        print(f"  [C3] adversarial AUC = {auc:.4f}  ({time.perf_counter() - t_av:.1f}s)")
    except Exception as e:
        print(f"  [C3] adversarial classifier failed: {type(e).__name__}: {e}")
        print(f"  [C3] falling back to baseline fold (= no drop)")
        auc, train_likelihood = float("nan"), {}

    if train_likelihood:
        fold_id_c3 = apply_adversarial_drop_fold(
            train_df, fold_id_baseline, train_likelihood, drop_quantile=0.20,
        )
    else:
        fold_id_c3 = fold_id_baseline.copy()
    summary.append({
        **_save_fold(train_df["well"].values, fold_id_c3, out_dir / "C3.parquet", "C3"),
        "adversarial_auc": auc,
    })

    # ────────── C4 — typewell hash GroupKFold + b_cluster post-hoc audit ──────────
    fold_id_c4, groups_c4 = build_edge_q_folds(
        train_df, TRAIN_DIR, n_splits=n_splits, seed=seed,
    )
    ok, msg = verify_edge_q_no_leak(fold_id_c4, groups_c4)
    print(f"  [C4] {msg}")
    assert ok, f"C4 leak: {msg}"
    s4 = _save_fold(train_df["well"].values, fold_id_c4, out_dir / "C4.parquet", "C4")
    # Audit b_cluster balance
    if B_CLUSTER_PATH.exists():
        bdf = pd.read_parquet(B_CLUSTER_PATH)
        b_lookup = dict(zip(bdf["well_id"].astype(str), bdf["b_ANCC_med"].astype(float)))
        pivot, p_value = compute_b_cluster_balance(fold_id_c4, train_df, b_lookup)
        s4["b_cluster_balance_p_value"] = p_value
        s4["b_cluster_balance_pivot_shape"] = list(pivot.shape)
        print(f"  [C4] b_cluster balance chi-square p_value = {p_value:.4f}  pivot shape={pivot.shape}")
    summary.append(s4)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  summary written to {summary_path}")
    print(f"== done == ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "folds"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    main(Path(args.out_dir), args.seed, args.n_splits)
