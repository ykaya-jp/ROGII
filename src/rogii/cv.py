"""Cross-validation for ROGII.

Strategy:
- v1 (= exp001-006): plain GroupKFold by well — each well lives in exactly
  one validation fold so target leak across wells is impossible.
- v2 (= exp007 / Edge Q): typewell content-hash GroupKFold — wells sharing
  the same (TVT, GR, Geology) typewell file (= pseudo-typewell group of
  35 wells / 13 groups, host-confirmed) are forced into the same fold.
- v3 (= exp010 / fold reform): **stratified** typewell-hash GroupKFold —
  fold partition additionally balances the per-well difficulty distribution
  (default key = ``tw_gr_resid_std`` quartile from per-well-stats.parquet).
  Goal = drive σ_fold from 1.18 ft (exp007 observed) → 0.5 ft or lower,
  smashing the Jensen lower bound CV ≈ 10.77 ft to ≤ 10.2 ft.

Validation metric is RMSE on hidden rows of held-out wells only, which
mirrors the public leaderboard scoring exactly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_well_folds(
    well_ids: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) splits where each well is in exactly one val fold.

    Wells are shuffled deterministically using `seed` before being assigned to folds.
    """
    rng = np.random.RandomState(seed)
    unique_wells = np.array(sorted(np.unique(well_ids)))
    rng.shuffle(unique_wells)

    # Build a mapping: well → fold index. Spread wells round-robin.
    fold_of = {w: i % n_splits for i, w in enumerate(unique_wells)}
    well_fold = np.array([fold_of[w] for w in well_ids])

    folds = []
    for k in range(n_splits):
        val_idx = np.where(well_fold == k)[0]
        train_idx = np.where(well_fold != k)[0]
        folds.append((train_idx, val_idx))
    return folds


def rmse_hidden(
    y_true: np.ndarray, y_pred: np.ndarray, hidden_mask: np.ndarray
) -> float:
    """RMSE evaluated only on rows where hidden_mask is True."""
    err = y_true[hidden_mask] - y_pred[hidden_mask]
    return float(np.sqrt(np.mean(err**2)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err**2)))


# ────────────────────────────────────────────────────────────────────────────
# Edge Q (typewell content-hash) helpers
# ────────────────────────────────────────────────────────────────────────────

_EDGE_Q_HASH_COLS_DEFAULT: Tuple[str, ...] = ("TVT", "GR", "Geology")


def compute_typewell_hashes(
    train_dir: Path,
    well_ids: Iterable[str],
    cols: Tuple[str, ...] = _EDGE_Q_HASH_COLS_DEFAULT,
) -> dict[str, str]:
    """Compute an MD5 hash of typewell columns for each well.

    Wells whose typewell file is missing or unreadable fall back to their own
    well_id (= singleton group). Schema must match
    ``kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py::compute_typewell_hashes``
    so that fold partitions remain identical between repo and kernel.
    """
    hashes: dict[str, str] = {}
    train_dir = Path(train_dir)
    for wid in well_ids:
        tw_path = train_dir / f"{wid}__typewell.csv"
        if not tw_path.exists():
            hashes[wid] = wid
            continue
        try:
            df = pd.read_csv(tw_path)
        except Exception:
            hashes[wid] = wid
            continue
        h = hashlib.md5()
        for col in cols:
            if col not in df.columns:
                continue
            if df[col].dtype == "object":
                payload = (
                    df[col].fillna("NaN").astype(str).str.cat(sep="|").encode()
                )
            else:
                payload = (
                    df[col].fillna(-99999.0).round(4).values.tobytes()
                )
            h.update(payload)
        hashes[wid] = h.hexdigest()
    return hashes


def build_edge_q_folds(
    train_df: pd.DataFrame,
    train_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    fallback_by_well: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fold_id, groups) using plain typewell-hash GroupKFold.

    Mirrors the exp007/exp009 kernel implementation for offline testing.
    The kernel keeps its own inline copy (= no runtime import).
    """
    well_ids = train_df["well"].unique().tolist()
    try:
        hashes = compute_typewell_hashes(train_dir, well_ids)
    except Exception as e:
        if not fallback_by_well:
            raise
        print(
            f"  [Edge Q] hash computation failed ({type(e).__name__}: {e}) "
            f"→ fallback to GroupKFold(by well)"
        )
        hashes = {wid: wid for wid in well_ids}

    groups = train_df["well"].map(hashes).fillna("__unknown__").values
    cv = GroupKFold(n_splits=n_splits)
    fold_id = np.full(len(train_df), -1, dtype=np.int8)
    for k, (_, va_idx) in enumerate(
        cv.split(train_df, train_df["target"], groups)
    ):
        fold_id[va_idx] = k
    if (fold_id < 0).any():
        bad = np.where(fold_id < 0)[0]
        for i in bad:
            fold_id[i] = abs(hash(str(groups[i]))) % n_splits
    return fold_id.astype(np.int8), np.asarray(groups, dtype=object)


def verify_edge_q_no_leak(
    fold_id: np.ndarray, groups: np.ndarray
) -> Tuple[bool, str]:
    """Assert no group_id appears in more than one fold (= no fold leak)."""
    df = pd.DataFrame({"fold": fold_id, "group": groups})
    grp_to_folds = df.groupby("group")["fold"].nunique()
    bad = grp_to_folds[grp_to_folds > 1]
    if len(bad) > 0:
        return False, (
            f"LEAK: {len(bad)} groups span multiple folds: "
            f"{bad.head(3).to_dict()}"
        )
    return True, (
        f"OK: {df['group'].nunique()} groups, "
        f"{int(fold_id.max()) + 1} folds, no leak"
    )


# ────────────────────────────────────────────────────────────────────────────
# Stratified Edge Q fold (exp010 = fold reform)
# ────────────────────────────────────────────────────────────────────────────


def build_stratified_edge_q_folds(
    train_df: pd.DataFrame,
    train_dir: Path,
    per_well_stats_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    stratify_key: str = "tw_gr_resid_std",
    n_bins: int = 4,
    fallback_by_well: bool = True,
    well_col: str = "well",
    stats_well_col: str = "well_id",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fold_id, groups) using stratified typewell-hash GroupKFold.

    Algorithm:
        1. Compute typewell content-hash per well (= existing Edge Q logic).
           Same hash → same group → forced into the same fold (= no leak).
        2. For each group, pick a representative well and look up its
           ``stratify_key`` value in ``per_well_stats_df`` (default key is
           ``tw_gr_resid_std`` from per-well-stats.parquet).
        3. Bin the group-representative values into ``n_bins`` quantile bins
           (default = quartile).
        4. Within each bin, shuffle the group list with ``seed`` and assign
           groups to folds round-robin. This balances the bin distribution
           across folds while keeping every group atomic.

    Mathematical motivation:
        For stratified sampling with K bins of comparable size, the
        between-fold variance of any per-bin metric is reduced from the
        unstratified variance σ² to ~σ²/K in the best case. exp007 observed
        σ_fold = 1.175 ft for per-fold RMSE; with K = 4 quartiles we target
        σ_fold ≤ 0.5 ft after stratification.

    Leak guard:
        - ``per_well_stats_df`` must only reference train wells. Test wells
          are never used for stratification.
        - The hash group constraint dominates: stratification operates on
          groups, never on rows.
        - Hidden-region TVT (= target) is irrelevant here. ``stratify_key``
          is a well-level statistic from per-well-stats.parquet which is
          computed from visible-region GR vs typewell only (no TVT leak).

    Fallback:
        If ``stratify_key`` is missing from ``per_well_stats_df`` or all
        values are NaN, falls back to the unstratified ``build_edge_q_folds``
        with a warning print.
    """
    well_ids = train_df[well_col].unique().tolist()
    try:
        hashes = compute_typewell_hashes(train_dir, well_ids)
    except Exception as e:
        if not fallback_by_well:
            raise
        print(
            f"  [stratified Edge Q] hash computation failed "
            f"({type(e).__name__}: {e}) → fallback to GroupKFold(by well)"
        )
        hashes = {wid: wid for wid in well_ids}

    if stratify_key not in per_well_stats_df.columns:
        print(
            f"  [stratified Edge Q] stratify_key={stratify_key!r} not in "
            f"per-well-stats → falling back to plain build_edge_q_folds"
        )
        return build_edge_q_folds(
            train_df,
            train_dir,
            n_splits=n_splits,
            seed=seed,
            fallback_by_well=fallback_by_well,
        )

    stats_lookup = dict(
        zip(
            per_well_stats_df[stats_well_col].astype(str),
            per_well_stats_df[stratify_key].astype(float),
        )
    )

    # group → list[(well_id, stratify_value)]
    group_to_wells: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for wid in well_ids:
        g = hashes.get(wid, wid)
        v = stats_lookup.get(str(wid), np.nan)
        group_to_wells[g].append((wid, v))

    # representative stratify value per group = median of member wells
    group_records: list[tuple[str, float]] = []
    n_nan = 0
    for g, members in group_to_wells.items():
        vals = np.array([v for _, v in members], dtype=float)
        if np.isnan(vals).all():
            group_records.append((g, np.nan))
            n_nan += 1
        else:
            group_records.append((g, float(np.nanmedian(vals))))

    valid_records = [(g, v) for g, v in group_records if not np.isnan(v)]
    nan_records = [(g, v) for g, v in group_records if np.isnan(v)]

    if len(valid_records) < n_bins * n_splits:
        print(
            f"  [stratified Edge Q] only {len(valid_records)} valid groups vs "
            f"{n_bins * n_splits} needed → falling back to plain Edge Q"
        )
        return build_edge_q_folds(
            train_df,
            train_dir,
            n_splits=n_splits,
            seed=seed,
            fallback_by_well=fallback_by_well,
        )

    values = np.array([v for _, v in valid_records], dtype=float)
    quantile_edges = np.quantile(
        values, np.linspace(0, 1, n_bins + 1)[1:-1]
    )

    bin_to_groups: dict[int, list[str]] = defaultdict(list)
    for g, v in valid_records:
        b = int(np.searchsorted(quantile_edges, v, side="right"))
        b = min(b, n_bins - 1)
        bin_to_groups[b].append(g)

    rng = np.random.RandomState(seed)
    group_to_fold: dict[str, int] = {}
    for b in range(n_bins):
        bucket = list(bin_to_groups[b])
        rng.shuffle(bucket)
        # round-robin assignment: cycle fold offset per-bin so consecutive
        # bins do not pile up on the same fold.
        offset = b % n_splits
        for i, g in enumerate(bucket):
            group_to_fold[g] = (i + offset) % n_splits

    # Distribute NaN-only groups uniformly via hash
    for g, _ in nan_records:
        group_to_fold[g] = abs(hash(str(g))) % n_splits

    groups = train_df[well_col].map(hashes).fillna("__unknown__").values
    fold_id = np.array(
        [group_to_fold.get(g, abs(hash(str(g))) % n_splits) for g in groups],
        dtype=np.int8,
    )
    return fold_id, np.asarray(groups, dtype=object)


def summarize_fold_stratification(
    fold_id: np.ndarray,
    groups: np.ndarray,
    train_df: pd.DataFrame,
    per_well_stats_df: pd.DataFrame,
    stratify_key: str = "tw_gr_resid_std",
    n_bins: int = 4,
    well_col: str = "well",
    stats_well_col: str = "well_id",
) -> pd.DataFrame:
    """Return a fold × bin count matrix for sanity-check.

    Each cell = number of wells in the given (fold, bin) cell. A balanced
    stratification produces ~equal counts within each row.
    """
    stats_lookup = dict(
        zip(
            per_well_stats_df[stats_well_col].astype(str),
            per_well_stats_df[stratify_key].astype(float),
        )
    )
    # Use validation rows only — one row per well — to count groups, not rows.
    well_to_fold = (
        pd.DataFrame({"well": train_df[well_col].values, "fold": fold_id})
        .drop_duplicates(subset=["well"])
    )
    well_to_fold["value"] = (
        well_to_fold["well"].astype(str).map(stats_lookup)
    )
    valid = well_to_fold.dropna(subset=["value"]).copy()
    if len(valid) == 0:
        return pd.DataFrame()
    edges = np.quantile(
        valid["value"].values, np.linspace(0, 1, n_bins + 1)[1:-1]
    )
    valid["bin"] = np.minimum(
        np.searchsorted(edges, valid["value"].values, side="right"),
        n_bins - 1,
    )
    pivot = (
        valid.groupby(["fold", "bin"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return pivot
