"""Cross-validation for ROGII.

Strategy ladder:
- v1 (= exp001-006): plain GroupKFold by well — each well lives in exactly
  one validation fold so target leak across wells is impossible.
- v2 (= exp007 / Edge Q): typewell content-hash GroupKFold — wells sharing
  the same (TVT, GR, Geology) typewell file (= pseudo-typewell group of
  35 wells / 13 groups, host-confirmed) are forced into the same fold.
- v3 (= exp010 / fold reform): **single-key stratified** typewell-hash
  GroupKFold — fold partition additionally balances per-well difficulty
  (default key = ``tw_gr_resid_std`` quartile). σ_fold target 0.5 ft.
- v4 (= 2026-05-11 / 4 CV final): the 4 structurally-distinct CV strategies
  evaluated against public LB Spearman/Pearson to pick the best one before
  measuring winning-path candidates.

    C1  build_pseudo_test_fold                 (kNN-based test-similar val)
    C2  build_multi_key_stratified_edge_q_folds (multi-axis stratification)
    C3  see ``rogii.adversarial`` module       (adversarial validation drop)
    C4  build_edge_q_folds + compute_b_cluster_balance (post-hoc balance)

See ``.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml`` for the success
contract and ``docs/research/2026-05-11-cv-strategy-candidates.dense.md``
for the design rationale.

Validation metric is RMSE on hidden rows of held-out wells only, mirroring
the public leaderboard scoring exactly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

# Canonical hash + adversarial implementations live in dedicated modules.
# Re-exported below for backward-compat with callers that still
# ``from rogii.cv import compute_typewell_hashes``.
from rogii.typewell_hash import (  # noqa: E402
    HASH_COLS_DEFAULT as _EDGE_Q_HASH_COLS_DEFAULT,
    compute_typewell_hashes,
)
from rogii.adversarial import (  # noqa: E402
    apply_adversarial_drop_fold,
    compute_adversarial_classifier_oof,
)


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
#
# ``compute_typewell_hashes`` lives in ``rogii.typewell_hash`` and is
# re-exported at module top so callers using ``from rogii.cv import ...``
# continue to work. Kernel scripts keep an inline copy for self-containedness.
# ────────────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────────────
# C1 — Pseudo-test fold via kNN (= test-similar wells fixed to fold 0)
# ────────────────────────────────────────────────────────────────────────────


def compute_test_distance_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    pca_components: int = 10,
    pca_variance_min: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, float]:
    """Z-score normalize → PCA top-K project on (train_features, test_features).

    Args:
        train_features: per-well features for train wells, indexed by well_id.
        test_features: per-well features for test wells, indexed by well_id.
        feature_cols: column subset to use. None → intersection of numeric cols.
        pca_components: top-K principal components to keep.
        pca_variance_min: required cumulative variance ratio; below this →
            raise (= signal that PCA dimensionality is too small to be safe).

    Returns:
        ``(train_pca, test_pca, train_index, test_index, variance_ratio_sum)``
        where ``*_pca`` are (n, pca_components) float arrays.
    """
    if feature_cols is None:
        common = train_features.columns.intersection(test_features.columns)
        feature_cols = [
            c for c in common
            if pd.api.types.is_numeric_dtype(train_features[c])
            and not pd.api.types.is_bool_dtype(train_features[c])
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("no numeric feature_cols in common between train and test")

    Xtr = train_features[feature_cols].copy()
    Xte = test_features[feature_cols].copy()

    # Z-score on train stats (= robust = median + 1.4826*MAD-like, but stick
    # to plain mean/std for tractability; fill NaN with col median).
    medians = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(medians)
    Xte = Xte.fillna(medians)
    mu = Xtr.mean()
    sigma = Xtr.std().replace(0.0, 1.0)  # zero-variance cols → no scaling
    Xtr_z = (Xtr - mu) / sigma
    Xte_z = (Xte - mu) / sigma

    n_components = min(pca_components, len(feature_cols), len(Xtr_z))
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(Xtr_z.values)
    test_pca = pca.transform(Xte_z.values)
    variance_ratio_sum = float(pca.explained_variance_ratio_.sum())

    if variance_ratio_sum < pca_variance_min:
        raise ValueError(
            f"PCA variance ratio = {variance_ratio_sum:.3f} < "
            f"{pca_variance_min:.3f} required; refusing to build a fragile "
            f"distance metric on top-{n_components} components"
        )

    return train_pca, test_pca, Xtr_z.index, Xte_z.index, variance_ratio_sum


def build_pseudo_test_fold(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    k_per_test_well: int = 50,
    pca_components: int = 10,
    pca_variance_min: float = 0.5,
    feature_cols: Sequence[str] | None = None,
    well_col: str = "well",
) -> Tuple[np.ndarray, np.ndarray]:
    """C1: build fold 0 = wells most similar to test (= kNN aggregate).

    Fold 0 is the **pseudo-test fold**: union of top-``k_per_test_well``
    nearest train wells around each test well in PCA-projected feature
    space. Folds 1..n_splits-1 are the remaining train wells round-robined
    deterministically.

    This makes fold 0 RMSE the best proxy for true public-LB RMSE; the
    other folds measure model generalization on "typical" train wells.

    Args:
        train_df: long-format train DataFrame with ``well_col``.
        train_features: per-well features (index = well_id).
        test_features: per-well features for the 3 test wells (index = well_id).
        n_splits: total folds (default 5; fold 0 = pseudo-test).
        seed: shuffle seed for fold 1..n_splits-1.
        k_per_test_well: how many train wells per test well are pulled
            into fold 0 (default 50 → fold 0 size ≈ K * n_test_wells if
            no overlap, less if overlap).
        pca_components: PCA dimensions for distance computation.
        pca_variance_min: minimum cumulative variance ratio (else raises).
        feature_cols: feature subset; None → all numeric intersect.
        well_col: column in train_df with well_id.

    Returns:
        ``(fold_id, groups)`` arrays of length len(train_df).
            - fold 0 = pseudo-test (= test-similar wells)
            - fold 1..n_splits-1 = random partition of remaining wells
            - groups = well_id (= same as fold for this strategy; satisfies
              the verify_edge_q_no_leak API).
    """
    train_pca, test_pca, train_idx, _test_idx, _vr = (
        compute_test_distance_features(
            train_features, test_features,
            feature_cols=feature_cols,
            pca_components=pca_components,
            pca_variance_min=pca_variance_min,
        )
    )

    # Map train_idx (= well_id) → row in train_pca
    train_well_to_row = {str(w): i for i, w in enumerate(train_idx)}

    # For each test well, find top-K nearest train wells by euclidean dist
    pseudo_test_wells: set[str] = set()
    for ti in range(test_pca.shape[0]):
        diffs = train_pca - test_pca[ti : ti + 1]
        d2 = (diffs * diffs).sum(axis=1)
        order = np.argsort(d2)
        for j in order[: k_per_test_well]:
            pseudo_test_wells.add(str(train_idx[j]))

    # Remaining train wells = round-robined to folds 1..n_splits-1
    rng = np.random.RandomState(seed)
    remaining = sorted(
        w for w in train_well_to_row.keys() if w not in pseudo_test_wells
    )
    rng.shuffle(remaining)
    n_other = n_splits - 1
    well_to_fold: dict[str, int] = {w: 0 for w in pseudo_test_wells}
    for i, w in enumerate(remaining):
        well_to_fold[w] = 1 + (i % max(n_other, 1))

    # Map back to rows
    fold_id = np.array(
        [
            well_to_fold.get(str(w), abs(hash(str(w))) % n_splits)
            for w in train_df[well_col].astype(str).values
        ],
        dtype=np.int8,
    )
    groups = train_df[well_col].astype(str).values
    return fold_id, np.asarray(groups, dtype=object)


# ────────────────────────────────────────────────────────────────────────────
# C2 — Multi-key stratified Edge Q (= multi-axis bin × typewell hash group)
# ────────────────────────────────────────────────────────────────────────────


def _bin_well_by_keys(
    train_df: pd.DataFrame,
    per_well_stats_df: pd.DataFrame,
    stratify_specs: Sequence[Tuple[str, str, int]],
    well_col: str = "well",
    stats_well_col: str = "well_id",
) -> dict[str, int]:
    """Bin each well into a composite stratum index from multiple keys.

    ``stratify_specs`` is a list of ``(stats_col, bin_strategy, n_bins)``:
        - bin_strategy = "quantile": split into n_bins equal-frequency bins.
        - bin_strategy = "categorical": use value as-is (n_bins ignored).

    Returns ``{well_id: composite_stratum_index}``. Wells with NaN on any
    axis are assigned to a sentinel stratum (= the last index).
    """
    well_ids = (
        pd.DataFrame({"well": train_df[well_col].astype(str).values})
        .drop_duplicates()["well"]
        .tolist()
    )
    stats_indexed = per_well_stats_df.set_index(
        per_well_stats_df[stats_well_col].astype(str)
    )
    axis_codes: list[np.ndarray] = []
    axis_sizes: list[int] = []
    for stats_col, strategy, n_bins in stratify_specs:
        if stats_col not in stats_indexed.columns:
            # Missing column → constant axis = 0
            axis_codes.append(np.zeros(len(well_ids), dtype=np.int32))
            axis_sizes.append(1)
            continue
        vals = stats_indexed.loc[well_ids, stats_col].values
        if strategy == "quantile":
            valid_vals = pd.Series(vals).dropna().values
            if len(valid_vals) < n_bins:
                codes = np.zeros(len(well_ids), dtype=np.int32)
                axis_codes.append(codes)
                axis_sizes.append(1)
                continue
            edges = np.quantile(
                valid_vals, np.linspace(0, 1, n_bins + 1)[1:-1]
            )
            codes = np.searchsorted(edges, vals, side="right")
            codes = np.minimum(codes, n_bins - 1)
            # NaN → last bin
            codes = np.where(pd.isna(vals), n_bins - 1, codes).astype(np.int32)
            axis_codes.append(codes)
            axis_sizes.append(n_bins)
        elif strategy == "categorical":
            uniq = pd.unique(pd.Series(vals).fillna("__nan__"))
            cat_map = {v: i for i, v in enumerate(sorted(uniq, key=lambda x: str(x)))}
            codes = np.array(
                [cat_map.get(v if pd.notna(v) else "__nan__", 0) for v in vals],
                dtype=np.int32,
            )
            axis_codes.append(codes)
            axis_sizes.append(len(cat_map))
        else:
            raise ValueError(
                f"unknown bin_strategy={strategy!r} for col={stats_col!r}; "
                "expected 'quantile' or 'categorical'"
            )

    # Composite stratum = base-N encoding of (axis_0, axis_1, ...)
    composite = np.zeros(len(well_ids), dtype=np.int32)
    for codes, size in zip(axis_codes, axis_sizes):
        composite = composite * size + codes
    return dict(zip(well_ids, composite.tolist()))


def build_multi_key_stratified_edge_q_folds(
    train_df: pd.DataFrame,
    train_dir: Path,
    per_well_stats_df: pd.DataFrame,
    stratify_specs: Sequence[Tuple[str, str, int]] = (
        ("visible_ratio", "quantile", 5),
        ("tw_gr_resid_std", "quantile", 4),
    ),
    n_splits: int = 5,
    seed: int = 42,
    fallback_by_well: bool = True,
    well_col: str = "well",
    stats_well_col: str = "well_id",
) -> Tuple[np.ndarray, np.ndarray]:
    """C2: stratified Edge Q with **multiple** stratification axes.

    Generalizes ``build_stratified_edge_q_folds`` (= single-key) to any list
    of axes. Each well gets a composite stratum index (= base-N encoding
    over all axes), groups (= typewell hash) inherit their members'
    majority stratum, and groups within each stratum are round-robined
    over folds with a per-stratum offset.

    Default ``stratify_specs`` balances visible_ratio (5 quantile bins) and
    tw_gr_resid_std (4 quantile bins) = 20 strata, so each fold sees a
    full spread of (well duration, well difficulty). Override to taste:

        stratify_specs=[
            ("visible_ratio",       "quantile",    5),
            ("tw_gr_resid_std",     "quantile",    4),
            ("ar1_phi",             "quantile",    2),  # add 3rd axis
        ]

    Args:
        train_df: long-format train DataFrame with ``well_col``.
        train_dir: directory containing ``{wid}__typewell.csv``.
        per_well_stats_df: per-well-stats parquet (= columns referenced in
            ``stratify_specs``).
        stratify_specs: list of (col, strategy, n_bins).
        n_splits: number of folds.
        seed: shuffle seed.
        fallback_by_well: if hash computation fails, fall back to plain
            ``GroupKFold(by well)``.
        well_col, stats_well_col: column names.

    Returns:
        ``(fold_id, groups)``.
    """
    well_ids = train_df[well_col].astype(str).unique().tolist()
    try:
        hashes = compute_typewell_hashes(train_dir, well_ids)
    except Exception as e:
        if not fallback_by_well:
            raise
        print(
            f"  [multi-key stratified Edge Q] hash failed "
            f"({type(e).__name__}: {e}) → fallback to GroupKFold(by well)"
        )
        hashes = {wid: wid for wid in well_ids}

    well_stratum = _bin_well_by_keys(
        train_df, per_well_stats_df, stratify_specs,
        well_col=well_col, stats_well_col=stats_well_col,
    )

    # Group → majority stratum of member wells
    group_to_wells: dict[str, list[str]] = defaultdict(list)
    for wid in well_ids:
        group_to_wells[hashes.get(wid, wid)].append(wid)

    group_stratum: dict[str, int] = {}
    for g, members in group_to_wells.items():
        strata = [well_stratum.get(w, 0) for w in members]
        # majority vote (ties broken by min)
        counts = pd.Series(strata).value_counts()
        group_stratum[g] = int(counts.index[0])

    # Each stratum: shuffle groups, round-robin over folds with per-stratum offset
    stratum_to_groups: dict[int, list[str]] = defaultdict(list)
    for g, s in group_stratum.items():
        stratum_to_groups[s].append(g)

    rng = np.random.RandomState(seed)
    group_to_fold: dict[str, int] = {}
    for s_idx, (_s, gs) in enumerate(sorted(stratum_to_groups.items())):
        order = list(gs)
        rng.shuffle(order)
        offset = s_idx % n_splits
        for i, g in enumerate(order):
            group_to_fold[g] = (i + offset) % n_splits

    groups = train_df[well_col].astype(str).map(hashes).fillna("__unknown__").values
    fold_id = np.array(
        [group_to_fold.get(g, abs(hash(str(g))) % n_splits) for g in groups],
        dtype=np.int8,
    )
    return fold_id, np.asarray(groups, dtype=object)


# ────────────────────────────────────────────────────────────────────────────
# C4 — Post-hoc b_cluster balance check
# (C4 = typewell hash GroupKFold (= build_edge_q_folds) + this audit)
# ────────────────────────────────────────────────────────────────────────────


def compute_b_cluster_balance(
    fold_id: np.ndarray,
    train_df: pd.DataFrame,
    b_cluster_lookup: dict[str, float | int | str],
    well_col: str = "well",
) -> Tuple[pd.DataFrame, float]:
    """Compute per-fold × b_cluster pivot + chi-square p-value.

    Audits whether typewell-hash GroupKFold (= C4 primary fold function
    ``build_edge_q_folds``) accidentally segregates b_clusters across
    folds. A balanced fold set has roughly uniform b_cluster ratios per
    fold; chi-square p-value > 0.05 indicates "no significant imbalance".

    Args:
        fold_id: per-row fold assignment, shape (n_rows,).
        train_df: DataFrame with ``well_col``.
        b_cluster_lookup: ``{well_id: cluster_id}``; cluster_id can be int
            (= b_cluster code), float (= raw b_ANCC value), or str.
        well_col: column name in train_df.

    Returns:
        ``(pivot, p_value)``:
            pivot: DataFrame indexed by fold, columns = b_cluster ids,
                values = well count (= distinct wells per cell).
            p_value: chi-square test p-value of independence (fold ⊥ cluster).
                NaN if contingency is too sparse for chi-square.
    """
    df = (
        pd.DataFrame({
            "fold": fold_id,
            "well": train_df[well_col].astype(str).values,
        })
        .drop_duplicates()
    )
    df["cluster"] = df["well"].map(b_cluster_lookup)
    df = df.dropna(subset=["cluster"])
    if df.empty:
        return pd.DataFrame(), float("nan")
    pivot = (
        df.groupby(["fold", "cluster"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    try:
        _chi2, p_value, _dof, _expected = _scipy_stats.chi2_contingency(
            pivot.values
        )
        p_value = float(p_value)
    except (ValueError, _scipy_stats._stats_py.DegenerateDataWarning):  # type: ignore[attr-defined]
        p_value = float("nan")
    return pivot, p_value


__all__ = [
    # baseline + Edge Q v1/v2/v3
    "make_well_folds",
    "rmse",
    "rmse_hidden",
    "compute_typewell_hashes",  # re-exported from typewell_hash
    "build_edge_q_folds",
    "verify_edge_q_no_leak",
    "build_stratified_edge_q_folds",
    "summarize_fold_stratification",
    # C1 — pseudo-test fold
    "compute_test_distance_features",
    "build_pseudo_test_fold",
    # C2 — multi-key stratified Edge Q
    "build_multi_key_stratified_edge_q_folds",
    # C3 — adversarial validation drop (= re-exported from adversarial)
    "compute_adversarial_classifier_oof",
    "apply_adversarial_drop_fold",
    # C4 — b_cluster post-hoc balance audit
    "compute_b_cluster_balance",
]
