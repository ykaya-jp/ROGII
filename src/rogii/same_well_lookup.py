"""Phase Magic-1 / Track 3.2: N1 same-well lookup (= paradigm 6 強化).

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §IV N1.

Key insight: ROGII の train + test data structure で **同 well_id 共有** が確認済
(= 親 doc 2026-05-13-train-test-well-id-overlap-discovery.dense.md)。 train data
の TVT は全 row public、 test data の hidden TVT は host 側のみ。

ROGII rule (= 2026-05-13 確認、 「hand labeling or human prediction of test data
records 禁止」、 train data records の automated ML 利用 OK) で N1 path 合法。

実装: per-row kNN で **same_well_allowed=True**、 ただし **exclude_same_row=True**
で self-loop 禁止。 train rows を index にして、 test rows を query すると、 同 well_id
を含む K 近傍から TVT median + std + n_same_well を feature 化。

期待 lift: -0.10 〜 -0.40 ft (= 親 doc § 1 N1 期待)、 test 00e12e8b の 1280 row +3.30
ft bias を半減 〜 解消できる可能性大。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GR_WINDOW_DEFAULT = 5
SIGNATURE_COLS_DEFAULT = ['gr_smooth', 'MD', 'Z']


def build_well_signature_index(
    train_df: pd.DataFrame,
    *,
    gr_window: int = GR_WINDOW_DEFAULT,
    signature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a per-row index over train wells with rolling-smoothed GR signature.

    Args:
        train_df: long-format train DataFrame, must contain
            ['well', 'row', 'GR', 'MD', 'Z', 'TVT'] columns
        gr_window: rolling window for GR smoothing (= depth-independent signature)
        signature_cols: columns to use for kNN distance (= defaults to gr_smooth+MD+Z)

    Returns:
        DataFrame with smoothed signature columns added, ready for kNN.
        Each row has 'well', 'row', 'TVT' (= ground truth) + signature columns.
    """
    required = {'well', 'row', 'GR', 'MD', 'Z', 'TVT'}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df missing required columns: {missing}")

    df = train_df[list(required)].copy().reset_index(drop=True)
    # Rolling GR smooth per well (= depth-independent local signature)
    df['gr_smooth'] = (
        df.groupby('well')['GR']
        .transform(lambda s: s.rolling(gr_window, center=True, min_periods=1).mean())
        .astype(np.float32)
    )
    # Keep only TVT-known rows (= train data: all rows have ground truth TVT)
    df = df[df['TVT'].notna()].reset_index(drop=True)
    return df


def _normalize_features(arr: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-normalize columns; return (normalized, mean, std). If ref given, use its mean/std."""
    if ref is not None:
        mu = ref.mean(axis=0)
        sigma = ref.std(axis=0)
    else:
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return (arr - mu) / sigma, mu, sigma


def lookup_same_well_tvt(
    query_df: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    same_well_allowed: bool,
    k: int = 5,
    exclude_same_row: bool = True,
    signature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """For each query row, find k nearest train rows in signature space.

    Args:
        query_df: query rows (e.g., test data hidden rows), must have 'well', 'row',
            'GR', 'MD', 'Z' columns. GR is rolled internally.
        index_df: output of build_well_signature_index
        same_well_allowed: if True, include same well_id in candidates (= N1 path);
            if False, exclude same well_id (= standard paradigm 6 leak guard)
        k: number of neighbors
        exclude_same_row: if True, exclude row that matches (well, row) exactly
            (= prevent self-loop when query well is also in train)
        signature_cols: kNN feature columns (default: gr_smooth + MD + Z)

    Returns:
        DataFrame matching query_df rows with new columns:
            'lookup_tvt_median', 'lookup_tvt_std', 'lookup_tvt_mean',
            'lookup_n_same_well', 'lookup_dist_median'
    """
    if signature_cols is None:
        signature_cols = SIGNATURE_COLS_DEFAULT

    required_q = {'well', 'row', 'GR', 'MD', 'Z'}
    missing = required_q - set(query_df.columns)
    if missing:
        raise ValueError(f"query_df missing required columns: {missing}")

    q = query_df.copy().reset_index(drop=True)
    # Rolling smooth GR per well in query
    q['gr_smooth'] = (
        q.groupby('well')['GR']
        .transform(lambda s: s.rolling(GR_WINDOW_DEFAULT, center=True, min_periods=1).mean())
        .astype(np.float32)
    )

    # Build kNN tree on normalized signature
    idx_features = index_df[signature_cols].to_numpy(np.float32)
    idx_norm, mu, sigma = _normalize_features(idx_features)
    tree = cKDTree(idx_norm)

    q_features = q[signature_cols].to_numpy(np.float32)
    q_norm = (q_features - mu) / sigma

    # Fetch more than k to allow filtering (= same-well exclusion, same-row exclusion)
    nfetch = min(k * 5 + 10, len(idx_norm))
    dists, indices = tree.query(q_norm, k=nfetch, workers=-1)

    # For each query row, filter candidates per same_well + same_row constraints
    out_rows = []
    idx_well = index_df['well'].to_numpy()
    idx_row = index_df['row'].to_numpy()
    idx_tvt = index_df['TVT'].to_numpy(np.float32)
    q_well = q['well'].to_numpy()
    q_row = q['row'].to_numpy()

    for qi in range(len(q)):
        cand_idx = indices[qi]
        cand_dist = dists[qi]
        same_well_mask = idx_well[cand_idx] == q_well[qi]
        same_row_mask = (idx_well[cand_idx] == q_well[qi]) & (idx_row[cand_idx] == q_row[qi])

        keep = np.ones(len(cand_idx), dtype=bool)
        if not same_well_allowed:
            keep &= ~same_well_mask
        if exclude_same_row:
            keep &= ~same_row_mask

        kept_idx = cand_idx[keep][:k]
        kept_dist = cand_dist[keep][:k]
        if len(kept_idx) == 0:
            out_rows.append({
                'lookup_tvt_median': np.nan,
                'lookup_tvt_std': np.nan,
                'lookup_tvt_mean': np.nan,
                'lookup_n_same_well': 0,
                'lookup_dist_median': np.nan,
            })
            continue

        tvts = idx_tvt[kept_idx]
        n_same_well = int((idx_well[kept_idx] == q_well[qi]).sum())
        out_rows.append({
            'lookup_tvt_median': float(np.nanmedian(tvts)),
            'lookup_tvt_std': float(np.nanstd(tvts)),
            'lookup_tvt_mean': float(np.nanmean(tvts)),
            'lookup_n_same_well': n_same_well,
            'lookup_dist_median': float(np.median(kept_dist)),
        })

    out_df = pd.DataFrame(out_rows)
    return pd.concat([q.reset_index(drop=True), out_df], axis=1)
