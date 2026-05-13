"""Phase Magic-1 / Track 3.1: Train-as-test bootstrap self-validation.

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §I M1.

Idea: train 770 wells を 770 個の synthetic test として再利用。 各 well で
visible_ratio を test と同じ帯 (= 20-33%) に **re-mask**、 残り hidden を予測、
真値 (= TVT ground truth、 train data では全 row public) と比較。
per-row position × visible_ratio bucket で systematic bias を抽出。

これにより test 3 wells では見えない pattern (= 770 wells aggregate) を統計化、
LGB に「per-row position-aware bias correction」 を inject 可能。

Output: row_position_pct (= 0.0-1.0 = hidden 開始からの相対位置) × target_ratio
(= 0.20, 0.27, 0.33) × error 分布。
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


def mask_visible_to_ratio(
    df: pd.DataFrame,
    target_ratio: float,
    *,
    expand_ok: bool = False,
) -> pd.DataFrame:
    """Re-mask trailing rows so visible_ratio matches target.

    Args:
        df: well DataFrame with 'TVT_input' column (= visible TVT、 hidden は NaN)
        target_ratio: target visible_ratio in (0, 1)
        expand_ok: if False (default), do not expand visible region beyond original;
            if True, requesting a higher ratio raises (we cannot create labeled rows).

    Returns:
        Copy of df with TVT_input re-masked (= trailing rows after target_ratio*n set to NaN).
        TVT column (= ground truth) unchanged.
    """
    if not (0.0 < target_ratio < 1.0):
        raise ValueError(f"target_ratio must be in (0,1), got {target_ratio}")
    n = len(df)
    boundary = int(n * target_ratio)
    out = df.copy()
    if 'TVT_input' not in out.columns:
        raise ValueError("df must have 'TVT_input' column")
    current_vis = out['TVT_input'].notna().sum()
    current_ratio = current_vis / n
    if target_ratio > current_ratio and not expand_ok:
        # Cannot expand visible — return as-is (= keep existing mask).
        return out
    # Mask trailing rows beyond boundary
    out.loc[out.index[boundary:], 'TVT_input'] = np.nan
    return out


def bootstrap_per_well_eval(
    well_df: pd.DataFrame,
    target_ratios: Iterable[float],
    predictor: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    min_visible: int = 30,
    min_hidden: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Run bootstrap eval on a single well at multiple target_ratio.

    Args:
        well_df: full well DataFrame (= train, all TVT public)
        target_ratios: iterable of visible_ratio targets (e.g., [0.20, 0.27, 0.33])
        predictor: function (df_masked) -> df with column 'TVT_pred' added on hidden rows
        min_visible / min_hidden: skip if too short
        seed: reserved (= deterministic ordering, not currently random-sampled)

    Returns:
        long-format DataFrame with columns:
          ['target_ratio', 'row_position_pct', 'error', 'tvt_true', 'tvt_pred']
        row_position_pct = (row - hidden_start) / hidden_length, in [0, 1].
    """
    if 'TVT' not in well_df.columns or 'TVT_input' not in well_df.columns:
        return pd.DataFrame(columns=['target_ratio', 'row_position_pct', 'error', 'tvt_true', 'tvt_pred'])

    out_rows: list[dict] = []
    for tr in target_ratios:
        masked = mask_visible_to_ratio(well_df, tr, expand_ok=False)
        vis = masked[masked['TVT_input'].notna()]
        hid = masked[masked['TVT_input'].isna()]
        if len(vis) < min_visible or len(hid) < min_hidden:
            continue
        # Run predictor (= must return df with 'TVT_pred' column on hidden rows)
        pred_df = predictor(masked)
        if 'TVT_pred' not in pred_df.columns:
            continue
        # Match hidden rows
        hid_pred = pred_df.loc[hid.index, 'TVT_pred']
        hid_true = well_df.loc[hid.index, 'TVT']
        valid = hid_pred.notna() & hid_true.notna()
        if valid.sum() == 0:
            continue
        hid_start_idx = hid.index[0]
        hid_len = len(hid)
        for idx in hid.index[valid]:
            rp = (idx - hid_start_idx) / max(hid_len - 1, 1)
            out_rows.append({
                'target_ratio': tr,
                'row_position_pct': float(rp),
                'error': float(hid_pred[idx] - hid_true[idx]),
                'tvt_true': float(hid_true[idx]),
                'tvt_pred': float(hid_pred[idx]),
            })

    return pd.DataFrame(out_rows)


def aggregate_systematic_bias(
    eval_df: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Aggregate per-row eval into binned (target_ratio × row_position_bin) bias profile.

    Returns:
        DataFrame with columns ['target_ratio', 'row_bin', 'mean_error', 'std_error', 'n'].
    """
    if eval_df.empty:
        return pd.DataFrame(columns=['target_ratio', 'row_bin', 'mean_error', 'std_error', 'n'])
    df = eval_df.copy()
    df['row_bin'] = pd.cut(
        df['row_position_pct'],
        bins=np.linspace(0, 1, n_bins + 1),
        labels=[f'B{i}' for i in range(n_bins)],
        include_lowest=True,
    )
    agg = (
        df.groupby(['target_ratio', 'row_bin'], observed=True)
        .agg(mean_error=('error', 'mean'), std_error=('error', 'std'), n=('error', 'size'))
        .reset_index()
    )
    return agg
