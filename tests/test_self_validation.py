"""Phase Magic-1 / Track 3.1: Train-as-test bootstrap self-validation tests.

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §I M1.
Goal: train 770 wells を 770 synthetic test として使い、 per-row systematic bias を抽出。

Tests:
  - mask_visible_to_ratio: visible_ratio target で trailing rows を NaN にする
  - bootstrap_per_well_eval: 1 well × N mask iteration で error stats 集約
  - aggregate_systematic_bias: 全 wells で row-position-binned bias profile を生成
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.self_validation import (
    mask_visible_to_ratio,
    bootstrap_per_well_eval,
    aggregate_systematic_bias,
)


def make_synthetic_well(n=200, vis_ratio=0.3, seed=42, slope=0.05):
    """Generate a synthetic well with known TVT trajectory."""
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float)
    tvt_true = 1000.0 + slope * md + rng.normal(0, 0.01, n)  # smooth trajectory
    gr = 80.0 + 20 * np.sin(md / 10) + rng.normal(0, 2, n)
    df = pd.DataFrame({
        'MD': md,
        'GR': gr,
        'X': np.full(n, 100.0),
        'Y': np.full(n, 200.0),
        'Z': -md * 0.5,
        'TVT': tvt_true,
        'TVT_input': tvt_true.copy(),
    })
    # Mask trailing (1 - vis_ratio) rows
    cut = int(n * vis_ratio)
    df.loc[df.index[cut:], 'TVT_input'] = np.nan
    return df


def test_mask_visible_to_ratio_basic():
    df = make_synthetic_well(n=100, vis_ratio=0.50)
    # df starts with vis_ratio 0.50; re-mask to 0.30
    masked = mask_visible_to_ratio(df, target_ratio=0.30)
    n_vis = masked['TVT_input'].notna().sum()
    # 30 rows visible
    assert 28 <= n_vis <= 32, f"expected ~30 visible, got {n_vis}"


def test_mask_visible_to_ratio_preserves_tvt_truth():
    df = make_synthetic_well(n=100, vis_ratio=0.50)
    masked = mask_visible_to_ratio(df, target_ratio=0.30)
    # TVT column unchanged (= ground truth preserved)
    assert masked['TVT'].equals(df['TVT'])


def test_mask_visible_to_ratio_smaller_skips():
    # Already 0.30, request 0.50 — should keep at 0.30 (= cannot expand visible)
    df = make_synthetic_well(n=100, vis_ratio=0.30)
    masked = mask_visible_to_ratio(df, target_ratio=0.50, expand_ok=False)
    n_vis = masked['TVT_input'].notna().sum()
    assert 28 <= n_vis <= 32, f"should not expand, got {n_vis}"


def test_bootstrap_per_well_eval_returns_dataframe():
    df = make_synthetic_well(n=200, vis_ratio=0.50)

    def naive_predictor(df_masked):
        """Naive: continue last visible TVT slope linearly."""
        vis = df_masked[df_masked['TVT_input'].notna()]
        if len(vis) < 2:
            return df_masked
        slope = (vis['TVT_input'].iloc[-1] - vis['TVT_input'].iloc[-50:].mean()) / 50
        anchor = vis['TVT_input'].iloc[-1]
        last_idx = vis.index[-1]
        out = df_masked.copy()
        hid_mask = out['TVT_input'].isna()
        hid_idx = out.index[hid_mask]
        out.loc[hid_idx, 'TVT_pred'] = anchor + slope * np.arange(1, len(hid_idx) + 1)
        return out

    eval_df = bootstrap_per_well_eval(
        df,
        target_ratios=[0.20, 0.30, 0.40],
        predictor=naive_predictor,
        seed=42,
    )
    assert isinstance(eval_df, pd.DataFrame)
    assert {'target_ratio', 'row_position_pct', 'error'}.issubset(eval_df.columns)
    assert len(eval_df) > 0


def test_aggregate_systematic_bias_returns_profile():
    # Mock per-well eval results
    rng = np.random.default_rng(0)
    rows = []
    for w in range(5):
        for rp in np.linspace(0, 1, 20):
            for tr in [0.20, 0.30]:
                rows.append({
                    'well': f'w{w}',
                    'target_ratio': tr,
                    'row_position_pct': rp,
                    'error': rng.normal(0.1, 0.5),
                })
    eval_df = pd.DataFrame(rows)
    profile = aggregate_systematic_bias(eval_df, n_bins=10)
    assert 'row_bin' in profile.columns
    assert 'mean_error' in profile.columns
    assert 'std_error' in profile.columns
    assert len(profile) <= 10 * 2  # 10 bins × 2 ratios


def test_bootstrap_handles_short_wells():
    """Wells with < 60 visible should skip cleanly."""
    df = make_synthetic_well(n=30, vis_ratio=0.50)

    def naive(df_m):
        return df_m
    eval_df = bootstrap_per_well_eval(df, target_ratios=[0.30], predictor=naive, seed=42)
    # Too short — empty or with skip marker
    assert isinstance(eval_df, pd.DataFrame)
