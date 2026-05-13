"""Phase Magic-1 / Track 3.2: N1 same-well lookup paradigm 6 強化 tests.

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §IV N1.
Goal: train + test で 同 well_id 共有 (= 親 doc train-test-overlap で発見)、 train hidden
actual を per-row kNN で **直接 lookup** (= machine learning、 rule 合法)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.same_well_lookup import (
    build_well_signature_index,
    lookup_same_well_tvt,
)


def make_well_df(well_id, n=200, slope=0.05, gr_amp=20.0, seed=42, x_offset=0.0, gr_phase=0.0):
    """Make synthetic well. x_offset + gr_phase の差で wells を distinguishable に."""
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float)
    tvt = 1000 + slope * md + rng.normal(0, 0.01, n)
    gr = 80 + gr_amp * np.sin(md / 10 + gr_phase) + rng.normal(0, 2, n)
    df = pd.DataFrame({
        'well': well_id,
        'row': np.arange(n),
        'MD': md + x_offset,  # well 別に MD shift
        'GR': gr,
        'TVT': tvt,
        'TVT_input': tvt.copy(),
        'Z': -md * 0.5,
        'X': np.full(n, 100.0 + x_offset * 10),  # well 別 X
        'Y': np.full(n, 200.0),
    })
    # Make hidden region (last 70%)
    df.loc[df.index[int(n * 0.3):], 'TVT_input'] = np.nan
    return df


def test_build_well_signature_index_basic():
    """Index で全 train well rows (= visible + hidden、 TVT public) を kNN index 化."""
    train_wells = [make_well_df(f'w{i}', seed=i) for i in range(3)]
    full = pd.concat(train_wells, ignore_index=True)
    index = build_well_signature_index(full, gr_window=5)
    # 3 wells × 200 rows = 600 rows in index
    assert index.shape[0] == 600, f"expected 600 rows, got {index.shape[0]}"
    # Required columns
    for col in ['well', 'row', 'TVT', 'gr_smooth', 'MD', 'Z']:
        assert col in index.columns, f"missing column: {col}"


def test_lookup_same_well_returns_tvt_neighbors():
    """test well と同 well_id の train rows を直接 lookup、 same-row 除外."""
    train_wells = [
        make_well_df('w0', seed=0, x_offset=0.0, gr_phase=0.0),
        make_well_df('w1', seed=1, x_offset=10000.0, gr_phase=2.0),  # far MD
        make_well_df('w2', seed=2, x_offset=20000.0, gr_phase=4.0),
    ]
    full = pd.concat(train_wells, ignore_index=True)
    index = build_well_signature_index(full)
    # Query rows from w0 (= same well as one in train)
    query_rows = make_well_df('w0', seed=0, x_offset=0.0, gr_phase=0.0).iloc[100:105].copy()
    result = lookup_same_well_tvt(
        query_rows,
        index,
        same_well_allowed=True,
        k=5,
        exclude_same_row=True,
    )
    assert 'lookup_tvt_median' in result.columns
    assert 'lookup_tvt_std' in result.columns
    assert 'lookup_n_same_well' in result.columns
    assert len(result) == 5
    # well 別 MD/X 大きく違う → MD nearest は same well 優位、 same-well neighbors > 3 期待
    assert result['lookup_n_same_well'].mean() > 3, f"expected mostly same-well, got mean {result['lookup_n_same_well'].mean()}"


def test_lookup_excludes_same_row():
    """Same-row が kNN result に含まれないことを verify."""
    train_wells = [make_well_df('w0', seed=0)]
    full = pd.concat(train_wells, ignore_index=True)
    index = build_well_signature_index(full)
    query_rows = make_well_df('w0', seed=0).iloc[100:101].copy()
    result = lookup_same_well_tvt(
        query_rows,
        index,
        same_well_allowed=True,
        k=3,
        exclude_same_row=True,
    )
    # lookup_tvt_median should not be exactly the query row TVT (= same row excluded)
    query_tvt = query_rows['TVT'].iloc[0]
    median_lookup = result['lookup_tvt_median'].iloc[0]
    # Allow tiny floating diff but should not be byte-identical
    assert abs(median_lookup - query_tvt) > 1e-6, "same-row not excluded"


def test_lookup_same_well_disabled_uses_other_wells():
    """same_well_allowed=False で 他 well のみ lookup (= leak guard 強化、 標準 paradigm 6 と同等)."""
    train_wells = [
        make_well_df('w0', seed=0, x_offset=0.0, gr_phase=0.0),
        make_well_df('w1', seed=1, x_offset=10000.0, gr_phase=2.0),
        make_well_df('w2', seed=2, x_offset=20000.0, gr_phase=4.0),
    ]
    full = pd.concat(train_wells, ignore_index=True)
    index = build_well_signature_index(full)
    query_rows = make_well_df('w0', seed=0, x_offset=0.0, gr_phase=0.0).iloc[100:105].copy()
    result = lookup_same_well_tvt(
        query_rows,
        index,
        same_well_allowed=False,
        k=5,
    )
    assert result['lookup_n_same_well'].max() == 0, "same-well not excluded"
