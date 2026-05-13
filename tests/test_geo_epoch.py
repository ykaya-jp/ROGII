"""Phase Magic-1 / Track 3.3: M3 Geological epoch cluster tests.

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §I M3.
Goal: wells の (X, Y) + GR signature で hierarchical clustering、 同 cluster wells
の TVT pattern を transfer (= per-cluster TVT-vs-MD median 等を feature 化)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.geo_epoch import (
    extract_well_signatures,
    cluster_wells_by_geo_signature,
    compute_per_cluster_tvt_pattern,
)


def _make_well(well_id, x_center, y_center, gr_amp, slope, n=200, seed=42):
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float)
    tvt = 1000 + slope * md + rng.normal(0, 0.01, n)
    gr = 80 + gr_amp * np.sin(md / 10) + rng.normal(0, 2, n)
    return pd.DataFrame({
        'well': well_id,
        'row': np.arange(n),
        'MD': md,
        'GR': gr,
        'X': np.full(n, x_center),
        'Y': np.full(n, y_center),
        'Z': -md * 0.5,
        'TVT': tvt,
        'TVT_input': np.where(np.arange(n) < int(n * 0.3), tvt, np.nan),
    })


def test_extract_well_signatures_returns_per_well_features():
    """per-well で X/Y centroid + GR mean/std/fft 等 8-10 features を抽出."""
    wells = [
        _make_well('w0', 100, 200, 20, 0.05, seed=0),
        _make_well('w1', 105, 205, 22, 0.06, seed=1),
        _make_well('w2', 5000, 5000, 10, 0.01, seed=2),  # far cluster
    ]
    full = pd.concat(wells, ignore_index=True)
    sig = extract_well_signatures(full)
    assert isinstance(sig, pd.DataFrame)
    assert len(sig) == 3
    assert {'well', 'x_centroid', 'y_centroid', 'gr_mean', 'gr_std'}.issubset(sig.columns)


def test_cluster_wells_finds_2_clusters():
    """w0+w1 (= near) と w2 (= far) で 2 cluster になる."""
    wells = [
        _make_well('w0', 100, 200, 20, 0.05, seed=0),
        _make_well('w1', 105, 205, 22, 0.06, seed=1),
        _make_well('w2', 5000, 5000, 10, 0.01, seed=2),
    ]
    full = pd.concat(wells, ignore_index=True)
    sig = extract_well_signatures(full)
    clustered = cluster_wells_by_geo_signature(sig, n_clusters=2)
    assert 'cluster_id' in clustered.columns
    # w0+w1 と w2 は different cluster
    c0 = clustered[clustered['well'] == 'w0']['cluster_id'].iloc[0]
    c1 = clustered[clustered['well'] == 'w1']['cluster_id'].iloc[0]
    c2 = clustered[clustered['well'] == 'w2']['cluster_id'].iloc[0]
    assert c0 == c1, f'near wells different cluster: w0={c0}, w1={c1}'
    assert c0 != c2, f'far well same cluster as near: c0={c0}, c2={c2}'


def test_per_cluster_tvt_pattern_returns_lookup():
    """同 cluster の TVT median を per-MD で取得 (= LGB feature lookup table)."""
    wells = [
        _make_well('w0', 100, 200, 20, 0.05, seed=0),
        _make_well('w1', 105, 205, 22, 0.05, seed=1),  # similar slope
        _make_well('w2', 5000, 5000, 10, 0.01, seed=2),
    ]
    full = pd.concat(wells, ignore_index=True)
    sig = extract_well_signatures(full)
    clustered = cluster_wells_by_geo_signature(sig, n_clusters=2)
    pattern = compute_per_cluster_tvt_pattern(full, clustered, md_bin_size=10)
    assert {'cluster_id', 'md_bin', 'tvt_median', 'tvt_std', 'n'}.issubset(pattern.columns)
    # 2 clusters × 多 bins
    assert pattern['cluster_id'].nunique() == 2
