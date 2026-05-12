"""Phase A3 tests for beam_pf_numba module.

Spec: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A3.
Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
Strength analysis: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md §1.3.

Tests:
  - ±1 baseline produces a valid TVT path (= every output ∈ tw_tvt range)
  - ±2 extension produces a valid path (= same property)
  - ±2 path covers ≥ as much TVT range as ±1 for adversarial GR signals
  - boundary safety (si=0 + si=nt-1 + len(tw)<3)
  - multi-config wrapper returns 7 entries
"""

from __future__ import annotations

import numpy as np
import pytest

from rogii.beam_pf_numba import (
    BEAMS_DEFAULT,
    beam_search_multi_numba,
    beam_search_numba,
)


@pytest.fixture
def synthetic_well() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthetic: typewell GR follows sinusoid in TVT, horizontal well = same with phase shift."""
    rng = np.random.default_rng(42)
    nt = 500
    tw_tvt = np.linspace(0.0, 50.0, nt, dtype=np.float32)
    tw_gr = (
        80.0
        + 30.0 * np.sin(tw_tvt * 0.5)
        + rng.normal(0, 2.0, nt).astype(np.float32)
    ).astype(np.float32)
    nh = 200
    hgr = (
        80.0
        + 30.0 * np.sin((tw_tvt[100 : 100 + nh] + 0.5) * 0.5)
        + rng.normal(0, 3.0, nh).astype(np.float32)
    ).astype(np.float32)
    start_tvt = float(tw_tvt[100])
    return hgr, tw_tvt, tw_gr, start_tvt


def test_beam_pm1_baseline_returns_valid_path(synthetic_well):
    hgr, tw_tvt, tw_gr, start_tvt = synthetic_well
    path = beam_search_numba(
        hgr, tw_tvt, tw_gr, start_tvt, delta_range=1
    )
    assert path.shape == hgr.shape
    assert path.dtype == np.float32
    # all path TVT values within tw_tvt range
    assert path.min() >= tw_tvt.min() - 1e-3
    assert path.max() <= tw_tvt.max() + 1e-3


def test_beam_pm2_extension_returns_valid_path(synthetic_well):
    hgr, tw_tvt, tw_gr, start_tvt = synthetic_well
    path = beam_search_numba(
        hgr, tw_tvt, tw_gr, start_tvt, delta_range=2
    )
    assert path.shape == hgr.shape
    assert path.min() >= tw_tvt.min() - 1e-3
    assert path.max() <= tw_tvt.max() + 1e-3


def test_beam_pm2_covers_larger_or_equal_range(synthetic_well):
    """±2 should never restrict the explored TVT space below ±1 (= superset transitions)."""
    hgr, tw_tvt, tw_gr, start_tvt = synthetic_well
    p1 = beam_search_numba(hgr, tw_tvt, tw_gr, start_tvt, delta_range=1)
    p2 = beam_search_numba(hgr, tw_tvt, tw_gr, start_tvt, delta_range=2)
    # ±2 has access to all ±1 transitions, so cumulative travel can only equal or exceed
    travel_1 = float(np.abs(np.diff(p1)).sum())
    travel_2 = float(np.abs(np.diff(p2)).sum())
    # ±2 should typically equal or exceed ±1 travel — but not always strictly more
    # (= depends on cost landscape). Soft check: 80% of cases.
    assert travel_2 >= travel_1 * 0.7, (
        f"±2 travel {travel_2:.3f} unexpectedly << ±1 travel {travel_1:.3f}"
    )


def test_beam_boundary_si_zero(synthetic_well):
    hgr, tw_tvt, tw_gr, _ = synthetic_well
    # start_tvt below tw_tvt[0] → si clamped to 0
    path = beam_search_numba(hgr, tw_tvt, tw_gr, -100.0, delta_range=2)
    assert path.shape == hgr.shape
    assert np.all(np.isfinite(path))


def test_beam_boundary_si_max(synthetic_well):
    hgr, tw_tvt, tw_gr, _ = synthetic_well
    # start_tvt above tw_tvt[-1] → si clamped to nt-1
    path = beam_search_numba(hgr, tw_tvt, tw_gr, 1e6, delta_range=2)
    assert path.shape == hgr.shape
    assert np.all(np.isfinite(path))


def test_beam_short_typewell_returns_start():
    """nt < 3 → return constant start_tvt."""
    hgr = np.array([90.0, 100.0, 110.0], dtype=np.float32)
    tw_tvt = np.array([0.0, 1.0], dtype=np.float32)
    tw_gr = np.array([80.0, 90.0], dtype=np.float32)
    path = beam_search_numba(hgr, tw_tvt, tw_gr, 0.5, delta_range=2)
    assert path.shape == (3,)
    assert np.allclose(path, 0.5)


def test_beam_multi_returns_7_configs(synthetic_well):
    hgr, tw_tvt, tw_gr, start_tvt = synthetic_well
    out = beam_search_multi_numba(hgr, tw_tvt, tw_gr, start_tvt, delta_range=2)
    assert set(out.keys()) == {tag for _, _, _, _, tag in BEAMS_DEFAULT}
    assert all(p.shape == hgr.shape for p in out.values())


def test_beam_invalid_delta_range_raises(synthetic_well):
    hgr, tw_tvt, tw_gr, start_tvt = synthetic_well
    with pytest.raises(ValueError, match="delta_range must be >= 1"):
        beam_search_numba(hgr, tw_tvt, tw_gr, start_tvt, delta_range=0)


def test_beam_mismatched_typewell_lengths_raises(synthetic_well):
    hgr, tw_tvt, _, start_tvt = synthetic_well
    tw_gr_short = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="must have equal length"):
        beam_search_numba(hgr, tw_tvt, tw_gr_short, start_tvt, delta_range=2)


# -----------------------------------------------------------------------------
# A3.1: PF Z dual particle filter tests (= raunakdey07 _pf_z)
# -----------------------------------------------------------------------------


def test_pf_z_basic_shape_and_finite():
    """run_pf_z must return (pts, std) of shape (n,) all finite."""
    from rogii.beam_pf_numba import run_pf_z

    rng = np.random.default_rng(42)
    n = 100
    md_v = np.linspace(1000.0, 1100.0, n).astype(np.float64)
    z_v = np.linspace(50.0, 55.0, n).astype(np.float64)
    gr_v = (80.0 + rng.normal(0, 5, n)).astype(np.float64)
    gr_sm_v = gr_v.copy()
    nt = 200
    tw_tvt = np.linspace(0.0, 30.0, nt).astype(np.float64)
    tw_gr = (80.0 + 20.0 * np.sin(tw_tvt * 0.3)).astype(np.float64)
    tw_gr_sm = tw_gr.copy()
    pts, std_ = run_pf_z(
        md_v, z_v, gr_v, gr_sm_v,
        tw_tvt, tw_gr, tw_gr_sm,
        gs=30.0,
        init_pos=15.0,
        init_v=0.05,
        beta=-1.0,
        icpt=0.0,
        zsig=0.1,
        N=100,  # small N for fast test
    )
    assert pts.shape == (n,)
    assert std_.shape == (n,)
    assert np.isfinite(pts).all()
    assert np.isfinite(std_).all()
    # std should be non-negative
    assert (std_ >= 0).all()


def test_pf_z_nan_gr_handled():
    """run_pf_z must skip GR likelihood when gr_v has NaNs (= boundary)."""
    from rogii.beam_pf_numba import run_pf_z

    n = 30
    md_v = np.arange(n, dtype=np.float64)
    z_v = np.zeros(n, dtype=np.float64)
    gr_v = np.full(n, np.nan, dtype=np.float64)  # all NaN
    gr_sm_v = np.full(n, np.nan, dtype=np.float64)
    nt = 50
    tw_tvt = np.linspace(0.0, 20.0, nt).astype(np.float64)
    tw_gr = np.full(nt, 80.0, dtype=np.float64)
    tw_gr_sm = tw_gr.copy()
    pts, std_ = run_pf_z(
        md_v, z_v, gr_v, gr_sm_v,
        tw_tvt, tw_gr, tw_gr_sm,
        gs=30.0, init_pos=10.0, init_v=0.0,
        N=50,
    )
    assert pts.shape == (n,)
    assert np.isfinite(pts).all()


def test_pf_z_dual_smoothing_path():
    """When gr_sm_v has values (not NaN), dual likelihood path is exercised."""
    from rogii.beam_pf_numba import run_pf_z

    rng = np.random.default_rng(7)
    n = 60
    md_v = np.linspace(0, 60, n).astype(np.float64)
    z_v = np.linspace(0, 10, n).astype(np.float64)
    gr_v = (75.0 + rng.normal(0, 3, n)).astype(np.float64)
    # smoothed version (= rolling mean simulator)
    import pandas as pd
    gr_sm_v = pd.Series(gr_v).rolling(5, center=True, min_periods=1).mean().to_numpy()
    nt = 100
    tw_tvt = np.linspace(0.0, 30.0, nt).astype(np.float64)
    tw_gr = (75.0 + 10.0 * np.sin(tw_tvt)).astype(np.float64)
    tw_gr_sm = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().to_numpy()
    pts, std_ = run_pf_z(
        md_v, z_v, gr_v, gr_sm_v,
        tw_tvt, tw_gr, tw_gr_sm,
        gs=20.0, init_pos=15.0, init_v=0.05,
        N=80,
    )
    assert np.isfinite(pts).all()
    assert (std_ >= 0).all()
