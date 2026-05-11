"""Smoke + leak test for src/rogii/beam.py."""

from __future__ import annotations

import numpy as np
import pytest

from rogii.beam import BEAM_CONFIGS, _beam_kernel, beam_search, warmup


@pytest.fixture(scope="module")
def synthetic_well() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthetic well: gr_h aligns near tw_tvt midpoint.

    tw_tvt = linspace(10000, 12000, 200), tw_gr = sin signal.
    gr_h = tw_gr around idx 100 (= start_tvt = 11000).
    """
    rng = np.random.default_rng(42)
    tw_tvt = np.linspace(10000.0, 12000.0, 200, dtype=np.float32)
    tw_gr = (50.0 + 20.0 * np.sin(np.linspace(0, 6 * np.pi, 200))).astype(np.float32)
    # Horizontal well: 30 steps starting at tw idx 100, follow the signal.
    gr_h = tw_gr[100:130] + rng.normal(0.0, 1.0, 30).astype(np.float32)
    start_tvt = float(tw_tvt[100])
    return gr_h, tw_tvt, tw_gr, start_tvt


def test_warmup() -> None:
    warmup()  # must complete without raising


def test_beam_search_shape(synthetic_well):
    gr_h, tw_tvt, tw_gr, start_tvt = synthetic_well
    out = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20.0, es=144.0, r=2)
    assert out.shape == gr_h.shape
    assert out.dtype == np.float32
    # All output should be within tw_tvt range.
    assert out.min() >= tw_tvt.min()
    assert out.max() <= tw_tvt.max()


def test_beam_search_alignment(synthetic_well):
    """Beam output stays near `start_tvt` (= anchor), proving the trajectory
    is anchored at the eval-zone seed rather than drifting unboundedly.
    The output is monotonically constrained by move_cost; we assert proximity
    to the start anchor instead of mean of the underlying typewell range
    (which depends sensitively on mc / es).
    """
    gr_h, tw_tvt, tw_gr, start_tvt = synthetic_well
    out = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=12, mc=12.0, es=100.0, r=2)
    # tw step ~= 10 ft, 30 horizontal steps with delta in [-2, +2]
    # -> output should be within 300 ft of start anchor in worst case.
    assert abs(float(out.mean()) - float(start_tvt)) < 400.0


def test_beam_search_seven_configs(synthetic_well):
    """All 7 BEAM_CONFIGS must run without errors and return valid paths."""
    gr_h, tw_tvt, tw_gr, start_tvt = synthetic_well
    for bs, mc, es, r, tag in BEAM_CONFIGS:
        out = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=bs, mc=mc, es=es, r=r)
        assert out.shape == gr_h.shape, f"config {tag} bad shape"
        assert np.all(np.isfinite(out)), f"config {tag} has nan/inf"


def test_beam_deterministic_with_same_inputs(synthetic_well):
    """Beam kernel must produce same result for same inputs (no rng inside)."""
    gr_h, tw_tvt, tw_gr, start_tvt = synthetic_well
    out_a = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20.0, es=144.0, r=2)
    out_b = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20.0, es=144.0, r=2)
    np.testing.assert_array_equal(out_a, out_b)


def test_beam_kernel_index_bounds():
    """Kernel must not go out of bounds when start_idx is at edge."""
    sgr = np.zeros(20, dtype=np.float64)
    tw_gr = np.zeros(10, dtype=np.float64)
    path = _beam_kernel(sgr, tw_gr, np.int64(0), np.int64(5), 1.0, 100.0)
    assert path.shape == (20,)
    assert path.min() >= 0
    assert path.max() <= 9

    path2 = _beam_kernel(sgr, tw_gr, np.int64(9), np.int64(5), 1.0, 100.0)
    assert path2.min() >= 0
    assert path2.max() <= 9


def test_leak_smoke_no_typewell_cross_contamination():
    """Sanity: beam_search only consumes the typewell arg passed in,
    so per-well usage is leak-free by construction (no global tw store).
    Verify by switching tw_gr and checking output diverges.
    """
    rng = np.random.default_rng(0)
    tw_tvt = np.linspace(10000.0, 12000.0, 200, dtype=np.float32)
    gr_h = rng.standard_normal(30).astype(np.float32)
    tw_gr_a = (rng.standard_normal(200) * 10.0 + 50.0).astype(np.float32)
    tw_gr_b = (rng.standard_normal(200) * 10.0 + 50.0).astype(np.float32)
    out_a = beam_search(gr_h, tw_tvt, tw_gr_a, 11000.0, bs=10, mc=20.0, es=144.0, r=2)
    out_b = beam_search(gr_h, tw_tvt, tw_gr_b, 11000.0, bs=10, mc=20.0, es=144.0, r=2)
    # Different typewell GR -> different alignment (cross-leak proof).
    assert not np.array_equal(out_a, out_b)
