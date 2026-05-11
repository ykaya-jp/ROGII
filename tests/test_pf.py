"""Smoke + numerical-stability test for src/rogii/pf.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.pf import PFConfig, run_pf_ancc, run_pf_z


@pytest.fixture(scope="module")
def synthetic_well() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic 100-row well with 70 known + 30 eval rows."""
    rng = np.random.default_rng(7)
    n = 100
    md = np.arange(n, dtype=np.float64) * 1.0 + 5000.0
    z = (np.linspace(0, 100, n) + rng.normal(0, 0.5, n)).astype(np.float64)
    tvt_full = (np.linspace(11000, 11300, n)).astype(np.float64)
    gr = (50.0 + 20.0 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 2.0, n)).astype(
        np.float64
    )
    tvt_input = tvt_full.copy()
    tvt_input[70:] = np.nan  # hidden zone
    hw = pd.DataFrame(
        {
            "MD": md,
            "X": np.linspace(0, 1, n),
            "Y": np.linspace(0, 0.5, n),
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt_full,
        }
    )
    tw_tvt = np.linspace(10800.0, 11400.0, 80).astype(np.float32)
    tw_gr = (50.0 + 20.0 * np.sin(np.linspace(0, 4 * np.pi, 80))).astype(np.float32)
    return hw, tw_tvt, tw_gr


def test_run_pf_ancc_shape(synthetic_well):
    hw, tw_tvt, tw_gr = synthetic_well
    pts, std = run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42)
    assert pts.shape == (30,)
    assert std.shape == (30,)
    assert pts.dtype == np.float32
    assert std.dtype == np.float32


def test_run_pf_z_shape(synthetic_well):
    hw, tw_tvt, tw_gr = synthetic_well
    pts, std = run_pf_z(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42)
    assert pts.shape == (30,)
    assert std.shape == (30,)


def test_pf_numerical_floor_no_nan(synthetic_well):
    """Even with extreme GR mismatch, PF must not produce NaN
    (= particle collapse prevention)."""
    hw, tw_tvt, tw_gr = synthetic_well
    # Inject extreme GR values to push likelihoods toward zero.
    hw2 = hw.copy()
    hw2.loc[80:, "GR"] = 1e10  # absurd values
    pts, std = run_pf_ancc(hw2, tw_tvt, tw_gr, n_particles=100, rng_seed=42)
    assert np.all(np.isfinite(pts))
    assert np.all(np.isfinite(std))


def test_pf_position_clip(synthetic_well):
    """Particle filter must keep position within tmin-50, tmax+50."""
    hw, tw_tvt, tw_gr = synthetic_well
    pts, _ = run_pf_z(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42)
    assert pts.min() >= tw_tvt.min() - 50.0
    assert pts.max() <= tw_tvt.max() + 50.0


def test_pf_deterministic_with_same_seed(synthetic_well):
    """Same seed -> same output."""
    hw, tw_tvt, tw_gr = synthetic_well
    pts_a, _ = run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42)
    pts_b, _ = run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42)
    np.testing.assert_array_equal(pts_a, pts_b)

    pts_c, _ = run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=200, rng_seed=99)
    assert not np.array_equal(pts_a, pts_c)


def test_pf_no_eval_rows():
    """Empty eval zone -> empty output, no error."""
    hw = pd.DataFrame(
        {
            "MD": np.arange(50, dtype=np.float64),
            "X": np.zeros(50),
            "Y": np.zeros(50),
            "Z": np.linspace(0, 10, 50),
            "GR": np.full(50, 50.0),
            "TVT_input": np.linspace(11000, 11100, 50),
            "TVT": np.linspace(11000, 11100, 50),
        }
    )
    tw_tvt = np.linspace(10800.0, 11400.0, 80).astype(np.float32)
    tw_gr = np.full(80, 50.0, dtype=np.float32)
    pts, std = run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=100, rng_seed=42)
    assert len(pts) == 0
    assert len(std) == 0


def test_pf_config_dataclass_immutable():
    """PFConfig is frozen (= can't accidentally mutate global defaults)."""
    cfg = PFConfig()
    with pytest.raises(Exception):
        cfg.n_particles = 999  # type: ignore[misc]


def test_pf_self_wid_placeholder_does_not_break(synthetic_well):
    """`self_wid` kwarg is accepted (placeholder for future leak control)."""
    hw, tw_tvt, tw_gr = synthetic_well
    pts, _ = run_pf_ancc(
        hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42, self_wid="wA"
    )
    pts2, _ = run_pf_ancc(
        hw, tw_tvt, tw_gr, n_particles=200, rng_seed=42, self_wid=None
    )
    # Currently identical (PF is per-well, no cross-well leak surface).
    np.testing.assert_array_equal(pts, pts2)
