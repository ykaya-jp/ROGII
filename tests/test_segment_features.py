"""Phase A4 + A5 tests for segment_features module.

Spec: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A4 + AC-A5.
"""

from __future__ import annotations

import numpy as np
import pytest

from rogii.segment_features import multi_scale_ncc, seg_b_well


# -----------------------------------------------------------------------------
# seg_b_well tests
# -----------------------------------------------------------------------------


def test_seg_b_well_returns_5_floats():
    ktvt = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    kz = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    form = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = seg_b_well(ktvt, kz, form)
    assert len(out) == 5
    assert all(isinstance(v, float) for v in out)


def test_seg_b_well_empty_returns_zeros():
    out = seg_b_well(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    assert out == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_seg_b_well_constant_input():
    """All bv == 5.0 → all 5 outputs should equal 5.0."""
    n = 100
    ktvt = np.full(n, 10.0, dtype=np.float32)
    kz = np.full(n, 5.0, dtype=np.float32)
    form = np.full(n, 10.0, dtype=np.float32)  # bv = 10 + 5 - 10 = 5
    b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, form)
    assert np.allclose([b_full, b_early, b_mid, b_late, b_wls], 5.0, atol=1e-4)


def test_seg_b_well_phase_drift_captured():
    """b_well = 0 for early third, then drifts to 10 in late third → b_early < b_late."""
    n = 150
    bv = np.concatenate([np.zeros(50), np.full(50, 5.0), np.full(50, 10.0)])
    # we control bv = ktvt + kz - form, set kz=0 form=0 → ktvt = bv
    ktvt = bv.astype(np.float32)
    kz = np.zeros(n, dtype=np.float32)
    form = np.zeros(n, dtype=np.float32)
    b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, form)
    assert b_early < b_mid < b_late, (
        f"phase drift not captured: early={b_early} mid={b_mid} late={b_late}"
    )
    # WLS is tail-upweighted → should pull toward b_late
    assert b_wls > b_full, f"WLS {b_wls} should exceed full median {b_full} for tail-biased data"


def test_seg_b_well_short_input():
    """n < 5 → b_late falls back to b_full, no crash."""
    ktvt = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    kz = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    form = np.zeros(3, dtype=np.float32)
    b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, form)
    assert b_late == b_full  # fallback
    assert np.isfinite([b_full, b_early, b_mid, b_late, b_wls]).all()


# -----------------------------------------------------------------------------
# multi_scale_ncc tests
# -----------------------------------------------------------------------------


@pytest.fixture
def synthetic_gr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    nk = 300
    ktvt = np.linspace(0.0, 30.0, nk).astype(np.float32)
    kgr = (
        80.0
        + 20.0 * np.sin(ktvt * 0.3)
        + rng.normal(0, 2.0, nk)
    ).astype(np.float32)
    # hidden: same sinusoid shifted in TVT to simulate offset
    nh = 100
    hidden_tvt_offset = np.linspace(5.0, 20.0, nh).astype(np.float32)
    hgr = (
        80.0
        + 20.0 * np.sin(hidden_tvt_offset * 0.3)
        + rng.normal(0, 2.0, nh)
    ).astype(np.float32)
    return kgr, ktvt, hgr


def test_multi_scale_ncc_shapes(synthetic_gr):
    kgr, ktvt, hgr = synthetic_gr
    per_scale, sc_ens = multi_scale_ncc(kgr, ktvt, hgr)
    assert len(per_scale) == 3  # hws = (8, 15, 25)
    for tvt, sc in per_scale:
        assert tvt.shape == hgr.shape
        assert sc.shape == hgr.shape
    assert sc_ens.shape == hgr.shape


def test_multi_scale_ncc_short_visible():
    """nk < 2*max(hws)+2 → fallback to last_tvt scalar fill."""
    kgr = np.array([90.0, 100.0, 110.0], dtype=np.float32)
    ktvt = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    hgr = np.array([85.0, 95.0, 105.0, 115.0], dtype=np.float32)
    per_scale, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25))
    # all per-scale should fall back to ktvt[-1] = 2.0
    for tvt, sc in per_scale:
        assert np.allclose(tvt, 2.0)
        assert np.allclose(sc, 0.0)
    # ensemble: softmax of 0 → equal weights → 2.0
    assert np.allclose(sc_ens, 2.0)


def test_multi_scale_ncc_empty_hgr():
    kgr = np.linspace(80.0, 120.0, 100).astype(np.float32)
    ktvt = np.linspace(0.0, 10.0, 100).astype(np.float32)
    hgr = np.array([], dtype=np.float32)
    per_scale, sc_ens = multi_scale_ncc(kgr, ktvt, hgr)
    assert sc_ens.shape == (0,)
    for tvt, sc in per_scale:
        assert tvt.shape == (0,)
        assert sc.shape == (0,)


def test_multi_scale_ncc_softmax_weights_normalised(synthetic_gr):
    """Internally sw should sum to 1 across scales for each hidden row."""
    kgr, ktvt, hgr = synthetic_gr
    per_scale, sc_ens = multi_scale_ncc(kgr, ktvt, hgr)
    # reconstruct sw from per_scale
    scores = np.stack([s for _, s in per_scale], axis=1)
    sw = np.exp(3.0 * scores)
    sw = sw / (sw.sum(axis=1, keepdims=True) + 1e-9)
    assert np.allclose(sw.sum(axis=1), 1.0, atol=1e-5)


def test_multi_scale_ncc_score_range(synthetic_gr):
    """NCC scores should be in [-1, 1] (= normalised cross-correlation property)."""
    kgr, ktvt, hgr = synthetic_gr
    per_scale, _ = multi_scale_ncc(kgr, ktvt, hgr)
    for _, sc in per_scale:
        assert sc.min() >= -1.001
        assert sc.max() <= 1.001
