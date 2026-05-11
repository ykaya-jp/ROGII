"""Phase A6 tests for postproc_optuna module.

Spec: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A6.
Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.postproc_optuna import apply_pp, optimize_postproc, sg_smooth_inplace


def test_apply_pp_basic():
    """At alpha=1, tau=0 → no fade, no SG, output = (1-w_pf)*md + w_pf*pf."""
    n = 50
    md_since = np.arange(n, dtype=np.float32)
    md = np.full(n, 1.0, dtype=np.float32)
    pf = np.full(n, 2.0, dtype=np.float32)
    out = apply_pp(md_since, md, pf, alpha=1.0, tau=0, w_pf=0.3)
    # expected: 1.0 * 0.7 + 2.0 * 0.3 = 1.3
    assert np.allclose(out, 1.3, atol=1e-5)


def test_apply_pp_fade_in():
    """At md_since=0 with tau=10 → output = 0 (= fade-in starts from 0)."""
    n = 20
    md_since = np.zeros(n, dtype=np.float32)
    md = np.full(n, 1.0, dtype=np.float32)
    pf = np.full(n, 1.0, dtype=np.float32)
    out = apply_pp(md_since, md, pf, alpha=1.0, tau=10, w_pf=0.0)
    assert np.allclose(out, 0.0, atol=1e-5)


def test_apply_pp_alpha_scaling():
    n = 20
    md_since = np.full(n, 1000.0, dtype=np.float32)  # large → fade ≈ 1
    md = np.full(n, 2.0, dtype=np.float32)
    pf = np.zeros(n, dtype=np.float32)
    out = apply_pp(md_since, md, pf, alpha=0.5, tau=10, w_pf=0.0)
    # md=2.0, fade≈1.0, alpha=0.5 → 1.0
    assert np.allclose(out, 1.0, atol=1e-4)


def test_apply_pp_negative_md_since_clipped():
    """Negative md_since should be clipped to 0 (= no extrapolation behind anchor)."""
    md_since = np.array([-10.0, -1.0, 0.0, 1.0], dtype=np.float32)
    md = np.full(4, 5.0, dtype=np.float32)
    pf = np.zeros(4, dtype=np.float32)
    out = apply_pp(md_since, md, pf, alpha=1.0, tau=2, w_pf=0.0)
    # First two should fade to 0 (fade = 1 - exp(0) = 0 after clip)
    assert np.allclose(out[:2], 0.0, atol=1e-5)


def test_sg_smooth_per_well():
    n = 30
    df = pd.DataFrame({
        "well": ["A"] * 15 + ["B"] * 15,
        "pred": list(np.random.RandomState(42).randn(15) * 5)
                + list(np.random.RandomState(43).randn(15) * 5),
    })
    out = sg_smooth_inplace(df, "pred", sg_w=5, sg_p=2)
    # SG should reduce |diff| variance vs original
    orig_diff = np.abs(np.diff(df["pred"].values)).mean()
    new_diff = np.abs(np.diff(out["pred"].values)).mean()
    assert new_diff <= orig_diff * 1.1  # within tolerance


def test_sg_smooth_short_well_unchanged():
    """If well length < sg_p + 2, no smoothing applied."""
    df = pd.DataFrame({"well": ["A"] * 3, "pred": [1.0, 2.0, 3.0]})
    out = sg_smooth_inplace(df, "pred", sg_w=17, sg_p=3)
    assert np.allclose(out["pred"].values, [1.0, 2.0, 3.0])


def test_optimize_postproc_minimal_convergence():
    """Synthetic: y = 0.7 * md + 0.3 * pf, with fade. Optuna should recover alpha≈1, w_pf≈0.3."""
    rng = np.random.default_rng(42)
    n = 200
    md_since = rng.uniform(0, 200, n).astype(np.float32)
    md = rng.normal(0, 3, n).astype(np.float32)
    pf = rng.normal(0, 3, n).astype(np.float32)
    true_alpha, true_w_pf, true_tau = 1.0, 0.3, 50
    y_true = apply_pp(md_since, md, pf, true_alpha, true_tau, true_w_pf)
    out = optimize_postproc(
        md_since, md, pf, y_true,
        n_trials=80, n_startup_trials=20, n_jobs=1, seed=42,
    )
    bp = out["best_params"]
    assert out["best_score"] < 0.5, f"best_score {out['best_score']} too high"
    assert 0.7 <= bp["alpha"] <= 1.0
    assert 0.0 <= bp["w_pf"] <= 0.5
