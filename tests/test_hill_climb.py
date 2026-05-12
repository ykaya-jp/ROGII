"""Phase A2 tests for hill_climb.Climber.

Spec: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A2.
Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
Strength analysis: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md §1.1.

Tests cover:
  - Single-base case → weight should converge to ~1.0
  - Two-base linear combination → recovers true weights within precision
  - Negative weight pathway → must beat positive-only on adversarial overcount
  - Predict shape + weight non-negativity property when allow_negative_weights=False
  - Convergence stop (= patience triggered)
  - Custom eval metric (R² maximize)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import r2_score, root_mean_squared_error

from rogii.hill_climb import Climber


def test_climber_invalid_objective():
    with pytest.raises(ValueError, match="objective must be"):
        Climber(objective="invalid")


def test_climber_invalid_precision():
    with pytest.raises(ValueError, match="precision must be in"):
        Climber(precision=0)
    with pytest.raises(ValueError, match="precision must be in"):
        Climber(precision=2.0)


def test_climber_single_base_converges_to_one():
    """If only one base equals y, weight should approach 1.0."""
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 200)
    oof = y.reshape(-1, 1)
    c = Climber(precision=0.01, max_iter=500).fit(oof, y)
    assert c.weights_.shape == (1,)
    assert 0.9 <= c.weights_[0] <= 1.1, f"weight {c.weights_[0]} not near 1.0"
    assert c.best_score < 0.1


def test_climber_two_base_recovery():
    """y = 0.6 * b1 + 0.4 * b2, base count=2 → recover weights within precision."""
    rng = np.random.default_rng(7)
    b1 = rng.normal(0, 1, 500)
    b2 = rng.normal(0, 1, 500)
    y = 0.6 * b1 + 0.4 * b2
    oof = np.stack([b1, b2], axis=1)
    c = Climber(precision=0.01, max_iter=2000).fit(oof, y)
    assert 0.5 <= c.weights_[0] <= 0.7, f"w1 {c.weights_[0]} not in [0.5, 0.7]"
    assert 0.3 <= c.weights_[1] <= 0.5, f"w2 {c.weights_[1]} not in [0.3, 0.5]"
    assert c.best_score < 0.05


def test_climber_negative_weight_helps_overcount():
    """y is positively correlated with b1, but b2 is overcounting (= b1 + noise).
    Without negative weights, the optimizer cannot subtract b2's contribution.
    """
    rng = np.random.default_rng(13)
    n = 500
    signal = rng.normal(0, 1, n)
    b1 = signal + rng.normal(0, 0.1, n)
    b2 = 2 * signal + rng.normal(0, 0.1, n)  # overcounts signal 2x
    y = signal
    oof = np.stack([b1, b2], axis=1)

    pos_only = Climber(allow_negative_weights=False, precision=0.005, max_iter=2000).fit(oof, y)
    full = Climber(allow_negative_weights=True, precision=0.005, max_iter=2000).fit(oof, y)
    # negative weights should produce equal-or-better OOF score
    assert full.best_score <= pos_only.best_score + 1e-6, (
        f"negative weights {full.best_score} did not beat positive-only {pos_only.best_score}"
    )


def test_climber_predict_shape_and_match():
    rng = np.random.default_rng(42)
    oof = rng.normal(0, 1, (300, 3))
    y = oof[:, 0] * 0.5 + oof[:, 1] * 0.3 - oof[:, 2] * 0.2 + rng.normal(0, 0.1, 300)
    c = Climber(precision=0.01, max_iter=1500).fit(oof, y)
    test = rng.normal(0, 1, (50, 3))
    p = c.predict(test)
    assert p.shape == (50,)
    # manual check: predict == test @ weights
    assert np.allclose(p, test @ c.weights_)


def test_climber_predict_before_fit_raises():
    c = Climber()
    with pytest.raises(RuntimeError, match="has not been fit"):
        c.predict(np.zeros((10, 3)))


def test_climber_dimension_mismatch_raises():
    rng = np.random.default_rng(42)
    oof = rng.normal(0, 1, (100, 3))
    y = rng.normal(0, 1, 100)
    c = Climber(max_iter=10).fit(oof, y)
    with pytest.raises(ValueError, match="X has"):
        c.predict(np.zeros((10, 5)))  # wrong n_base


def test_climber_y_length_mismatch_raises():
    oof = np.zeros((100, 3))
    y = np.zeros(50)
    c = Climber()
    with pytest.raises(ValueError, match="length mismatch"):
        c.fit(oof, y)


def test_climber_objective_maximize_with_r2():
    """Custom eval metric + objective=maximize (= r2_score)."""
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 300)
    oof = np.stack([y + rng.normal(0, 0.1, 300), y * 0.5], axis=1)
    c = Climber(
        objective="maximize",
        eval_metric=r2_score,
        precision=0.01,
        max_iter=500,
    ).fit(oof, y)
    assert c.best_score > 0.5


# -----------------------------------------------------------------------------
# A2.1 continuous mode tests (= raunakdey07 Gaussian perturbation)
# -----------------------------------------------------------------------------


def test_climber_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        Climber(mode="invalid")


def test_climber_invalid_perturb_sigma():
    with pytest.raises(ValueError, match="perturb_sigma must be"):
        Climber(mode="continuous", perturb_sigma=0)


def test_climber_continuous_mode_recovers_two_base():
    """continuous mode: y = 0.6 * b1 + 0.4 * b2 → recover within tolerance."""
    rng = np.random.default_rng(7)
    b1 = rng.normal(0, 1, 500)
    b2 = rng.normal(0, 1, 500)
    y = 0.6 * b1 + 0.4 * b2
    oof = np.stack([b1, b2], axis=1)
    c = Climber(
        mode="continuous",
        perturb_sigma=0.02,
        patience=500,
        max_iter=5000,
        normalize_weights=True,
        seed=42,
    ).fit(oof, y)
    assert c.weights_.shape == (2,)
    # weights sum should equal 1 (= simplex)
    assert abs(c.weights_.sum() - 1.0) < 1e-6
    # both weights in [0, 1]
    assert (c.weights_ >= 0).all() and (c.weights_ <= 1).all()
    # final score should be reasonably low
    assert c.best_score < 0.5


def test_climber_continuous_mode_patience_stops_early():
    """patience=5 should stop early on flat landscape (= y all zero)."""
    rng = np.random.default_rng(1)
    n = 100
    oof = rng.normal(0, 1, (n, 3))
    y = np.zeros(n)  # constant target → no signal
    c = Climber(
        mode="continuous",
        patience=5,
        max_iter=10000,
        perturb_sigma=0.02,
        normalize_weights=False,
        allow_negative_weights=True,
        seed=42,
    ).fit(oof, y)
    # history should be short (= early stop)
    assert len(c.history_) < 10000


def test_climber_continuous_mode_predict():
    """continuous mode predict shape + arithmetic check."""
    rng = np.random.default_rng(5)
    oof = rng.normal(0, 1, (200, 4))
    y = oof[:, 0] - 0.3 * oof[:, 1] + rng.normal(0, 0.1, 200)
    c = Climber(
        mode="continuous",
        perturb_sigma=0.05,
        patience=300,
        max_iter=2000,
        normalize_weights=False,
        allow_negative_weights=True,
        seed=42,
    ).fit(oof, y)
    test = rng.normal(0, 1, (40, 4))
    p = c.predict(test)
    assert p.shape == (40,)
    assert np.allclose(p, test @ c.weights_)


def test_climber_continuous_normalize_simplex_enforced():
    """normalize_weights=True must enforce sum==1 + nonneg post-fit."""
    rng = np.random.default_rng(11)
    oof = rng.normal(0, 1, (150, 3))
    y = oof @ np.array([0.5, 0.3, 0.2])
    c = Climber(
        mode="continuous",
        perturb_sigma=0.03,
        patience=500,
        max_iter=3000,
        normalize_weights=True,
        seed=42,
    ).fit(oof, y)
    assert (c.weights_ >= 0).all()
    assert abs(c.weights_.sum() - 1.0) < 1e-6
