"""Phase A6: 3-axis (alpha × tau × w_pf) postproc + SG smoothing via Optuna 500-trial.

Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
Strength analysis: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md §1.2.
Criteria: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A6.

This module ports the 3-axis Optuna postproc from Hill Climb (LB 9.43) which
extends our existing 2-axis (alpha × tau) grid search with a w_pf blend axis
controlling the LGB-vs-PF ratio. Hill Climb claims +0.4 LB from this single
extension (= LGB+XGB notebook v4 → v5 internal breakdown).

License / attribution:
  - apply_pp + sg_smooth + Optuna objective: ported from
    https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
    (Roman Tarasov, Apache-2.0 fair-use)
  - Optuna: BSD-3-Clause
  - Savitzky-Golay: scipy.signal (BSD-3-Clause)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import root_mean_squared_error


def apply_pp(
    md_since: np.ndarray,
    model_delta: np.ndarray,
    pf_delta: np.ndarray,
    alpha: float,
    tau: float | int,
    w_pf: float,
) -> np.ndarray:
    """Apply postproc to combine model delta and PF delta.

    Formula:
        d = model_delta * (1 - w_pf) + pf_delta * w_pf
        if tau > 0:
            d *= 1 - exp(-md_since / tau)  # fade-in from anchor
        return d * alpha

    Args:
        md_since: rows distance from last_known anchor (= MD - last_known_MD), shape (n,)
        model_delta: residual model prediction (= hc_oof or test_preds), shape (n,)
        pf_delta: PF prediction residual (= pf_ancc - last_known_tvt), shape (n,)
        alpha: scalar gain ∈ [0.5, 1.0]
        tau: fade-in time constant in MD units; 0 = no fade-in
        w_pf: PF blend weight ∈ [0, 0.5]

    Returns:
        d: postproc'd residual prediction, shape (n,)
    """
    md_since = np.maximum(md_since.astype(np.float32), 0.0)
    d = model_delta.astype(np.float32) * (1.0 - w_pf) + pf_delta.astype(np.float32) * w_pf
    if tau:
        fade = 1.0 - np.exp(-md_since / float(tau))
        d = d * fade.astype(np.float32)
    return d * np.float32(alpha)


def sg_smooth_inplace(
    df: pd.DataFrame,
    col: str,
    well_col: str = "well",
    sg_w: int = 17,
    sg_p: int = 3,
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing per-well in place. Returns the (mutated) df."""
    df = df.copy()
    for _, g in df.groupby(well_col, sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df


def optimize_postproc(
    md_since: np.ndarray,
    model_delta: np.ndarray,
    pf_delta: np.ndarray,
    y_true_delta: np.ndarray,
    n_trials: int = 500,
    n_startup_trials: int = 50,
    n_jobs: int = -1,
    seed: int = 42,
    alpha_range: tuple[float, float, float] = (0.5, 1.0, 0.01),
    tau_range: tuple[int, int, int] = (5, 500, 5),
    w_pf_range: tuple[float, float, float] = (0.0, 0.5, 0.01),
    verbose: bool = False,
) -> dict[str, Any]:
    """Run Optuna TPE search over (alpha, tau, w_pf) to minimize RMSE on OOF.

    Args:
        md_since: rows distance from anchor (used inside apply_pp)
        model_delta: OOF residual prediction
        pf_delta: PF residual prediction
        y_true_delta: ground-truth residual (= y - last_known_tvt)
        n_trials: total trials (Hill Climb uses 500)
        n_startup_trials: random startup before TPE kicks in
        n_jobs: -1 = all CPU; 1 = single thread (= reproducible)
        seed: TPESampler seed
        alpha_range / tau_range / w_pf_range: (low, high, step)
        verbose: print study trial progress

    Returns:
        {
            'best_params': {'alpha': ..., 'tau': ..., 'w_pf': ...},
            'best_score': RMSE on OOF,
            'study': optuna.Study,
        }
    """

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], step=alpha_range[2])
        tau = trial.suggest_int("tau", tau_range[0], tau_range[1], step=tau_range[2])
        w_pf = trial.suggest_float("w_pf", w_pf_range[0], w_pf_range[1], step=w_pf_range[2])
        d = apply_pp(md_since, model_delta, pf_delta, alpha, tau, w_pf)
        return float(root_mean_squared_error(y_true_delta, d))

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

    return {
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "study": study,
    }
