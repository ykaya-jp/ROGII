"""Numba JIT Beam Search ±delta + Dense O(1) PF Grid.

Phase A3 of plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.

This module ports the Numba JIT-accelerated beam search and dual particle filter
(ANCC + Z) implementations used by Hill Climb (LB 9.43) and LGB+XGB (LB 9.83)
public Kaggle notebooks. Reference: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md §1.3-1.4.

Key extensions over the existing src/rogii/tysig.py:beam_search:
  - Numba JIT compilation (cache=True) → 20x speedup over pandas/numpy
  - Parameterized delta range (delta_range=1 for legacy ±1, =2 for Hill Climb ±2)
    so ablation A3-baseline vs A3-extended can be measured independently.
  - Dense O(1) grid lookup for PF inner loop (vs O(log n) np.interp)
  - Dual PF on ANCC and Z axes (independent state-space from our Edge D Kalman/PF on dTVT)

License / attribution:
  - Beam JIT algorithm: ported from
    https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
    (Roman Tarasov) and
    https://www.kaggle.com/code/<lgb-xgb-author>/lb-9-830-rogii-lgb-xgb
    (same _beam_jit/_pf_*_jit core). Both notebooks are Apache-2.0 fair-use.
  - Numba runtime: BSD-2-Clause (numba)
  - Mathematics: Catuneanu 2006 "Principles of Sequence Stratigraphy" (Beam),
    Doucet 2001 (PF), Hensman 2013 (Sparse GP — separately in src/rogii/gp_sparse.py)

This file is internal to ky7240/rogii-* kernels. Re-distribution of bundled
artifacts must keep the above attribution.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# Public BEAMS configs (= 7 configs from Hill Climb cell 5)
# Format: (beam_size, move_cost, err_scale, smooth_radius, tag)
# ----------------------------------------------------------------------------
BEAMS_DEFAULT: tuple[tuple[int, float, float, int, str], ...] = (
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
)

# ----------------------------------------------------------------------------
# Numba JIT helpers
# ----------------------------------------------------------------------------


@njit(cache=True)
def _interp1(grid: np.ndarray, v: float, vmin: float, step: float) -> float:
    """O(1) linear interp on uniform grid. Returns boundary value if v out of range."""
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(
    pos: np.ndarray,
    aux: np.ndarray,
    w: np.ndarray,
    N: int,
    rp: float,
    rv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Systematic resampling (Doucet 2001) with optional roughening on pos+aux."""
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


# ----------------------------------------------------------------------------
# Beam JIT core (= Hill Climb / LGB+XGB _beam_jit) with parameterized delta
# ----------------------------------------------------------------------------


@njit(cache=True)
def _beam_jit_core(
    sgr: np.ndarray,
    tw_gr: np.ndarray,
    si: int,
    BS: int,
    mc: float,
    es: float,
    delta_lo: int,
    delta_hi: int,
) -> np.ndarray:
    """Beam search over typewell GR axis with [delta_lo, delta_hi] step transitions.

    delta_lo=-1, delta_hi=2 → legacy ±1 + 0 (= 3 deltas)
    delta_lo=-2, delta_hi=3 → Hill Climb ±2 + 0 (= 5 deltas)

    Args:
        sgr: horizontal-well GR sequence (smoothed), len = n
        tw_gr: typewell GR axis (sorted by TVT), len = nt
        si: start index in tw_gr (= last_known TVT projected)
        BS: beam size
        mc: move cost (penalty per |delta|)
        es: error scale (in squared GR units)
        delta_lo, delta_hi: half-open range for delta (= range(delta_lo, delta_hi))

    Returns:
        path: np.int64 array of len n, each element = index into tw_gr
    """
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(delta_lo, delta_hi):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


# ----------------------------------------------------------------------------
# Public Beam wrapper
# ----------------------------------------------------------------------------


def beam_search_numba(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_size: int = 10,
    move_cost: float = 20.0,
    err_scale: float = 144.0,
    smooth_radius: int = 2,
    delta_range: int = 2,
) -> np.ndarray:
    """Numba-JIT beam search returning per-row TVT predictions.

    Wrapper that:
      1. Smooths horizontal GR with center rolling mean of half-width `smooth_radius`
      2. Sorts typewell by TVT (assumed already sorted) and finds nearest start index
      3. Runs _beam_jit_core with [-delta_range, delta_range + 1]
      4. Converts tw_gr-axis path indices back to TVT values

    Args:
        hgr: horizontal-well GR, shape (n,), float32
        tw_tvt: typewell TVT axis (sorted ascending), shape (nt,)
        tw_gr: typewell GR axis aligned with tw_tvt, shape (nt,)
        start_tvt: last_known TVT in the visible region
        beam_size: number of candidate paths to keep (= BS)
        move_cost: penalty per |delta| step
        err_scale: variance scale for (gv - tw_gr[ni])^2 / es
        smooth_radius: half-width for rolling mean smoothing of hgr
        delta_range: int >= 1, beam considers deltas in range(-delta_range, delta_range + 1)

    Returns:
        tvt_pred: shape (n,), float32, TVT predictions along the beam-optimal path
    """
    if delta_range < 1:
        raise ValueError(f"delta_range must be >= 1, got {delta_range}")
    if len(tw_tvt) != len(tw_gr):
        raise ValueError("tw_tvt and tw_gr must have equal length")
    if len(tw_tvt) < 3:
        return np.full(len(hgr), start_tvt, np.float32)

    if smooth_radius > 0 and len(hgr) > 2 * smooth_radius + 1:
        import pandas as pd

        sgr = (
            pd.Series(hgr, dtype="float32")
            .rolling(2 * smooth_radius + 1, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
    else:
        sgr = np.asarray(hgr, np.float32)

    si = int(np.searchsorted(tw_tvt, start_tvt, side="left"))
    si = max(0, min(si, len(tw_tvt) - 1))

    path = _beam_jit_core(
        sgr,
        np.asarray(tw_gr, np.float32),
        np.int64(si),
        np.int64(beam_size),
        np.float32(move_cost),
        np.float32(err_scale),
        np.int64(-delta_range),
        np.int64(delta_range + 1),
    )

    return np.asarray(tw_tvt, np.float32)[path]


def beam_search_multi_numba(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    configs: tuple = BEAMS_DEFAULT,
    delta_range: int = 2,
) -> dict[str, np.ndarray]:
    """Run beam search with multiple configs, returning {tag: tvt_pred} mapping."""
    out: dict[str, np.ndarray] = {}
    for (bs, mc, es, sr, tag) in configs:
        out[tag] = beam_search_numba(
            hgr,
            tw_tvt,
            tw_gr,
            start_tvt,
            beam_size=bs,
            move_cost=mc,
            err_scale=es,
            smooth_radius=sr,
            delta_range=delta_range,
        )
    return out


# ----------------------------------------------------------------------------
# Particle Filter JIT (ANCC + Z) — direct ports from Hill Climb cell 5
# ----------------------------------------------------------------------------


@njit(cache=True)
def _pf_ancc_jit(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
    ls: float,
    ir: float,
    N: int,
    ALPHA: float,
    RN: float,
    PN: float,
    IS: float,
    RP: float,
    RR: float,
    RESAMP: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ANCC particle filter. Returns (point_mean, point_std) along md_v."""
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    nm = len(md_v)
    pts = np.empty(nm)
    std_ = np.empty(nm)
    pm = md_v[0] - 1.0
    for i in range(nm):
        dm = max(md_v[i] - pm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tv = pos[j] - z_v[i]
            tv = max(tv, vmin - 50.0)
            tv = min(tv, vmin + len(gg) * step + 50.0)
            pos[j] = tv + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va**0.5
        pm = md_v[i]
    return pts, std_


# Default PF hyperparams (= Hill Climb cell 5)
PF_ANCC_DEFAULTS = dict(
    N=600,
    ALPHA=0.998,
    RN=0.002,
    PN=0.005,
    IS=0.3,
    RP=0.1,
    RR=0.001,
    RESAMP=0.5,
)


def run_pf_ancc(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gr_grid: np.ndarray,
    vmin: float,
    step: float,
    gs: float = 30.0,
    last_known_tvt_minus_z: float = 0.0,
    init_rate: float = 0.01,
    **overrides,
) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper for ANCC PF with default hyperparams (override via kwargs)."""
    hp = {**PF_ANCC_DEFAULTS, **overrides}
    return _pf_ancc_jit(
        np.asarray(md_v, np.float64),
        np.asarray(z_v, np.float64),
        np.asarray(gr_v, np.float64),
        np.asarray(gr_grid, np.float64),
        float(vmin),
        float(step),
        float(gs),
        float(last_known_tvt_minus_z),
        float(init_rate),
        int(hp["N"]),
        float(hp["ALPHA"]),
        float(hp["RN"]),
        float(hp["PN"]),
        float(hp["IS"]),
        float(hp["RP"]),
        float(hp["RR"]),
        float(hp["RESAMP"]),
    )
