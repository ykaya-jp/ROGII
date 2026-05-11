"""Numba JIT Beam Search for ROGII typewell GR matching.

Inspired by romantamrazov/rogii-super-solution-lb-top-3 (Apache 2.0):
  _research_kernels/romantamrazov__rogii-super-solution-lb-top-3/
    rogii-super-solution-lb-top-3.py:108-144

Self-reimplemented for ROGII-exp004 with explicit type hints, leak-safe
public API (`self_wid` placeholder), numerical floor, and configurable
delta range. Numba @njit inner kernel is pure numerical (no Python objects).

The Beam Search aligns a horizontal-well GR series (`sgr`) onto the typewell
GR series (`tw_gr`) under a 1-step-delta state transition with cost:

    cost = sum_step (gv - tw_gr[idx])**2 / emit_scale + move_cost * |delta|

with `delta in [-2, +2]` for 7 beam configs.

Public API:
  - `beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r)`
    Returns aligned TVT values for each step in `gr_h`.
  - `BEAM_CONFIGS`: list of 7 (bs, mc, es, smooth_radius, tag) tuples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# 7 beam configs (preserves diversity for ensemble), inspired by upstream §1.1.
BEAM_CONFIGS: list[tuple[int, float, float, int, str]] = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]


@njit(cache=True)
def _beam_kernel(
    sgr: np.ndarray,
    tw_gr: np.ndarray,
    start_idx: np.int64,
    beam_size: np.int64,
    move_cost: float,
    emit_scale: float,
) -> np.ndarray:
    """Pure-numerical Numba JIT Beam Search inner kernel.

    Parameters
    ----------
    sgr : float64[n]
        Smoothed horizontal-well GR (state-evidence sequence).
    tw_gr : float64[n_tw]
        Typewell GR (target index space).
    start_idx : int64
        Initial typewell index (= nearest tw_tvt to start_tvt).
    beam_size : int64
        Beam width (BS).
    move_cost : float
        Penalty per unit delta in typewell index step.
    emit_scale : float
        Emission variance scale (gv - tw_gr[idx])**2 / emit_scale.

    Returns
    -------
    path : int64[n]
        Typewell index chosen for each horizontal step.
    """
    n = len(sgr)
    n_tw = len(tw_gr)
    cap = beam_size * 5

    # Active beams (idx, cost) ranking.
    beam_idx = np.zeros(beam_size, dtype=np.int64)
    beam_idx[0] = start_idx
    beam_cost = np.full(beam_size, 1e30, dtype=np.float64)
    beam_cost[0] = 0.0
    n_active = np.int64(1)

    # Backtrack tables.
    hist_idx = np.zeros((n, beam_size), dtype=np.int64)
    hist_parent = np.zeros((n, beam_size), dtype=np.int64)

    # Candidate buffers (re-used per step).
    cand_idx = np.zeros(cap, dtype=np.int64)
    cand_cost = np.full(cap, 1e30, dtype=np.float64)
    cand_parent = np.zeros(cap, dtype=np.int64)

    for step in range(n):
        gv = sgr[step]
        n_cand = np.int64(0)
        for bi in range(n_active):
            cur_idx = beam_idx[bi]
            cur_cost = beam_cost[bi]
            # delta in [-2, +2]
            for d in range(-2, 3):
                next_idx = cur_idx + d
                if next_idx < 0 or next_idx >= n_tw:
                    continue
                ad = d if d >= 0 else -d
                tot = (
                    cur_cost
                    + (gv - tw_gr[next_idx]) * (gv - tw_gr[next_idx]) / emit_scale
                    + move_cost * ad
                )
                # Merge by destination idx: keep min cost.
                found = np.int64(-1)
                for ci in range(n_cand):
                    if cand_idx[ci] == next_idx:
                        found = ci
                        break
                if found >= 0:
                    if tot < cand_cost[found]:
                        cand_cost[found] = tot
                        cand_parent[found] = bi
                else:
                    if n_cand < cap:
                        cand_idx[n_cand] = next_idx
                        cand_cost[n_cand] = tot
                        cand_parent[n_cand] = bi
                        n_cand += 1

        # Selection sort top-K by cost (K = min(beam_size, n_cand)).
        kept = min(beam_size, n_cand)
        for i in range(kept):
            min_i = i
            for j in range(i + 1, n_cand):
                if cand_cost[j] < cand_cost[min_i]:
                    min_i = j
            if min_i != i:
                tmp_i = cand_idx[i]
                cand_idx[i] = cand_idx[min_i]
                cand_idx[min_i] = tmp_i
                tmp_c = cand_cost[i]
                cand_cost[i] = cand_cost[min_i]
                cand_cost[min_i] = tmp_c
                tmp_p = cand_parent[i]
                cand_parent[i] = cand_parent[min_i]
                cand_parent[min_i] = tmp_p
        hist_idx[step, :kept] = cand_idx[:kept]
        hist_parent[step, :kept] = cand_parent[:kept]
        beam_idx[:kept] = cand_idx[:kept]
        beam_cost[:kept] = cand_cost[:kept]
        n_active = kept

    # Pick best terminal beam.
    best = np.int64(0)
    for b in range(1, n_active):
        if beam_cost[b] < beam_cost[best]:
            best = b
    path = np.zeros(n, dtype=np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hist_idx[s, b]
        b = hist_parent[s, b]
    return path


def _nearest_index(arr: np.ndarray, v: float) -> int:
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(v: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    s = pd.Series(v, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(np.float32)


def beam_search(
    gr_h: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    bs: int = 10,
    mc: float = 20.0,
    es: float = 144.0,
    r: int = 2,
) -> np.ndarray:
    """Beam-search GR alignment, returning the aligned TVT trajectory.

    Parameters
    ----------
    gr_h : np.ndarray
        Horizontal-well GR series.
    tw_tvt : np.ndarray
        Typewell TVT values.
    tw_gr : np.ndarray
        Typewell GR values.
    start_tvt : float
        Last-known TVT of the horizontal-well (eval-zone anchor).
    bs : int, default 10
        Beam width.
    mc : float, default 20.0
        Move cost per delta unit.
    es : float, default 144.0
        Emission scale (squared-error denominator).
    r : int, default 2
        Smoothing radius for `gr_h`.

    Returns
    -------
    tvt_path : np.ndarray (float32)
        Aligned TVT at each step in `gr_h`.
    """
    si = _nearest_index(tw_tvt, float(start_tvt))
    fallback = float(np.nanmean(tw_gr))
    sgr = _smooth(gr_h, fallback, r).astype(np.float64)
    tw_gr64 = tw_gr.astype(np.float64)
    path = _beam_kernel(sgr, tw_gr64, np.int64(si), np.int64(bs), float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def warmup() -> None:
    """Warm the JIT cache with a tiny synthetic problem."""
    rng = np.random.default_rng(0)
    _beam_kernel(
        rng.standard_normal(30),
        rng.standard_normal(50),
        np.int64(25),
        np.int64(8),
        15.0,
        100.0,
    )
