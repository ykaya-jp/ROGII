"""Typewell-signal features — the missing link in exp002.

These are the features that public LB 10-12 notebooks use to teach the model
"where in the typewell does the bit's GR currently match?". exp002 only had
formation-plane geometry; this module adds the GR-matching signal family.

Implements (from `docs/research/notebooks-deepdive.dense.md` priority list):
- affine_gr_cal: linear calibration of horizontal GR to typewell GR axis
- self_ncc: normalized cross-correlation between visible-prefix and hidden GR
- tw_diff: 3 anchors × 11 offsets x typewell GR difference features
- xcorr_tvt: local Pearson correlation maximization (tasmim-style)
- beam_search: discrete typewell-tie path search
- gr_detrend_resid: GR linear detrending residual

All functions accept numpy arrays and return numpy arrays of equal length to
the horizontal-well input. They are pure (no I/O, no global state) so they
can be parallelised with joblib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------- helpers ----------


def _nearest_index(sorted_arr: np.ndarray, v: float) -> int:
    """Index of element in sorted_arr closest to v."""
    i = int(np.searchsorted(sorted_arr, v, side="left"))
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    if i > 0 and abs(sorted_arr[i - 1] - v) <= abs(sorted_arr[i] - v):
        return i - 1
    return i


def _smooth_gr(gr: np.ndarray, fallback: float, radius: int = 2) -> np.ndarray:
    """Interpolate NaN, fill edges, then center-rolling-mean."""
    s = pd.Series(gr, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(np.float32)


# ---------- affine_gr_cal ----------


def affine_gr_cal(
    kgr: np.ndarray, tw_at_kvis: np.ndarray, min_pts: int = 20
) -> tuple[float, float]:
    """Fit kgr ≈ a * tw_at_kvis + b on visible rows.

    Returns (a, b). When fitting fails (too few points or zero variance),
    returns (1, mean(kgr) - mean(tw_at_kvis)).
    """
    valid = np.isfinite(kgr) & np.isfinite(tw_at_kvis)
    if valid.sum() < min_pts or np.std(tw_at_kvis[valid]) < 1e-6:
        if valid.any():
            return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_kvis))
        return 1.0, 0.0
    a, b = np.polyfit(tw_at_kvis[valid], kgr[valid], 1)
    return float(a), float(b)


# ---------- self_ncc ----------


def self_ncc(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    half_window: int = 15,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-correlation NCC.

    For each hidden row, finds the visible-prefix window whose GR shape best
    matches the hidden row's local window, returning the corresponding TVT
    candidate from the visible region.

    Returns (sc_raw_tvt, sc_score). Both length len(hgr).
    """
    win = 2 * half_window + 1
    nk = len(kgr)
    nh = len(hgr)
    if nk < win + 1 or nh == 0:
        last_tvt = float(ktvt[-1]) if nk else 0.0
        return np.full(nh, last_tvt, np.float32), np.zeros(nh, np.float32)
    kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
    hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
    sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
    M = len(sts)
    if M == 0:
        last_tvt = float(ktvt[-1])
        return np.full(nh, last_tvt, np.float32), np.zeros(nh, np.float32)
    C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
    Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
    hp = np.pad(hg, half_window, mode="edge")
    H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
    Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
    ncc = Hn @ Cn.T / win
    best = ncc.argmax(1)
    score = ncc.max(1).astype(np.float32)
    centers = np.clip(sts[best] + half_window, 0, nk - 1)
    return ktvt[centers].astype(np.float32), score


# ---------- tw_diff (3 anchors × N offsets) ----------


def tw_diff_features(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    anchor_last: float,
    anchor_beam: np.ndarray | None = None,
    anchor_sc: np.ndarray | None = None,
    offsets_last: tuple[int, ...] = (-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80),
    offsets_beam: tuple[int, ...] = (-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40),
    offsets_sc: tuple[int, ...] = (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30),
) -> dict[str, np.ndarray]:
    """For each hidden-row GR, compute (gr_horizontal − tw_GR(anchor + offset)).

    Returns dict of {f"tda{offset}": ..., f"tdbc{offset}": ..., f"tdsc{offset}": ...}
    """
    out: dict[str, np.ndarray] = {}
    for o in offsets_last:
        ref = float(np.interp(anchor_last + o, tw_tvt, tw_gr))
        out[f"tda{int(o)}"] = (hgr - ref).astype(np.float32)
    if anchor_beam is not None:
        for o in offsets_beam:
            ref = np.interp(anchor_beam + o, tw_tvt, tw_gr).astype(np.float32)
            out[f"tdbc{int(o)}"] = (hgr - ref).astype(np.float32)
    if anchor_sc is not None:
        for o in offsets_sc:
            ref = np.interp(anchor_sc + o, tw_tvt, tw_gr).astype(np.float32)
            out[f"tdsc{int(o)}"] = (hgr - ref).astype(np.float32)
    return out


# ---------- beam_search ----------


def beam_search(
    gr_horiz: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_size: int = 10,
    move_cost: float = 20.0,
    emit_scale: float = 144.0,
    smooth_radius: int = 2,
) -> np.ndarray:
    """Beam-search the optimal typewell index path for a horizontal GR sequence.

    State: typewell index i (= TVT axis). Transition: i → {i-1, i, i+1}.
    Emit: (gr_horiz - tw_gr[i])^2 / emit_scale.
    Move: move_cost * |delta|.
    """
    tw_tvt = np.asarray(tw_tvt, np.float32)
    tw_gr = np.asarray(tw_gr, np.float32)
    T = len(tw_tvt)
    fb = float(np.nanmean(tw_gr))
    sg = _smooth_gr(gr_horiz.astype(np.float32), fb, radius=smooth_radius)
    si = _nearest_index(tw_tvt, start_tvt)
    bi = np.full(beam_size, si, np.int32)
    bc = np.zeros(beam_size, np.float64)
    ns = len(sg)
    bps = np.empty((ns, beam_size), np.int32)
    bpb = np.empty((ns, beam_size), np.int32)
    for s, gv in enumerate(sg):
        ci = np.clip(bi[:, None] + np.array([-1, 0, 1]), 0, T - 1)
        em = (gv - tw_gr[ci]) ** 2 / emit_scale
        mv = move_cost * np.array([1, 0, 1])[None, :]
        cc = bc[:, None] + em + mv
        fi = ci.ravel()
        fc = cc.ravel()
        fp = np.repeat(np.arange(beam_size), 3)
        order = np.argsort(fc, kind="stable")
        kept = []
        seen = set()
        for o in order:
            t = int(fi[o])
            if t not in seen:
                seen.add(t)
                kept.append(o)
            if len(kept) == beam_size:
                break
        while len(kept) < beam_size:
            kept.append(kept[-1])
        kept = np.asarray(kept, np.int32)
        bps[s] = fp[kept]
        bpb[s] = fi[kept]
        bi = fi[kept].astype(np.int32)
        bc = fc[kept]
    path = np.empty(ns, np.int32)
    cb = int(np.argmin(bc))
    for s in range(ns - 1, -1, -1):
        path[s] = bpb[s, cb]
        cb = bps[s, cb]
    return tw_tvt[path]


# Standard 5 beam configs (tags: cons / loose / vcons / sm5 / vloose)
DEFAULT_BEAM_CONFIGS = (
    ("cons", 10, 20.0, 144.0, 2),
    ("loose", 10, 8.0, 64.0, 2),
    ("vcons", 8, 35.0, 220.0, 1),
    ("sm5", 10, 14.0, 90.0, 5),
    ("vloose", 20, 4.0, 36.0, 3),
)


def beam_search_multi(
    gr_horiz: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    configs: tuple = DEFAULT_BEAM_CONFIGS,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tag, bs, mc, es, r in configs:
        out[tag] = beam_search(gr_horiz, tw_tvt, tw_gr, start_tvt, bs, mc, es, r)
    return out


# ---------- xcorr_tvt (tasmim-style local Pearson) ----------


def xcorr_tvt(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    half_window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Like self_ncc but uses Pearson correlation (not centered NCC) over a longer
    window. Higher resolution at the cost of more compute. Returns (xcorr_tvt, score).
    """
    win = 2 * half_window + 1
    nk = len(kgr)
    nh = len(hgr)
    if nk < win + 1 or nh == 0:
        last = float(ktvt[-1]) if nk else 0.0
        return np.full(nh, last, np.float32), np.zeros(nh, np.float32)
    kg = pd.Series(kgr).rolling(3, center=True, min_periods=1).mean().to_numpy(np.float32)
    hg = pd.Series(hgr).rolling(3, center=True, min_periods=1).mean().to_numpy(np.float32)
    starts = np.arange(0, nk - win + 1, max(1, half_window // 4), dtype=np.int32)
    M = len(starts)
    if M == 0:
        last = float(ktvt[-1])
        return np.full(nh, last, np.float32), np.zeros(nh, np.float32)
    C = kg[starts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
    Cm = C.mean(1, keepdims=True)
    Cs = C.std(1, keepdims=True) + 1e-6
    Cn = (C - Cm) / Cs
    hp = np.pad(hg, half_window, mode="edge")
    H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
    Hm = H.mean(1, keepdims=True)
    Hs = H.std(1, keepdims=True) + 1e-6
    Hn = (H - Hm) / Hs
    corr = Hn @ Cn.T / win
    best = corr.argmax(1)
    score = corr.max(1).astype(np.float32)
    ctrs = np.clip(starts[best] + half_window, 0, nk - 1)
    return ktvt[ctrs].astype(np.float32), score


# ---------- gr_detrend_resid ----------


def gr_detrend_resid(md: np.ndarray, gr: np.ndarray) -> np.ndarray:
    """GR with a linear MD trend removed (well-specific). Returns residual."""
    valid = np.isfinite(gr)
    if valid.sum() < 5:
        return np.zeros_like(gr, dtype=np.float32)
    a, b = np.polyfit(md[valid], gr[valid], 1)
    return (gr - (a * md + b)).astype(np.float32)


# ---------- wls_b_well (exponential decay weighted) ----------


def wls_b_well(
    visible_tvt_input: np.ndarray, visible_z: np.ndarray, visible_f_imp: np.ndarray, decay: float = 0.02
) -> float:
    """Exponentially-weighted estimate of b_well (recent visible rows weighted higher).

    b_F = TVT_input + Z - f_imp; weight = exp(-decay * distance_from_end).
    """
    n = len(visible_tvt_input)
    if n == 0:
        return 0.0
    vals = visible_tvt_input + visible_z - visible_f_imp
    dist = np.arange(n - 1, -1, -1, dtype=np.float32)  # 0 at last visible row
    w = np.exp(-decay * dist)
    return float(np.sum(vals * w) / np.sum(w))
