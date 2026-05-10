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

    Note: this is the visible-prefix-window-matching variant (similar in spirit to
    `self_ncc`). For tasmim's TVT-offset sweep variant, see `xcorr_tvt_offsets`.
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


# ---------- xcorr_tvt_offsets (tasmim-style: TVT-offset sweep around lkt) ----------


def xcorr_tvt_offsets(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_known_tvt: float,
    window: int = 30,
    search_half: float = 100.0,
    step: float = 2.0,
) -> dict[str, np.ndarray]:
    """Per row, find TVT offset around `last_known_tvt` that maximises Pearson
    correlation between local-window observed GR and tw_gr at that offset.

    Returns dict with 3 numpy arrays of length len(hgr):
      - xcorr_delta:    best TVT offset (signed; bounded to ±search_half)
      - xcorr_corr:     best Pearson correlation in [-1, 1]
      - xcorr_absdiff:  mean abs diff between observed window and tw_gr at best offset
                       (999 sentinel when window invalid)

    Vectorised version of tasmim's loop. Builds:
      - tw_grid[K, W] = tw_gr at depth (last_known_tvt + offset[k] + rel[w])
        where rel = -(W//2)..+(W//2). Shape K=n_offsets, W=window+1.
      - obs[N, W]    = sliding window of hgr (edge-padded).
    Then NCC across all (i, k) is a single matmul. ~50× speedup vs Python loop.

    Reference: `_research_kernels/tasmim__lb-11-068.../lb-11-068...py:236-263`.
    """
    n = len(hgr)
    if n == 0:
        return {
            "xcorr_delta": np.zeros(0, np.float32),
            "xcorr_corr": np.zeros(0, np.float32),
            "xcorr_absdiff": np.zeros(0, np.float32),
        }
    hgr = np.asarray(hgr, np.float32)
    tw_tvt = np.asarray(tw_tvt, np.float32)
    tw_gr = np.asarray(tw_gr, np.float32)
    offsets = np.arange(-search_half, search_half + step, step, dtype=np.float32)
    K = len(offsets)
    half = int(window) // 2
    W = 2 * half + 1
    rel = np.arange(-half, half + 1, dtype=np.float32)

    # Build observation matrix: edge-pad hgr → slice with strided windows.
    # NaN handling: replace NaN with global mean for matmul; track validity
    # for absdiff/correlation gating per row.
    fb = float(np.nanmean(hgr)) if np.isfinite(hgr).any() else 0.0
    hgr_f = np.where(np.isfinite(hgr), hgr, fb).astype(np.float32)
    hp = np.pad(hgr_f, half, mode="edge")
    obs = np.lib.stride_tricks.sliding_window_view(hp, W).astype(np.float32)  # (n, W)

    # Build tw grid: positions [K, W] then vectorised np.interp.
    positions = last_known_tvt + offsets[:, None] + rel[None, :]  # (K, W)
    tw_vals = np.interp(positions.ravel(), tw_tvt, tw_gr).reshape(K, W).astype(np.float32)

    # Standardise rows (NCC = (x - mean) / std).
    obs_m = obs.mean(axis=1, keepdims=True)
    obs_s = obs.std(axis=1, keepdims=True) + 1e-6
    obs_n = (obs - obs_m) / obs_s  # (n, W)
    tw_m = tw_vals.mean(axis=1, keepdims=True)
    tw_s = tw_vals.std(axis=1, keepdims=True) + 1e-6
    tw_n = (tw_vals - tw_m) / tw_s  # (K, W)

    corr = (obs_n @ tw_n.T) / W  # (n, K)
    best = corr.argmax(axis=1)
    best_corr = np.clip(corr[np.arange(n), best], -1.0, 1.0).astype(np.float32)
    best_off = offsets[best].astype(np.float32)

    # Absdiff at best offset: |obs[i] - tw_vals[best[i]]|.mean(axis=1)
    abs_mat = np.abs(obs - tw_vals[best])  # (n, W)
    best_abs = abs_mat.mean(axis=1).astype(np.float32)

    # For rows where obs has too few valid samples (or zero variance),
    # fall back to MAE-min and zero corr.
    obs_valid = np.isfinite(hgr).astype(np.int32)
    valid_w = np.lib.stride_tricks.sliding_window_view(
        np.pad(obs_valid, half, mode="edge"), W
    ).sum(axis=1)
    low_quality = (valid_w < max(5, window // 4)) | (obs.std(axis=1) < 1e-3)
    if low_quality.any():
        mae = np.abs(obs[low_quality][:, None, :] - tw_vals[None, :, :]).mean(axis=2)  # (m, K)
        bi = mae.argmin(axis=1)
        best_off[low_quality] = offsets[bi]
        best_corr[low_quality] = 0.0
        best_abs[low_quality] = mae[np.arange(len(bi)), bi].astype(np.float32)
    return {
        "xcorr_delta": best_off,
        "xcorr_corr": best_corr,
        "xcorr_absdiff": best_abs,
    }


# ---------- multi_scale_sc (3 window sizes, romantamrazov-style) ----------


def multi_scale_sc(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    half_windows: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Multi-scale Self-NCC: 3 independent TVT signals at different window sizes.

    Reference: `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/...py:188-209`.
    Returns dict[scale_tag -> (sc_raw_tvt[nh], sc_score[nh])] where tag = "h{hw}".
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for hw in half_windows:
        sc_tvt, sc_score = self_ncc(kgr, ktvt, hgr, half_window=int(hw), stride=stride)
        out[f"h{hw}"] = (sc_tvt, sc_score)
    return out


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
