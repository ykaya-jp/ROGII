"""Phase A4 + A5: Segment b_well per formation × phase + Score-weighted Softmax NCC.

Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
Strength analysis: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md
  §1.3 (Numba beam ±2 shared) and §2 diff matrix (= "Hill Climb has these too").
Criteria: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A4 + AC-A5.

A4 (`seg_b_well`): partition each visible region into early/mid/late thirds + WLS
tail-upweighted, return 5 b_well variants. Hill Climb's `build_well` uses these
per formation (bw_early_<fn>, bw_mid_<fn>, bw_late_<fn>, bw_wls_<fn>) to capture
b_well phase drift. data-driven boundary = N/3 thirds, never hard-coded depth.

A5 (`multi_scale_ncc`): NCC at hws ∈ (8, 15, 25), then softmax-weighted ensemble
across scales (sw = exp(3·score) / Σ). Returns per-scale (tvt, score) tuples and
the score-weighted ensemble signal. Hill Climb's NEW addition over Self-NCC.

License / attribution:
  - seg_b_well + multi_scale_ncc: ported from
    https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
    cell 5 (Apache-2.0 fair-use; also identical in LB-9.830 LGB+XGB notebook)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def seg_b_well(
    ktvt: np.ndarray,
    kz: np.ndarray,
    form_col: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Segment b_well: early / mid / late thirds + WLS tail-upweighted + full prefix.

    b_well = TVT + Z - formation_top. For each well's visible region, compute
    5 variants of b_well median so the model can pick up phase-dependent drift.

    Args:
        ktvt: visible TVT values, shape (n,)
        kz: visible Z values, shape (n,)
        form_col: formation_top at the visible rows, shape (n,)

    Returns:
        (b_full, b_early, b_mid, b_late, b_wls):
            - b_full: median of all b_well values (= legacy single estimate)
            - b_early: median of first third (= depth/MD early phase)
            - b_mid: median of mid third
            - b_late: median of tail max(0, n-50) rows (= recent stable phase)
            - b_wls: WLS tail-upweighted mean (= exp(0.02 · i) weights normalised)
    """
    bv = ktvt + kz - form_col
    n = len(bv)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1 = n // 3
    t2 = 2 * n // 3
    b_early = float(np.median(bv[: max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1 : max(t1 + 1, t2)])) if t2 > t1 else b_full
    # WLS tail-upweighted: exp(0.02 * i) normalised so tail rows weigh more
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def multi_scale_ncc(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    hws: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Multi-scale Normalised Cross-Correlation + score-weighted softmax ensemble.

    For each half-window hw ∈ hws, compute NCC between rolling-smoothed visible
    GR templates and hidden GR. The per-scale (tvt_match, score) tuples are
    returned, plus a softmax-weighted ensemble signal across scales:
        sw_i = exp(3 · score_i) / Σ_j exp(3 · score_j)
        sc_ens = Σ_i sw_i · tvt_i

    Args:
        kgr: known/visible GR sequence, shape (nk,)
        ktvt: known/visible TVT aligned with kgr, shape (nk,)
        hgr: hidden GR sequence to be matched, shape (nh,)
        hws: tuple of half-window sizes
        stride: stride for template start positions in visible region

    Returns:
        (per_scale, sc_ens):
            per_scale: list of (tvt_match: shape (nh,), score: shape (nh,)) one per hw
            sc_ens: shape (nh,), softmax-weighted ensemble TVT prediction
    """
    nk = len(kgr)
    nh = len(hgr)
    last_tvt = float(ktvt[-1]) if nk else 0.0
    out: list[tuple[np.ndarray, np.ndarray]] = []

    for hw in hws:
        win = 2 * hw + 1
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, last_tvt, np.float32), np.zeros(nh, np.float32)))
            continue
        kg = (
            pd.Series(kgr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        hg = (
            pd.Series(hgr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, last_tvt, np.float32), np.zeros(nh, np.float32)))
            continue
        # build template matrix: M templates × win cols
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(
            np.float32
        )
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win  # shape (nh, M)
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        tvt_match = ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32)
        out.append((tvt_match, score))

    # softmax across scales (Hill Climb cell 5: sw = exp(3.*scores))
    tvts = np.stack([o[0] for o in out], axis=1)  # (nh, n_scales)
    scores = np.stack([o[1] for o in out], axis=1)  # (nh, n_scales)
    sw = np.exp(3.0 * scores)
    sw = sw / (sw.sum(axis=1, keepdims=True) + 1e-9)
    sc_ens = (tvts * sw).sum(axis=1).astype(np.float32)
    return out, sc_ens
