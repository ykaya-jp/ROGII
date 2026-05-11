"""Particle Filter for ROGII typewell GR matching (ANCC + Z variants).

Inspired by romantamrazov/rogii-super-solution-lb-top-3 (Apache 2.0):
  _research_kernels/romantamrazov__rogii-super-solution-lb-top-3/
    rogii-super-solution-lb-top-3.py:223-313

Self-reimplemented for ROGII-exp004 with:
  - Explicit type hints on public API.
  - Numerical floor `lk = max(lk, 1e-300)` to prevent particle collapse.
  - Position clip `tmin - 50, tmax + 50` (50 ft buffer).
  - Resampling fallback when total weight collapses to zero.
  - Self-exclusion placeholder in public signature (`self_wid`).

Public API:
  - `run_pf_ancc(hw, tw_tvt, tw_gr, *, n_particles, rng_seed)` -> (pts, std)
  - `run_pf_z(hw, tw_tvt, tw_gr, *, n_particles, rng_seed)` -> (pts, std)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class PFConfig:
    """Particle-filter hyperparameters (defaults match upstream §2)."""

    n_particles: int = 500
    momentum: float = 0.993
    vel_noise: float = 0.005
    pos_noise: float = 0.01
    gr_sigma_min: float = 10.0
    gr_sigma_max: float = 60.0
    gr_sigma_default: float = 30.0
    init_v_std: float = 0.02
    init_spread: float = 0.5
    resample_thresh: float = 0.5
    rough_pos: float = 0.2
    rough_vel: float = 0.003
    gr_win: int = 5
    gr_weight: float = 0.3

    # ANCC-specific (separate momentum / noise).
    ancc_alpha: float = 0.998
    ancc_rate_noise: float = 0.002
    ancc_pos_noise: float = 0.005
    ancc_init_rate: float = 0.01
    ancc_init_spread: float = 0.3
    ancc_resamp_pos: float = 0.1
    ancc_resamp_rate: float = 0.001


PF_DEFAULTS = PFConfig()


def _gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray, cfg: PFConfig) -> float:
    """Robust per-well GR sigma estimate, clipped to [min, max]."""
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return cfg.gr_sigma_default
    diff = kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    return float(np.clip(np.std(diff), cfg.gr_sigma_min, cfg.gr_sigma_max))


def _safe_normalize(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s > 0:
        return w / s
    return np.full_like(w, 1.0 / len(w))


def run_pf_ancc(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    *,
    n_particles: int | None = None,
    rng_seed: int = 42,
    cfg: PFConfig = PF_DEFAULTS,
    self_wid: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ANCC-modeled Particle Filter.

    Tracks (pos = TVT + Z, rate = d pos / d MD). Likelihood from GR alignment.

    Parameters
    ----------
    hw : pd.DataFrame
        Horizontal-well dataframe with columns: MD, Z, GR, TVT_input.
    tw_tvt, tw_gr : np.ndarray
        Typewell TVT and GR arrays (1D).
    n_particles : int, optional
        Override `cfg.n_particles`.
    rng_seed : int, default 42
        Random seed for deterministic output.
    cfg : PFConfig
        Hyperparameters.
    self_wid : str | None
        Reserved for cross-well self-exclusion (currently unused at PF level;
        typewell is per-well already, so leak risk is at spatial imputer).

    Returns
    -------
    pts : float32[n_eval]
        Predicted TVT at each eval row.
    std : float32[n_eval]
        Particle posterior std.
    """
    _ = self_wid  # reserved
    n = cfg.n_particles if n_particles is None else int(n_particles)
    rng = np.random.default_rng(rng_seed)
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    gs = _gr_sigma(hw, tw_tvt, tw_gr, cfg)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    if m.sum() >= 3:
        init_rate = float(np.median((dt + dz)[m] / dm[m]))
    else:
        init_rate = 0.0

    pos = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1]) + rng.normal(
        0.0, cfg.ancc_init_spread, n
    )
    rate = init_rate + rng.normal(0.0, cfg.ancc_init_rate, n)
    w = np.ones(n) / n

    md_v = ev["MD"].values
    z_v = ev["Z"].values
    gr_v = ev["GR"].values
    pm = float(kn["MD"].iloc[-1])
    pts = np.empty(len(ev))
    std_o = np.empty(len(ev))

    for i in range(len(ev)):
        dm2 = max(md_v[i] - pm, 1.0)
        rate = cfg.ancc_alpha * rate + rng.normal(0.0, cfg.ancc_rate_noise, n)
        pos = pos + rate * dm2 + rng.normal(0.0, cfg.ancc_pos_noise, n)
        tvt_e = np.clip(pos - z_v[i], tmin - 50.0, tmax + 50.0)
        pos = tvt_e + z_v[i]
        if not np.isnan(gr_v[i]):
            eg = np.interp(tvt_e, tw_tvt, tw_gr)
            lk = np.exp(-0.5 * ((gr_v[i] - eg) / gs) ** 2)
            lk = np.maximum(lk, 1e-300)
            w = _safe_normalize(w * lk)
        ne = 1.0 / np.sum(w * w)
        if ne < cfg.resample_thresh * n:
            cdf = np.cumsum(w)
            ix = np.searchsorted(cdf, (np.arange(n) + rng.uniform()) / n)
            ix = np.clip(ix, 0, n - 1)
            pos = pos[ix]
            rate = rate[ix]
            w[:] = 1.0 / n
            pos = pos + rng.normal(0.0, cfg.ancc_resamp_pos, n)
            rate = rate + rng.normal(0.0, cfg.ancc_resamp_rate, n)
        tv = float(np.average(pos - z_v[i], weights=w))
        pts[i] = tv
        std_o[i] = float(np.sqrt(np.average((pos - z_v[i] - tv) ** 2, weights=w)))
        pm = md_v[i]

    return pts.astype(np.float32), std_o.astype(np.float32)


def run_pf_z(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    *,
    n_particles: int | None = None,
    rng_seed: int = 42,
    cfg: PFConfig = PF_DEFAULTS,
    self_wid: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-coupled Particle Filter (TVT position tracked, vel linked to dZ/dMD).

    Returns (TVT pts, posterior std).
    """
    _ = self_wid  # reserved
    n = cfg.n_particles if n_particles is None else int(n_particles)
    rng = np.random.default_rng(rng_seed + 1)

    tw_smooth = (
        pd.Series(tw_gr).rolling(cfg.gr_win, center=True, min_periods=1).mean().values
    )
    tf_primary = interp1d(
        tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1])
    )
    tf_smooth = interp1d(
        tw_tvt,
        tw_smooth,
        bounds_error=False,
        fill_value=(tw_smooth[0], tw_smooth[-1]),
    )
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    gs = _gr_sigma(hw, tw_tvt, tw_gr, cfg)

    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        try:
            c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        except np.linalg.LinAlgError:
            c = np.array([-1.0, 0.0])
        beta, icpt = float(c[0]), float(c[1])
        zsig = max(float(np.std(vt - (beta * vz + icpt))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1

    tail = kna.tail(20)
    dvt2 = np.diff(tail["TVT_input"].values)
    dmd2 = np.diff(tail["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0

    gr_sm = hw["GR"].rolling(cfg.gr_win, center=True, min_periods=1).mean()
    pos = float(kna["TVT_input"].iloc[-1]) + rng.normal(0.0, cfg.init_spread, n)
    vel = iv + rng.normal(0.0, cfg.init_v_std, n)
    w = np.ones(n) / n

    md_v = ev["MD"].values
    gr_v = ev["GR"].values
    z_v = ev["Z"].values
    pm = float(kna["MD"].iloc[-1])
    pz = float(kna["Z"].iloc[-1])
    pts = np.empty(len(ev))
    std_o = np.empty(len(ev))

    for i, idx in enumerate(ev.index):
        dm = max(md_v[i] - pm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        vel = cfg.momentum * vel + rng.normal(0.0, cfg.vel_noise, n)
        pos = pos + vel * dm + rng.normal(0.0, cfg.pos_noise, n)
        pos = np.clip(pos, tmin - 50.0, tmax + 50.0)
        if not np.isnan(gr_v[i]):
            ep = tf_primary(pos)
            lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)
            try:
                gsm = gr_sm.iloc[hw.index.get_loc(idx)]
            except KeyError:
                gsm = np.nan
            if not np.isnan(gsm):
                ls2 = np.exp(-0.5 * ((gsm - tf_smooth(pos)) / (gs * 1.5)) ** 2)
                lk = (1 - cfg.gr_weight) * lp + cfg.gr_weight * ls2
            else:
                lk = lp
            lk = np.maximum(lk, 1e-300)
            w = _safe_normalize(w * lk)
        lz = np.exp(-0.5 * ((vel - ve) / max(zsig * 2.0, 0.005)) ** 2)
        lz = np.maximum(lz, 1e-300)
        w = _safe_normalize(w * lz)
        ne = 1.0 / np.sum(w * w)
        if ne < cfg.resample_thresh * n:
            cdf = np.cumsum(w)
            ix = np.searchsorted(cdf, (np.arange(n) + rng.uniform()) / n)
            ix = np.clip(ix, 0, n - 1)
            pos = pos[ix]
            vel = vel[ix]
            w[:] = 1.0 / n
            pos = pos + rng.normal(0.0, cfg.rough_pos, n)
            vel = vel + rng.normal(0.0, cfg.rough_vel, n)
        pts[i] = np.average(pos, weights=w)
        std_o[i] = np.sqrt(np.average((pos - pts[i]) ** 2, weights=w))
        pm = md_v[i]
        pz = z_v[i]

    return pts.astype(np.float32), std_o.astype(np.float32)
