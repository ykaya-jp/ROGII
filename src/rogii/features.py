"""Feature engineering for ROGII (exp002 onwards).

Public-notebook insight (`docs/research/problem-essence.dense.md §1-2`):
    TVT = -Z + ANCC + b_well   (Pearson -1.0, resid_std 0.007 ft)

The decisive features are:
    target = TVT - last_known_TVT  (residual target, full train rows)
    tvt_formula_F = -Z + impute_F(X, Y) + b_well_F   for F in 6 formations
    delta_to_anchor = MD - last_known_MD,  Z - last_known_Z,  X - ..., etc.
    GR rolling / lag / diff (multi-scale)
    typewell-aligned features (Beam/PF/NCC) — added later in exp003+

This module computes all per-well features in a vectorized loop. The
formation imputer is supplied externally (avoid re-building it per well).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .imputers import FORMATIONS, FormationPlaneKNN

GR_ROLL_WINDOWS = (5, 21, 51, 101)
GR_DIFF_LAGS = (1, 5, 15, 30)


def _planefit(visible_x, visible_y, visible_z, visible_t):
    """Fit T = a*X + b*Y + c*Z + d on visible rows."""
    if len(visible_t) < 4:
        return 0.0, 0.0, 0.0, float(visible_t.mean()) if len(visible_t) else 0.0
    A = np.column_stack([visible_x, visible_y, visible_z, np.ones(len(visible_t))])
    coef, *_ = np.linalg.lstsq(A, visible_t, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])


def _rolling_mean_std(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=1, center=True).mean().to_numpy()
    sd = s.rolling(window, min_periods=1, center=True).std().fillna(0.0).to_numpy()
    return m, sd


def _per_well_features(
    well_df: pd.DataFrame, formation_imputed: np.ndarray | None = None
) -> dict[str, Any]:
    """Compute features for one well.

    Returns a dict[name -> array(length n_rows)]. All arrays are float32 unless noted.
    """
    n = len(well_df)
    md = well_df["MD"].to_numpy(np.float32)
    x = well_df["X"].to_numpy(np.float32)
    y = well_df["Y"].to_numpy(np.float32)
    z = well_df["Z"].to_numpy(np.float32)
    gr = well_df["GR"].to_numpy(np.float32)
    tvt_in = well_df["TVT_input"].to_numpy(np.float32)

    visible_mask = ~np.isnan(tvt_in)
    visible_n = int(visible_mask.sum())
    visible_ratio = visible_n / max(n, 1)

    if visible_n:
        last_idx = int(np.nonzero(visible_mask)[0][-1])
        last_md = float(md[last_idx])
        last_x = float(x[last_idx])
        last_y = float(y[last_idx])
        last_z = float(z[last_idx])
        last_tvt = float(tvt_in[last_idx])
    else:
        last_md = last_x = last_y = last_z = last_tvt = float("nan")

    pf_a, pf_b, pf_c, pf_d = _planefit(
        x[visible_mask], y[visible_mask], z[visible_mask], tvt_in[visible_mask]
    )
    tvt_planefit = pf_a * x + pf_b * y + pf_c * z + pf_d

    gr_mean = float(np.nanmean(gr))
    gr_std = float(np.nanstd(gr) + 1e-9)
    gr_z = (gr - gr_mean) / gr_std

    out: dict[str, Any] = {
        "MD": md,
        "X": x,
        "Y": y,
        "Z": z,
        "GR": gr,
        "GR_z": gr_z.astype(np.float32),
        "tvt_planefit": tvt_planefit.astype(np.float32),
        "last_known_TVT": np.full(n, last_tvt, dtype=np.float32),
        "last_known_MD": np.full(n, last_md, dtype=np.float32),
        "md_since": (md - last_md).astype(np.float32),
        "dx_since": (x - last_x).astype(np.float32),
        "dy_since": (y - last_y).astype(np.float32),
        "dz_since": (z - last_z).astype(np.float32),
        "lateral_since": np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2).astype(np.float32),
        "visible_n": np.full(n, visible_n, dtype=np.int32),
        "visible_ratio": np.full(n, visible_ratio, dtype=np.float32),
        "pf_a": np.full(n, pf_a, dtype=np.float32),
        "pf_b": np.full(n, pf_b, dtype=np.float32),
        "pf_c": np.full(n, pf_c, dtype=np.float32),
        "pf_d": np.full(n, pf_d, dtype=np.float32),
        "gr_mean": np.full(n, gr_mean, dtype=np.float32),
        "gr_std": np.full(n, gr_std, dtype=np.float32),
    }

    nan = np.float32("nan")
    dx = np.concatenate(([nan], np.diff(x))).astype(np.float32)
    dy = np.concatenate(([nan], np.diff(y))).astype(np.float32)
    dz = np.concatenate(([nan], np.diff(z))).astype(np.float32)
    out["dX"] = dx
    out["dY"] = dy
    out["dZ"] = dz
    out["d2Z"] = np.concatenate(([nan], np.diff(dz))).astype(np.float32)
    lateral_v = np.sqrt(np.nan_to_num(dx) ** 2 + np.nan_to_num(dy) ** 2).astype(np.float32)
    out["lateral_velocity"] = lateral_v
    out["inclination"] = np.arctan2(np.nan_to_num(dz), np.maximum(lateral_v, 1e-9)).astype(np.float32)

    for w in GR_ROLL_WINDOWS:
        m, sd = _rolling_mean_std(gr, w)
        out[f"GR_roll_mean_{w}"] = m.astype(np.float32)
        out[f"GR_roll_std_{w}"] = sd.astype(np.float32)
    for lag in GR_DIFF_LAGS:
        d = np.concatenate(([nan] * lag, gr[lag:] - gr[:-lag])).astype(np.float32)
        out[f"GR_diff_{lag}"] = d

    if formation_imputed is not None and visible_n:
        for fi, fname in enumerate(FORMATIONS):
            f_imp = formation_imputed[:, fi].astype(np.float32)
            b_F_vec = tvt_in[visible_mask] + z[visible_mask] - f_imp[visible_mask]
            b_F_all = float(np.median(b_F_vec))
            b_F_50 = float(np.median(b_F_vec[-50:])) if visible_n >= 50 else b_F_all
            tvt_F_all = (-z + f_imp + b_F_all).astype(np.float32)
            tvt_F_50 = (-z + f_imp + b_F_50).astype(np.float32)
            out[f"tvtF_{fname}"] = tvt_F_all
            out[f"tvtF50_{fname}"] = tvt_F_50
            out[f"bw_{fname}"] = np.full(n, b_F_all, dtype=np.float32)
            out[f"bw50_{fname}"] = np.full(n, b_F_50, dtype=np.float32)
            out[f"tvtF_{fname}_d"] = (tvt_F_all - last_tvt).astype(np.float32)
            out[f"f_imp_{fname}"] = f_imp

    return out


def add_features(
    h: pd.DataFrame, imputer: FormationPlaneKNN | None = None, exclude_self: bool = True
) -> pd.DataFrame:
    """Per-well loop with optional formation imputation.

    Args:
        h: long DataFrame with all wells concatenated.
        imputer: trained FormationPlaneKNN. If None, formation features are skipped.
        exclude_self: when imputing for a TRAIN well, exclude its own centroid
            (proper leave-one-out). For test wells, set to False (test wells are not
            in the training centroids anyway).
    """
    h = h.sort_values(["well", "row_idx"]).reset_index(drop=True)
    well_arr = h["well"].to_numpy()
    n_wells = h["well"].nunique()
    print(f"   processing {n_wells} wells", flush=True)

    boundaries = np.concatenate(([0], np.where(well_arr[1:] != well_arr[:-1])[0] + 1, [len(h)]))
    chunks: list[dict] = []
    t0 = time.perf_counter()
    for wi in range(len(boundaries) - 1):
        s, e = boundaries[wi], boundaries[wi + 1]
        sub = h.iloc[s:e]
        wid = sub["well"].iloc[0]

        formation_imputed = None
        if imputer is not None:
            xy = sub[["X", "Y"]].to_numpy(np.float64)
            self_wid = wid if exclude_self else None
            formation_imputed, _min_dist = imputer.impute(xy, self_wid=self_wid)

        feats = _per_well_features(sub, formation_imputed=formation_imputed)
        chunks.append(feats)
        if (wi + 1) % 100 == 0 or (wi + 1) == n_wells:
            elapsed = time.perf_counter() - t0
            print(f"   ... {wi + 1}/{n_wells} wells  ({elapsed:.1f}s)", flush=True)

    keys = list(chunks[0].keys())
    feat_dict: dict[str, np.ndarray] = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    feat_df = pd.DataFrame(feat_dict)
    feat_df.insert(0, "row_idx", h["row_idx"].to_numpy())
    feat_df.insert(0, "well", h["well"].to_numpy())
    feat_df["TVT_input"] = h["TVT_input"].to_numpy()
    if "TVT" in h.columns:
        feat_df["TVT"] = h["TVT"].to_numpy()
    return feat_df


def feature_columns(formation_avail: bool = True) -> list[str]:
    base = [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
        "GR_z",
        "tvt_planefit",
        "last_known_TVT",
        "last_known_MD",
        "md_since",
        "dx_since",
        "dy_since",
        "dz_since",
        "lateral_since",
        "visible_n",
        "visible_ratio",
        "pf_a",
        "pf_b",
        "pf_c",
        "pf_d",
        "gr_mean",
        "gr_std",
        "dX",
        "dY",
        "dZ",
        "d2Z",
        "lateral_velocity",
        "inclination",
        *[f"GR_roll_mean_{w}" for w in GR_ROLL_WINDOWS],
        *[f"GR_roll_std_{w}" for w in GR_ROLL_WINDOWS],
        *[f"GR_diff_{lag}" for lag in GR_DIFF_LAGS],
    ]
    if formation_avail:
        for fname in FORMATIONS:
            base += [
                f"tvtF_{fname}",
                f"tvtF50_{fname}",
                f"bw_{fname}",
                f"bw50_{fname}",
                f"tvtF_{fname}_d",
                f"f_imp_{fname}",
            ]
    return base


FEATURE_COLS = feature_columns(formation_avail=True)
