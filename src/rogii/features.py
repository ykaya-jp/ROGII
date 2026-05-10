"""Feature engineering for ROGII (vectorized, per-well loop for speed).

Per-well features (broadcast to rows):
- visible_n, visible_ratio, visible_md_max
- plane fit (a, b, c, d) where TVT_input = a*X + b*Y + c*Z + d on visible rows
- per-well GR mean / std (for z-score)

Per-row features:
- raw: MD, X, Y, Z, GR
- z-score: GR_z
- trajectory derivatives: dX, dY, dZ, d2Z, lateral_velocity, inclination
- GR rolling (within well): mean / std for windows {10, 50, 200}
- GR diff (within well): {1, 5, 20} step
- plane fit baseline: tvt_planefit_pred
- visible-end deltas: md_to_visible_end, x/y/z_to_visible_end, tvt_input_visible_last

The plane fit is fit on visible rows of THIS well only — for hidden rows
this is an extrapolation, exactly mirroring inference.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

GR_ROLL_WINDOWS = (10, 50, 200)
GR_DIFF_LAGS = (1, 5, 20)


def _planefit(visible_x: np.ndarray, visible_y: np.ndarray, visible_z: np.ndarray, visible_t: np.ndarray) -> tuple[float, float, float, float]:
    """Fit TVT = a*X + b*Y + c*Z + d on visible rows."""
    if len(visible_t) < 4:
        return 0.0, 0.0, 0.0, float(visible_t.mean()) if len(visible_t) else 0.0
    A = np.column_stack([visible_x, visible_y, visible_z, np.ones(len(visible_t))])
    coef, *_ = np.linalg.lstsq(A, visible_t, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])


def _rolling_mean_std(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal (left-aligned) rolling mean and std over a 1D array, min_periods=1."""
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=1).mean().to_numpy()
    sd = s.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
    return m, sd


def _per_well_features(well_df: pd.DataFrame) -> dict[str, np.ndarray | float]:
    """Compute features for one well, return dict of column name → array (or scalar)."""
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

    # Plane fit on visible
    pf_a, pf_b, pf_c, pf_d = _planefit(
        x[visible_mask], y[visible_mask], z[visible_mask], tvt_in[visible_mask]
    )
    tvt_planefit_pred = pf_a * x + pf_b * y + pf_c * z + pf_d

    # GR mean/std
    gr_mean = float(np.nanmean(gr))
    gr_std = float(np.nanstd(gr) + 1e-9)
    gr_z = (gr - gr_mean) / gr_std

    # Visible-end markers
    if visible_n:
        last_vis_idx = int(np.nonzero(visible_mask)[0][-1])
        visible_md_max = float(md[last_vis_idx])
        tvt_input_visible_last = float(tvt_in[last_vis_idx])
        x_vis_end = float(x[last_vis_idx])
        y_vis_end = float(y[last_vis_idx])
        z_vis_end = float(z[last_vis_idx])
    else:
        visible_md_max = float("nan")
        tvt_input_visible_last = float("nan")
        x_vis_end = float("nan")
        y_vis_end = float("nan")
        z_vis_end = float("nan")

    md_to_visible_end = md - visible_md_max
    x_to_visible_end = x - x_vis_end
    y_to_visible_end = y - y_vis_end
    z_to_visible_end = z - z_vis_end

    # Derivatives (concatenate prepend NaN)
    nan = np.float32("nan")
    dX = np.concatenate(([nan], np.diff(x))).astype(np.float32)
    dY = np.concatenate(([nan], np.diff(y))).astype(np.float32)
    dZ = np.concatenate(([nan], np.diff(z))).astype(np.float32)
    d2Z = np.concatenate(([nan], np.diff(dZ))).astype(np.float32)
    lateral_velocity = np.sqrt(np.nan_to_num(dX) ** 2 + np.nan_to_num(dY) ** 2).astype(np.float32)
    inclination = np.arctan2(np.nan_to_num(dZ), np.maximum(lateral_velocity, 1e-9)).astype(np.float32)

    out: dict[str, np.ndarray | float] = {
        "MD": md,
        "X": x,
        "Y": y,
        "Z": z,
        "GR": gr,
        "GR_z": gr_z.astype(np.float32),
        "tvt_planefit_pred": tvt_planefit_pred.astype(np.float32),
        "tvt_input_visible_last": np.full(n, tvt_input_visible_last, dtype=np.float32),
        "md_to_visible_end": md_to_visible_end.astype(np.float32),
        "x_to_visible_end": x_to_visible_end.astype(np.float32),
        "y_to_visible_end": y_to_visible_end.astype(np.float32),
        "z_to_visible_end": z_to_visible_end.astype(np.float32),
        "visible_md_max": np.full(n, visible_md_max, dtype=np.float32),
        "visible_n": np.full(n, visible_n, dtype=np.int32),
        "visible_ratio": np.full(n, visible_ratio, dtype=np.float32),
        "pf_a": np.full(n, pf_a, dtype=np.float32),
        "pf_b": np.full(n, pf_b, dtype=np.float32),
        "pf_c": np.full(n, pf_c, dtype=np.float32),
        "pf_d": np.full(n, pf_d, dtype=np.float32),
        "gr_mean": np.full(n, gr_mean, dtype=np.float32),
        "gr_std": np.full(n, gr_std, dtype=np.float32),
        "dX": dX,
        "dY": dY,
        "dZ": dZ,
        "d2Z": d2Z,
        "lateral_velocity": lateral_velocity,
        "inclination": inclination,
    }

    # GR rolling mean/std for each window (causal, min_periods=1)
    for w in GR_ROLL_WINDOWS:
        m, sd = _rolling_mean_std(gr, w)
        out[f"GR_roll_mean_{w}"] = m.astype(np.float32)
        out[f"GR_roll_std_{w}"] = sd.astype(np.float32)

    # GR diff lags
    for lag in GR_DIFF_LAGS:
        d = np.concatenate(([nan] * lag, gr[lag:] - gr[:-lag])).astype(np.float32)
        out[f"GR_diff_{lag}"] = d

    return out


def add_features(h: pd.DataFrame) -> pd.DataFrame:
    """Append features by iterating per-well (vectorized within each well).

    Expects columns: well, row_idx, MD, X, Y, Z, GR, TVT_input. Sorted in place.
    Returns a new DataFrame with original + feature columns.
    """
    h = h.sort_values(["well", "row_idx"]).reset_index(drop=True)
    well_arr = h["well"].to_numpy()
    n_wells = h["well"].nunique()
    print(f"   processing {n_wells} wells", flush=True)

    # Group by well via numpy boundaries (faster than pandas groupby)
    boundaries = np.concatenate(([0], np.where(well_arr[1:] != well_arr[:-1])[0] + 1, [len(h)]))
    feature_chunks: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    for wi in range(len(boundaries) - 1):
        s, e = boundaries[wi], boundaries[wi + 1]
        feats = _per_well_features(h.iloc[s:e])
        feature_chunks.append(feats)
        if (wi + 1) % 100 == 0 or (wi + 1) == n_wells:
            elapsed = time.perf_counter() - t0
            print(f"   ... {wi + 1}/{n_wells} wells  ({elapsed:.1f}s)", flush=True)

    # Combine — every chunk has the same keys
    keys = list(feature_chunks[0].keys())
    feat_dict: dict[str, np.ndarray] = {}
    for k in keys:
        feat_dict[k] = np.concatenate([c[k] for c in feature_chunks])

    feat_df = pd.DataFrame(feat_dict)
    # row_idx is from input; concat
    feat_df.insert(0, "row_idx", h["row_idx"].to_numpy())
    feat_df.insert(0, "well", h["well"].to_numpy())
    feat_df["TVT_input"] = h["TVT_input"].to_numpy()
    if "TVT" in h.columns:
        feat_df["TVT"] = h["TVT"].to_numpy()

    return feat_df


FEATURE_COLS = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "GR_z",
    "tvt_planefit_pred",
    "tvt_input_visible_last",
    "md_to_visible_end",
    "x_to_visible_end",
    "y_to_visible_end",
    "z_to_visible_end",
    "visible_md_max",
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
