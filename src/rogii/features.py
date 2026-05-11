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
from .tysig import (
    DEFAULT_BEAM_CONFIGS,
    affine_gr_cal,
    beam_search_multi,
    gr_detrend_resid,
    multi_scale_sc,
    self_ncc,
    tw_diff_features,
    wls_b_well,
    xcorr_tvt,
    xcorr_tvt_offsets,
)

GR_ROLL_WINDOWS = (5, 21, 51, 101)
GR_DIFF_LAGS = (1, 5, 15, 30)
BEAM_TAGS = tuple(c[0] for c in DEFAULT_BEAM_CONFIGS)
SC_HALF_WINDOWS = (8, 15, 25)  # multi-scale Self-NCC (romantamrazov-style)


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
    well_df: pd.DataFrame,
    formation_imputed: np.ndarray | None = None,
    typewell: tuple[np.ndarray, np.ndarray] | None = None,
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
            # WLS (recent-weighted) b_well — romantamrazov-style.
            # Reference: `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/...py:181-186`.
            b_F_wls = wls_b_well(
                tvt_in[visible_mask].astype(np.float64),
                z[visible_mask].astype(np.float64),
                f_imp[visible_mask].astype(np.float64),
                decay=0.02,
            )
            tvt_F_all = (-z + f_imp + b_F_all).astype(np.float32)
            tvt_F_50 = (-z + f_imp + b_F_50).astype(np.float32)
            tvt_F_wls = (-z + f_imp + b_F_wls).astype(np.float32)
            out[f"tvtF_{fname}"] = tvt_F_all
            out[f"tvtF50_{fname}"] = tvt_F_50
            out[f"tvtFw_{fname}"] = tvt_F_wls
            out[f"bw_{fname}"] = np.full(n, b_F_all, dtype=np.float32)
            out[f"bw50_{fname}"] = np.full(n, b_F_50, dtype=np.float32)
            out[f"bww_{fname}"] = np.full(n, b_F_wls, dtype=np.float32)
            out[f"tvtF_{fname}_d"] = (tvt_F_all - last_tvt).astype(np.float32)
            out[f"tvtFw_{fname}_d"] = (tvt_F_wls - last_tvt).astype(np.float32)
            out[f"f_imp_{fname}"] = f_imp

    # Typewell-aligned features (Beam / Self-NCC / tw_diff / xcorr / detrend / affine cal)
    if typewell is not None and visible_n:
        tw_tvt, tw_gr = typewell
        tw_tvt = np.asarray(tw_tvt, np.float32)
        tw_gr = np.asarray(tw_gr, np.float32)

        # Affine GR cal on visible: kgr ≈ a * tw_gr(known_TVT) + b
        kgr = gr[visible_mask]
        ktvt = tvt_in[visible_mask]
        tw_at_kvis = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        a_cal, b_cal = affine_gr_cal(kgr, tw_at_kvis)
        out["a_cal"] = np.full(n, a_cal, dtype=np.float32)
        out["b_cal"] = np.full(n, b_cal, dtype=np.float32)

        # GR detrending (linear MD trend removed)
        out["gr_detrend"] = gr_detrend_resid(md, gr)

        # Multi-scale Self-NCC: 3 window sizes (8/15/25) → 3 independent TVT signals.
        # Reference: `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/...py:188-209`.
        sc_results = multi_scale_sc(kgr, ktvt, gr, half_windows=SC_HALF_WINDOWS, stride=3)
        for hw in SC_HALF_WINDOWS:
            sc_tvt_hw, sc_score_hw = sc_results[f"h{hw}"]
            out[f"sc{hw}_raw_tvt"] = sc_tvt_hw
            out[f"sc{hw}_score"] = sc_score_hw
            out[f"sc{hw}_d"] = (sc_tvt_hw - last_tvt).astype(np.float32)
        # Multi-scale aggregates (median + std across 3 windows)
        sc_arr = np.stack([sc_results[f"h{hw}"][0] for hw in SC_HALF_WINDOWS], axis=1)
        sc_raw_med = np.median(sc_arr, axis=1).astype(np.float32)
        out["sc_med_tvt"] = sc_raw_med
        out["sc_std_tvt"] = sc_arr.std(axis=1).astype(np.float32)
        out["sc_med_d"] = (sc_raw_med - last_tvt).astype(np.float32)
        sc_trust = float(np.clip(visible_n / 200.0, 0.0, 0.6))
        out["sc_trust"] = np.full(n, sc_trust, dtype=np.float32)
        # Use the medium-scale (h15) signal as the anchor for tw_diff (sc_raw)
        sc_raw = sc_results["h15"][0]

        # XCorr (visible-prefix-window matching, longer window)
        xc_tvt, xc_score = xcorr_tvt(kgr, ktvt, gr, half_window=30)
        out["xcorr_tvt"] = xc_tvt
        out["xcorr_score"] = xc_score
        out["xcorr_d"] = (xc_tvt - last_tvt).astype(np.float32)

        # Tasmim-style xcorr: TVT-offset sweep around last_known_TVT.
        # Reference: `_research_kernels/tasmim__lb-11-068.../lb-11-068...py:236-263`.
        xco = xcorr_tvt_offsets(
            gr.astype(np.float32),
            tw_tvt,
            tw_gr,
            last_known_tvt=last_tvt,
            window=30,
            search_half=100.0,
            step=2.0,
        )
        out["xcorr_delta"] = xco["xcorr_delta"]
        out["xcorr_corr"] = xco["xcorr_corr"]
        out["xcorr_absdiff"] = xco["xcorr_absdiff"]
        # Implied TVT from offset (around lkt)
        out["xcorr_tvt_off"] = (last_tvt + xco["xcorr_delta"]).astype(np.float32)

        # Beam search 5 configs over the entire well GR
        bpaths = beam_search_multi(gr, tw_tvt, tw_gr, start_tvt=last_tvt)
        beam_arr = np.stack([bpaths[t] for t in BEAM_TAGS], axis=1)  # (n, 5)
        for tag in BEAM_TAGS:
            out[f"beam_{tag}"] = bpaths[tag]
            out[f"beam_{tag}_d"] = (bpaths[tag] - last_tvt).astype(np.float32)
        out["beam_mean"] = beam_arr.mean(1).astype(np.float32)
        out["beam_std"] = beam_arr.std(1).astype(np.float32)
        out["beam_med"] = np.median(beam_arr, axis=1).astype(np.float32)
        out["beam_mean_d"] = (beam_arr.mean(1) - last_tvt).astype(np.float32)
        out["beam_med_d"] = (np.median(beam_arr, axis=1) - last_tvt).astype(np.float32)

        # tw_diff features (3 anchors × 11 offsets = 33 features)
        beam_ref_arr = bpaths["cons"]  # use the conservative beam as the beam-anchor
        td = tw_diff_features(
            gr.astype(np.float32),
            tw_tvt,
            tw_gr,
            anchor_last=last_tvt,
            anchor_beam=beam_ref_arr,
            anchor_sc=sc_raw,
        )
        for k, v in td.items():
            out[k] = v

        # Prefix RMSE (= how well visible GR matches typewell at known TVT)
        pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_kvis) ** 2)))
        out["pfx_rmse"] = np.full(n, pfx_rmse, dtype=np.float32)

        # Typewell summary stats
        out["tw_gr_mean"] = np.full(n, float(tw_gr.mean()), dtype=np.float32)
        out["tw_gr_std"] = np.full(n, float(tw_gr.std()), dtype=np.float32)
        out["tw_tvt_range"] = np.full(n, float(tw_tvt.max() - tw_tvt.min()), dtype=np.float32)

    return out


def add_features(
    h: pd.DataFrame,
    imputer: FormationPlaneKNN | None = None,
    typewells: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Per-well loop with optional formation imputation and typewell signals.

    Args:
        h: long DataFrame with all wells concatenated.
        imputer: trained FormationPlaneKNN. If None, formation features are skipped.
        typewells: dict[well_id -> (tw_tvt array, tw_gr array)]. If provided,
            typewell-aligned features (Beam/SC/xcorr/tw_diff/affine cal/detrend/pfx_rmse)
            are computed.
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

        typewell = typewells.get(wid) if typewells is not None else None

        feats = _per_well_features(
            sub, formation_imputed=formation_imputed, typewell=typewell
        )
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


def feature_columns(
    formation_avail: bool = True,
    typewell_avail: bool = False,
    wls_avail: bool = False,
) -> list[str]:
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
            if wls_avail:
                base += [
                    f"tvtFw_{fname}",
                    f"bww_{fname}",
                    f"tvtFw_{fname}_d",
                ]
    if typewell_avail:
        base += [
            "a_cal",
            "b_cal",
            "gr_detrend",
            "sc_trust",
            "sc_med_tvt",
            "sc_std_tvt",
            "sc_med_d",
            "xcorr_tvt",
            "xcorr_score",
            "xcorr_d",
            "xcorr_delta",
            "xcorr_corr",
            "xcorr_absdiff",
            "xcorr_tvt_off",
        ]
        # Multi-scale SC features (one block per scale; replaces single-scale sc_*)
        for hw in SC_HALF_WINDOWS:
            base += [f"sc{hw}_raw_tvt", f"sc{hw}_score", f"sc{hw}_d"]
        for tag in BEAM_TAGS:
            base += [f"beam_{tag}", f"beam_{tag}_d"]
        base += [
            "beam_mean",
            "beam_std",
            "beam_med",
            "beam_mean_d",
            "beam_med_d",
        ]
        # tw_diff: 11 + 11 + 11 = 33 features
        for o in (-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80):
            base.append(f"tda{int(o)}")
        for o in (-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40):
            base.append(f"tdbc{int(o)}")
        for o in (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30):
            base.append(f"tdsc{int(o)}")
        base += ["pfx_rmse", "tw_gr_mean", "tw_gr_std", "tw_tvt_range"]
    return base


# Default = formation only (exp002 baseline). Set typewell+wls True for exp003.
FEATURE_COLS = feature_columns(formation_avail=True)
FEATURE_COLS_V3 = feature_columns(
    formation_avail=True, typewell_avail=True, wls_avail=True
)


# ────────────────────────────────────────────────────────────────────────────
# deepest EDA D1 — b_well 67 cluster ID (= categorical wellbore zone fingerprint)
# task: kaggle-rogii-winning-candidates-cv-test-2026-05-12 (= 次タスク 優勝路手法 B)
# ────────────────────────────────────────────────────────────────────────────


def compute_b_well_cluster_id(
    well_ids: "Iterable[str]",  # type: ignore[name-defined]
    b_cluster_parquet: "Any" = None,  # type: ignore[name-defined]
    round_to: int = 3,
) -> "dict[str, int]":
    """Return ``{well_id: cluster_id_int}`` from deepest EDA b_ANCC_med.

    D1 (= deepest EDA finding §3.1): the 773 train wells collapse into
    **67 unique b_well clusters** at 0.001 ft rounding (= deepest EDA's
    own setting). Empirical sensitivity:

        round_to=0 (raw float):  87 unique  (= float accumulation noise)
        round_to=1 (0.1 ft):     55 unique  (= over-merged)
        round_to=2 (0.01 ft):    66 unique  (= 1 borderline pair merges)
        round_to=3 (0.001 ft):   67 unique  ← default, matches deepest EDA
        round_to=4 (0.0001 ft):  67 unique  (= same equivalence classes)

    Public top kernels treat b_well as continuous; this categorical view
    is the structural lift documented as "expected LB contribution
    -0.2 〜 -0.5 ft" in the design doc.

    Args:
        well_ids: train + test well_ids whose cluster IDs are needed.
        b_cluster_parquet: path to b-cluster-xy.parquet (defaults to the
            standard deepest EDA output location).
        round_to: decimals to round the raw b_ANCC_med value before using
            it as a cluster key. Default 3 reproduces the canonical 67
            cluster snapshot from deepest EDA §3.1. Use 2 for the
            quantization-aware variant (= 66 clusters under F-D3 TVT
            native 0.01 ft step).

    Returns:
        Mapping ``well_id`` → integer cluster_id (0..n_clusters-1). Stable
        across calls thanks to ``sorted`` over unique values. Missing wells
        get cluster_id = -1.
    """
    import numpy as _np
    import pandas as _pd
    from pathlib import Path as _Path

    if b_cluster_parquet is None:
        b_cluster_parquet = _Path("outputs/eda/deepest_eda/b-cluster-xy.parquet")
    b_cluster_parquet = _Path(b_cluster_parquet)
    if not b_cluster_parquet.exists():
        # graceful: caller can also fall back to per-well-stats.parquet b_well_mean
        return {str(wid): -1 for wid in well_ids}

    df = _pd.read_parquet(b_cluster_parquet)[["well_id", "b_ANCC_med"]]
    df = df.drop_duplicates(subset=["well_id"]).copy()
    if round_to > 0:
        df["b_key"] = df["b_ANCC_med"].round(round_to)
    else:
        df["b_key"] = df["b_ANCC_med"]
    # Stable integer encoding via sort
    unique_keys = sorted(df["b_key"].dropna().unique().tolist())
    key_to_int = {k: i for i, k in enumerate(unique_keys)}
    df["cluster_id"] = df["b_key"].map(key_to_int).fillna(-1).astype(int)
    lookup = dict(zip(df["well_id"].astype(str), df["cluster_id"]))
    return {str(wid): int(lookup.get(str(wid), -1)) for wid in well_ids}


def is_test_cluster(
    well_ids: "Iterable[str]",  # type: ignore[name-defined]
    test_wells: "tuple[str, ...]" = ("000d7d20", "00bbac68", "00e12e8b"),  # type: ignore[name-defined]
    b_cluster_parquet: "Any" = None,  # type: ignore[name-defined]
) -> "dict[str, int]":
    """Return ``{well_id: 0|1}`` where 1 marks wells in the same b_cluster
    as any of the 3 test wells.

    F1 finding: test wells fall in only 2 b_clusters (11373 × 2 wells,
    11855 × 1 well). 1 here means "this train well shares its b_cluster
    with at least one test well" — a useful binary feature for any
    per-well-stratified pipeline and for ablation of test-similar wells.
    """
    cluster_lookup = compute_b_well_cluster_id(
        list(set(list(well_ids) + list(test_wells))),
        b_cluster_parquet=b_cluster_parquet,
    )
    test_clusters = {cluster_lookup.get(str(w), -1) for w in test_wells}
    test_clusters.discard(-1)
    return {
        str(wid): int(cluster_lookup.get(str(wid), -1) in test_clusters)
        for wid in well_ids
    }


# ────────────────────────────────────────────────────────────────────────────
# deepest EDA D4 — b_jump_max_abs (= per-row 6-formation b std)
# ────────────────────────────────────────────────────────────────────────────


def compute_b_jump_max_abs(
    well_ids: "Iterable[str]",  # type: ignore[name-defined]
    high_order_parquet: "Any" = None,  # type: ignore[name-defined]
) -> "dict[str, float]":
    """Return ``{well_id: b_jump_max_abs}`` for D4 feature.

    D4 (= deepest EDA §1.4): per-well std of the 6 formation b values
    (= ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA). Higher std = larger
    formation-to-formation discrepancy = harder TVT extrapolation. Has
    Spearman 0.40 vs formula_rmse (= second-strongest predictor after
    visible_ratio) and is the natural MoE gate signal.

    Output: float (b_jump_max_abs) or NaN if the well is not in the parquet.
    """
    import pandas as _pd
    from pathlib import Path as _Path

    if high_order_parquet is None:
        high_order_parquet = _Path(
            "outputs/eda/deepest_eda/per-well-high-order.parquet"
        )
    high_order_parquet = _Path(high_order_parquet)
    if not high_order_parquet.exists():
        return {str(wid): float("nan") for wid in well_ids}

    df = _pd.read_parquet(high_order_parquet)
    if "b_jump_max_abs" not in df.columns:
        return {str(wid): float("nan") for wid in well_ids}
    df = df[["well_id", "b_jump_max_abs"]].drop_duplicates(subset=["well_id"])
    lookup = dict(
        zip(df["well_id"].astype(str), df["b_jump_max_abs"].astype(float))
    )
    return {str(wid): float(lookup.get(str(wid), float("nan"))) for wid in well_ids}


# ────────────────────────────────────────────────────────────────────────────
# deepest EDA D9 — 6 formation top per-well median
# ────────────────────────────────────────────────────────────────────────────


def compute_formation_top_medians(
    well_ids: "Iterable[str]",  # type: ignore[name-defined]
    formation_uniqueness_parquet: "Any" = None,  # type: ignore[name-defined]
) -> "dict[str, dict[str, float]]":
    """Return ``{well_id: {"top_ANCC_med": ..., ...}}`` for D9 feature group.

    D9 (= deepest EDA §3.7): each of the 6 formation top depths is unique
    per well at 0.01 ft precision. The per-well median over rows of each
    of (top_ANCC_med, top_ASTNU_med, top_ASTNL_med, top_EGFDU_med,
    top_EGFDL_med, top_BUDA_med) is a 6-dim fingerprint that public top
    kernels only use via ``bw_{fn}`` (= partial coverage).

    Output: nested dict ``{well_id: {col_name: float}}``. Missing wells
    or columns become NaN.
    """
    import pandas as _pd
    from pathlib import Path as _Path

    if formation_uniqueness_parquet is None:
        formation_uniqueness_parquet = _Path(
            "outputs/eda/deepest_eda/formation-uniqueness.parquet"
        )
    formation_uniqueness_parquet = _Path(formation_uniqueness_parquet)
    cols_wanted = (
        "top_ANCC_med", "top_ASTNU_med", "top_ASTNL_med",
        "top_EGFDU_med", "top_EGFDL_med", "top_BUDA_med",
    )
    if not formation_uniqueness_parquet.exists():
        return {str(wid): {c: float("nan") for c in cols_wanted} for wid in well_ids}

    df = _pd.read_parquet(formation_uniqueness_parquet)
    keep = ["well_id"] + [c for c in cols_wanted if c in df.columns]
    df = df[keep].drop_duplicates(subset=["well_id"])
    df["well_id"] = df["well_id"].astype(str)
    df.set_index("well_id", inplace=True)
    result: "dict[str, dict[str, float]]" = {}
    for wid in well_ids:
        wid_s = str(wid)
        if wid_s in df.index:
            row = df.loc[wid_s]
            result[wid_s] = {
                c: float(row[c]) if c in row.index else float("nan")
                for c in cols_wanted
            }
        else:
            result[wid_s] = {c: float("nan") for c in cols_wanted}
    return result
