"""Per-well feature builder for exp004 (Approach A self-reimplementation).

Inspired by romantamrazov/rogii-super-solution-lb-top-3 (Apache 2.0):
  build_well() function (lines 412-632).

Self-reimplemented for ROGII-exp004 with:
  - Explicit type hints on public API.
  - Numba beam + PF imported from `rogii.beam` / `rogii.pf`.
  - D1 cluster id column added (`d1_cluster_id`) per AC-13.
  - Self-exclusion enforced in imputer calls (= leak-free OOF CV).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.beam import BEAM_CONFIGS, beam_search
from rogii.features_d1 import UNKNOWN_CLUSTER_ID
from rogii.imputers import FORMATIONS, DenseANCCImputer, FormationPlaneKNN
from rogii.pf import run_pf_ancc, run_pf_z

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr: np.ndarray, tw_at_k: np.ndarray, min_pts: int = 20) -> tuple[float, float]:
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def wls_b_well(ktvt: np.ndarray, kz: np.ndarray, form_col: np.ndarray, decay: float = 0.02) -> float:
    n = len(ktvt)
    if n < 3:
        return float(np.median(ktvt + kz - form_col))
    w = np.exp(decay * np.arange(n))
    w = w / w.sum()
    return float(np.dot(w, ktvt + kz - form_col))


def multi_scale_sc(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    hws: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1] if len(ktvt) else 0.0, np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        ctrs = np.clip(sts[best] + hw, 0, nk - 1)
        out.append((ktvt[ctrs].astype(np.float32), score))
    return out


def gr_detrend_resid(gr_arr: np.ndarray, md_arr: np.ndarray) -> np.ndarray:
    m = np.isfinite(gr_arr) & np.isfinite(md_arr)
    if m.sum() < 5:
        return gr_arr.copy()
    slope = robust_slope(md_arr[m], gr_arr[m])
    return (gr_arr - slope * md_arr).astype(np.float32)


def build_well(
    hw_path: Path,
    tw_path: Path,
    is_train: bool,
    fi: FormationPlaneKNN,
    di: DenseANCCImputer,
    cluster_map: dict[str, int],
    use_self_exclusion: bool = True,
) -> pd.DataFrame | None:
    """Build feature dataframe for one well.

    Returns None if the well has insufficient data (< 10 known rows or no eval).
    """
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and hw["TVT"].isna().all():
        return None

    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    # ---- Particle Filters ----
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr, rng_seed=42)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr, rng_seed=42)
    pf_use = pf_a
    std_use = std_a
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    # ---- GR helpers ----
    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0] :].to_numpy(np.float32)
    kgr = gr_full.iloc[: len(kn)].to_numpy(np.float32)
    hmd_s = ev["MD"].to_numpy(np.float32)
    gr_arr = gr_full.values.astype(np.float32)
    md_arr = hw["MD"].values.astype(np.float32)
    gr_detr = gr_detrend_resid(gr_arr, md_arr)
    hgr_detr = gr_detr[ev.index]

    # ---- 7 beam configs ----
    bpaths: dict[str, np.ndarray] = {}
    for bs, mc, es, r, tag in BEAM_CONFIGS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    # ---- Multi-scale NCC self-corr ----
    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res = multi_scale_sc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]
    sc15, sc15s = sc_res[1]
    sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc15

    # ---- Affine calibration ----
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)

    # ---- Prefix stats ----
    kmd = kn["MD"].to_numpy(np.float32)
    kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt)
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)
    pfx_gr_slope = robust_slope(kmd, kgr)

    # ---- Spatial imputation (with self-exclusion) ----
    swid = wid if (is_train and use_self_exclusion) else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = fi.impute(xy_ev, self_wid=swid)
    form_kn, _ = fi.impute(xy_kn, self_wid=swid)

    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    # ---- Per-formation TVT + WLS b_well ----
    tvt_fs: dict[str, np.ndarray | np.float32] = {}
    form_rmse: dict[str, float] = {}
    form_list: list[np.ndarray] = []
    for fi2, fn in enumerate(FORMATIONS):
        b_v = ktvt + z_kn - form_kn[:, fi2]
        b_all = float(np.median(b_v))
        b_wls = wls_b_well(ktvt, z_kn, form_kn[:, fi2])
        b_50 = float(np.median(b_v[-50:])) if len(b_v) >= 5 else b_all
        tvt_f = (-z_ev + form_ev[:, fi2] + b_all).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_fs[fn] = tvt_f
        tvt_fs[fn + "_wls"] = tvt_fw
        tvt_fs[f"bw_{fn}"] = np.float32(b_all)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_50)
        tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_all)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    # ---- Dense ANCC + WLS ----
    d_ancc, d_std, d_dist = di.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = di.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    b_d = float(np.median(b_vd))
    b_d_wls = wls_b_well(ktvt, z_kn, d_kn)
    b_d50 = float(np.median(b_vd[-50:])) if len(b_vd) >= 5 else b_d
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_d_wls).astype(np.float32)
    tvt_d50 = (-z_ev + d_ancc + b_d50).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))

    # ---- Inter-signal consensus ----
    all_sigs = [pf_use, *(p for p in bpaths.values()), sc8, sc15, sc25, tvt_fs["ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    # ---- GR rolling features ----
    gr_s = pd.Series(gr_full.values)
    rolls: dict[str, np.ndarray] = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"grm{w}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.0)).iloc[
        ev.index
    ].values.astype(np.float32)

    # ---- Slope baselines ----
    md_since = hmd_s - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    # ---- Trajectory ----
    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    def sc(v: float) -> np.ndarray:
        return np.full(nh, np.float32(v), np.float32)

    # ---- D1 cluster id (AC-13) ----
    d1_cluster = np.full(nh, cluster_map.get(wid, UNKNOWN_CLUSTER_ID), dtype=np.int32)

    feats: dict[str, np.ndarray | list[str] | str] = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "d1_cluster_id": d1_cluster,
        "pf_ancc": pf_use,
        "pf_ancc_std": std_use,
        "pf_ancc_d": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_d": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(
            np.float32
        ),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32),
        "sc8_score": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32),
        "sc15_score": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32),
        "sc25_score": sc25s,
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust),
        "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "signal_std": sig_std,
        "signal_mean_d": sig_mean,
        **{f"tvtF_{fn}_d": (tvt_fs[fn] - last_tvt).astype(np.float32) for fn in FORMATIONS},
        **{f"tvtFw_{fn}_d": (tvt_fs[fn + "_wls"] - last_tvt).astype(np.float32) for fn in FORMATIONS},
        **{f"bw_{fn}": tvt_fs[f"bw_{fn}"] for fn in FORMATIONS},
        **{f"bww_{fn}": tvt_fs[f"bww_{fn}"] for fn in FORMATIONS},
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d,
        "form_std_d": form_std_d,
        "form_rng_d": form_rng_d,
        "knn_d": knn_d,
        "dense_ancc": d_ancc,
        "dense_std": d_std,
        "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_d50_d": (tvt_d50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse),
        "dense_bias": sc(d_bias),
        "pf_vs_form": (pf_use - tvt_fs["ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "form_vs_dense": (tvt_fs["ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_form": (bpaths["cons"] - tvt_fs["ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc15 - bpaths["cons"]).astype(np.float32),
        "cal_a": sc(a_cal),
        "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse),
        "known_len": sc(len(kn)),
        "eval_len": sc(nh),
        "slp_all": sc(slp_all),
        "slp_50": sc(slp_50),
        "slp_z": sc(slp_z),
        "pfx_gr_slope": sc(pfx_gr_slope),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "ktvt_range": sc(float(np.ptp(ktvt))),
        "ktvt_std": sc(float(ktvt.std())),
        "md_since": md_since,
        "frac": frac,
        "frac2": frac ** 2,
        "sqrt_frac": np.sqrt(frac),
        "z": z_ev,
        "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
        "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
        "dz": (z_ev - float(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(
            np.float32
        ),
        "dzdmd": dzdmd,
        "dxdmd": dxdmd,
        "dydmd": dydmd,
        "gr": hgr,
        "gr_d1": gr_d1,
        "gr_d2": gr_d2,
        "gr_env": gr_env,
        "gr_nrg": gr_nrg,
        "gr_detr": hgr_detr,
        "gr_vs_tw": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_vs_slp": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{
            f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32)
            for o in BEAM_OFFS
        },
        **{f"tdsc{int(o)}": hgr - np.interp(sc15 + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))),
        "tw_gr_mean": sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items():
        feats[k] = v

    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        result["target"] = ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt)
    return result
