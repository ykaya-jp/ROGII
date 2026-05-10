"""ROGII exp003 — Kaggle script submission (self-contained).

Stack on top of exp002 (residual target + 6-formation plane-fit) with:
  1. xcorr_tvt_offsets : tasmim-style TVT-offset sweep around last_known_TVT
                         → 4 features (delta, corr, absdiff, implied_tvt).
  2. multi_scale_sc    : romantamrazov-style 3-window Self-NCC (h8/h15/h25)
                         → 3 × (raw_tvt, score, d) + median/std aggregates.
  3. wls_b_well        : recent-weighted b_well per formation
                         → tvtFw_*, bww_*, tvtFw_*_d (18 features).
  4. gr_detrend_resid  : GR linear-MD trend residual (1 feature, well-specific).
  5. Beam search (5 configs), Self-NCC, tw_diff (33 anchor×offset features),
     affine GR cal, prefix RMSE, typewell summary stats.

Targets: TVT - last_known_TVT (residual), 5-fold GroupKFold by well.

Local CV (exp002): TVT hidden RMSE = 13.82 ft.
Local CV (exp003 target): < 12.5 ft (LB 12 切り 11 帯狙い).

Internet disabled. CPU runtime budget < 4 hr.

Refs:
- _research_kernels/tasmim__lb-11-068.../...py:236-263 (xcorr offsets)
- _research_kernels/romantamrazov__rogii-super-solution.../...py:181-209
  (wls_b_well + multi_scale_sc + gr_detrend_resid)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# -------------------- env --------------------


def _find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for p in candidates:
        if (p / "train").exists() and (p / "test").exists():
            return p
    if Path("/kaggle/input").exists():
        for sub in Path("/kaggle/input").iterdir():
            if (sub / "train").exists() and (sub / "test").exists():
                return sub
    return Path("data/raw")


DATA_DIR = _find_data_root()
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_local")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_DIR={DATA_DIR}", flush=True)
print(f"  train CSVs: {len(list(TRAIN_DIR.glob('*__horizontal_well.csv')))}", flush=True)
print(f"  test CSVs : {len(list(TEST_DIR.glob('*__horizontal_well.csv')))}", flush=True)
if Path("/kaggle/input").exists():
    print(f"  /kaggle/input contents: {os.listdir('/kaggle/input')}", flush=True)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
GR_ROLL_WINDOWS = (5, 21, 51, 101)
GR_DIFF_LAGS = (1, 5, 15, 30)
SC_HALF_WINDOWS = (8, 15, 25)
N_SPLITS = 5
SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 128,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.0,
    "verbosity": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": -1,
}
NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 100


# -------------------- imputer --------------------


class FormationPlaneKNN:
    def __init__(self, train_dir: Path, k: int = 10):
        self.k = k
        rows: list[dict] = []
        n_glob = 0
        n_loaded = 0
        n_with_formation = 0
        for p in sorted(Path(train_dir).glob("*__horizontal_well.csv")):
            n_glob += 1
            wid = p.stem.replace("__horizontal_well", "")
            try:
                cols_present = pd.read_csv(p, nrows=1).columns.tolist()
                if all(c in cols_present for c in FORMATIONS + ["X", "Y"]):
                    df = pd.read_csv(p, usecols=["X", "Y", *FORMATIONS]).dropna()
                    n_with_formation += 1
                else:
                    continue
            except Exception as e:
                print(f"   skip {p.name}: {e}", flush=True)
                continue
            n_loaded += 1
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_med"] = float(df[c].median())
            rows.append(row)
        print(
            f"   FormationPlaneKNN: glob={n_glob}, loaded={n_loaded}, with_formation={n_with_formation}, rows={len(rows)}",
            flush=True,
        )
        if len(rows) == 0:
            raise RuntimeError(
                f"FormationPlaneKNN: no training wells with formation columns found in {train_dir}."
            )
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"].to_numpy())}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(axis=0) < 1e-3, 1.0, xy.std(axis=0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_med" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None):
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q = xy_q / self.scale
        nf = min(self.k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid is not None and self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        order = np.argpartition(dist, min(self.k - 1, nf - 1), 1)[:, : self.k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        for i in range(3):
            A[:, i, i] += 1e-9
        fn = self.fa[ik]
        rhs = np.stack(
            [
                (wx[:, :, None] * fn).sum(1),
                (wy[:, :, None] * fn).sum(1),
                (w[:, :, None] * fn).sum(1),
            ],
            1,
        )
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (
            Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]
        ).astype(np.float32)
        nofit = ~vk.any(1)
        if nofit.any():
            pred[nofit] = self.fa.mean(0).astype(np.float32)
        return pred


# -------------------- tysig (typewell signal helpers) --------------------


def _nearest_index(sorted_arr: np.ndarray, v: float) -> int:
    i = int(np.searchsorted(sorted_arr, v, side="left"))
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    if i > 0 and abs(sorted_arr[i - 1] - v) <= abs(sorted_arr[i] - v):
        return i - 1
    return i


def _smooth_gr(gr: np.ndarray, fallback: float, radius: int = 2) -> np.ndarray:
    s = pd.Series(gr, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(np.float32)


def affine_gr_cal(kgr: np.ndarray, tw_at_kvis: np.ndarray, min_pts: int = 20):
    valid = np.isfinite(kgr) & np.isfinite(tw_at_kvis)
    if valid.sum() < min_pts or np.std(tw_at_kvis[valid]) < 1e-6:
        if valid.any():
            return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_kvis))
        return 1.0, 0.0
    a, b = np.polyfit(tw_at_kvis[valid], kgr[valid], 1)
    return float(a), float(b)


def self_ncc(kgr, ktvt, hgr, half_window=15, stride=3):
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


def multi_scale_sc(kgr, ktvt, hgr, half_windows=(8, 15, 25), stride=3):
    out = {}
    for hw in half_windows:
        sc_tvt, sc_score = self_ncc(kgr, ktvt, hgr, half_window=int(hw), stride=stride)
        out[f"h{hw}"] = (sc_tvt, sc_score)
    return out


def tw_diff_features(
    hgr,
    tw_tvt,
    tw_gr,
    anchor_last,
    anchor_beam=None,
    anchor_sc=None,
    offsets_last=(-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80),
    offsets_beam=(-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40),
    offsets_sc=(-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30),
):
    out = {}
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


def beam_search(gr_horiz, tw_tvt, tw_gr, start_tvt, beam_size=10, move_cost=20.0,
                emit_scale=144.0, smooth_radius=2):
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


DEFAULT_BEAM_CONFIGS = (
    ("cons", 10, 20.0, 144.0, 2),
    ("loose", 10, 8.0, 64.0, 2),
    ("vcons", 8, 35.0, 220.0, 1),
    ("sm5", 10, 14.0, 90.0, 5),
    ("vloose", 20, 4.0, 36.0, 3),
)
BEAM_TAGS = tuple(c[0] for c in DEFAULT_BEAM_CONFIGS)


def beam_search_multi(gr_horiz, tw_tvt, tw_gr, start_tvt, configs=DEFAULT_BEAM_CONFIGS):
    out = {}
    for tag, bs, mc, es, r in configs:
        out[tag] = beam_search(gr_horiz, tw_tvt, tw_gr, start_tvt, bs, mc, es, r)
    return out


def xcorr_tvt(kgr, ktvt, hgr, half_window=30):
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
    Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
    hp = np.pad(hg, half_window, mode="edge")
    H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
    Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
    corr = Hn @ Cn.T / win
    best = corr.argmax(1)
    score = corr.max(1).astype(np.float32)
    ctrs = np.clip(starts[best] + half_window, 0, nk - 1)
    return ktvt[ctrs].astype(np.float32), score


def xcorr_tvt_offsets(hgr, tw_tvt, tw_gr, last_known_tvt, window=30,
                     search_half=100.0, step=2.0):
    """Tasmim-style: find TVT offset around last_known_tvt that maximises Pearson
    correlation in local window. Vectorised matmul. Returns dict with 3 arrays.

    Reference: tasmim__lb-11-068.../...py:236-263.
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
    fb = float(np.nanmean(hgr)) if np.isfinite(hgr).any() else 0.0
    hgr_f = np.where(np.isfinite(hgr), hgr, fb).astype(np.float32)
    hp = np.pad(hgr_f, half, mode="edge")
    obs = np.lib.stride_tricks.sliding_window_view(hp, W).astype(np.float32)
    positions = last_known_tvt + offsets[:, None] + rel[None, :]
    tw_vals = np.interp(positions.ravel(), tw_tvt, tw_gr).reshape(K, W).astype(np.float32)
    obs_m = obs.mean(axis=1, keepdims=True)
    obs_s = obs.std(axis=1, keepdims=True) + 1e-6
    obs_n = (obs - obs_m) / obs_s
    tw_m = tw_vals.mean(axis=1, keepdims=True)
    tw_s = tw_vals.std(axis=1, keepdims=True) + 1e-6
    tw_n = (tw_vals - tw_m) / tw_s
    corr = (obs_n @ tw_n.T) / W
    best = corr.argmax(axis=1)
    best_corr = np.clip(corr[np.arange(n), best], -1.0, 1.0).astype(np.float32)
    best_off = offsets[best].astype(np.float32)
    abs_mat = np.abs(obs - tw_vals[best])
    best_abs = abs_mat.mean(axis=1).astype(np.float32)
    obs_valid = np.isfinite(hgr).astype(np.int32)
    valid_w = np.lib.stride_tricks.sliding_window_view(
        np.pad(obs_valid, half, mode="edge"), W
    ).sum(axis=1)
    low_quality = (valid_w < max(5, window // 4)) | (obs.std(axis=1) < 1e-3)
    if low_quality.any():
        mae = np.abs(obs[low_quality][:, None, :] - tw_vals[None, :, :]).mean(axis=2)
        bi = mae.argmin(axis=1)
        best_off[low_quality] = offsets[bi]
        best_corr[low_quality] = 0.0
        best_abs[low_quality] = mae[np.arange(len(bi)), bi].astype(np.float32)
    return {"xcorr_delta": best_off, "xcorr_corr": best_corr, "xcorr_absdiff": best_abs}


def gr_detrend_resid(md, gr):
    valid = np.isfinite(gr)
    if valid.sum() < 5:
        return np.zeros_like(gr, dtype=np.float32)
    a, b = np.polyfit(md[valid], gr[valid], 1)
    return (gr - (a * md + b)).astype(np.float32)


def wls_b_well(visible_tvt_input, visible_z, visible_f_imp, decay=0.02):
    """Recent-weighted b_well: tail points weighted higher.
    Reference: romantamrazov__rogii-super-solution.../...py:181-186.
    """
    n = len(visible_tvt_input)
    if n == 0:
        return 0.0
    vals = visible_tvt_input + visible_z - visible_f_imp
    dist = np.arange(n - 1, -1, -1, dtype=np.float32)
    w = np.exp(-decay * dist)
    return float(np.sum(vals * w) / np.sum(w))


# -------------------- features --------------------


def _planefit(visible_x, visible_y, visible_z, visible_t):
    if len(visible_t) < 4:
        return 0.0, 0.0, 0.0, float(visible_t.mean()) if len(visible_t) else 0.0
    A = np.column_stack([visible_x, visible_y, visible_z, np.ones(len(visible_t))])
    coef, *_ = np.linalg.lstsq(A, visible_t, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])


def _rolling_mean_std(arr, window):
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=1, center=True).mean().to_numpy()
    sd = s.rolling(window, min_periods=1, center=True).std().fillna(0.0).to_numpy()
    return m, sd


def _per_well_features(well_df: pd.DataFrame, formation_imputed, typewell):
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

    out = {
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

    if typewell is not None and visible_n:
        tw_tvt, tw_gr = typewell
        tw_tvt = np.asarray(tw_tvt, np.float32)
        tw_gr = np.asarray(tw_gr, np.float32)

        kgr = gr[visible_mask]
        ktvt = tvt_in[visible_mask]
        tw_at_kvis = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        a_cal, b_cal = affine_gr_cal(kgr, tw_at_kvis)
        out["a_cal"] = np.full(n, a_cal, dtype=np.float32)
        out["b_cal"] = np.full(n, b_cal, dtype=np.float32)
        out["gr_detrend"] = gr_detrend_resid(md, gr)

        # Multi-scale Self-NCC
        sc_results = multi_scale_sc(kgr, ktvt, gr, half_windows=SC_HALF_WINDOWS, stride=3)
        for hw in SC_HALF_WINDOWS:
            sc_tvt_hw, sc_score_hw = sc_results[f"h{hw}"]
            out[f"sc{hw}_raw_tvt"] = sc_tvt_hw
            out[f"sc{hw}_score"] = sc_score_hw
            out[f"sc{hw}_d"] = (sc_tvt_hw - last_tvt).astype(np.float32)
        sc_arr = np.stack([sc_results[f"h{hw}"][0] for hw in SC_HALF_WINDOWS], axis=1)
        sc_raw_med = np.median(sc_arr, axis=1).astype(np.float32)
        out["sc_med_tvt"] = sc_raw_med
        out["sc_std_tvt"] = sc_arr.std(axis=1).astype(np.float32)
        out["sc_med_d"] = (sc_raw_med - last_tvt).astype(np.float32)
        sc_trust = float(np.clip(visible_n / 200.0, 0.0, 0.6))
        out["sc_trust"] = np.full(n, sc_trust, dtype=np.float32)
        sc_raw = sc_results["h15"][0]

        xc_tvt, xc_score = xcorr_tvt(kgr, ktvt, gr, half_window=30)
        out["xcorr_tvt"] = xc_tvt
        out["xcorr_score"] = xc_score
        out["xcorr_d"] = (xc_tvt - last_tvt).astype(np.float32)

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
        out["xcorr_tvt_off"] = (last_tvt + xco["xcorr_delta"]).astype(np.float32)

        bpaths = beam_search_multi(gr, tw_tvt, tw_gr, start_tvt=last_tvt)
        beam_arr = np.stack([bpaths[t] for t in BEAM_TAGS], axis=1)
        for tag in BEAM_TAGS:
            out[f"beam_{tag}"] = bpaths[tag]
            out[f"beam_{tag}_d"] = (bpaths[tag] - last_tvt).astype(np.float32)
        out["beam_mean"] = beam_arr.mean(1).astype(np.float32)
        out["beam_std"] = beam_arr.std(1).astype(np.float32)
        out["beam_med"] = np.median(beam_arr, axis=1).astype(np.float32)
        out["beam_mean_d"] = (beam_arr.mean(1) - last_tvt).astype(np.float32)
        out["beam_med_d"] = (np.median(beam_arr, axis=1) - last_tvt).astype(np.float32)

        beam_ref_arr = bpaths["cons"]
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

        pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_kvis) ** 2)))
        out["pfx_rmse"] = np.full(n, pfx_rmse, dtype=np.float32)
        out["tw_gr_mean"] = np.full(n, float(tw_gr.mean()), dtype=np.float32)
        out["tw_gr_std"] = np.full(n, float(tw_gr.std()), dtype=np.float32)
        out["tw_tvt_range"] = np.full(n, float(tw_tvt.max() - tw_tvt.min()), dtype=np.float32)

    return out


def feature_columns():
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
    for fname in FORMATIONS:
        base += [
            f"tvtF_{fname}",
            f"tvtF50_{fname}",
            f"bw_{fname}",
            f"bw50_{fname}",
            f"tvtF_{fname}_d",
            f"f_imp_{fname}",
            f"tvtFw_{fname}",
            f"bww_{fname}",
            f"tvtFw_{fname}_d",
        ]
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
    for hw in SC_HALF_WINDOWS:
        base += [f"sc{hw}_raw_tvt", f"sc{hw}_score", f"sc{hw}_d"]
    for tag in BEAM_TAGS:
        base += [f"beam_{tag}", f"beam_{tag}_d"]
    base += ["beam_mean", "beam_std", "beam_med", "beam_mean_d", "beam_med_d"]
    for o in (-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80):
        base.append(f"tda{int(o)}")
    for o in (-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40):
        base.append(f"tdbc{int(o)}")
    for o in (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30):
        base.append(f"tdsc{int(o)}")
    base += ["pfx_rmse", "tw_gr_mean", "tw_gr_std", "tw_tvt_range"]
    return base


FEATURE_COLS = feature_columns()


# -------------------- IO --------------------


def list_wells(split_dir: Path):
    return sorted({p.stem.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv")})


def load_horizontal(split_dir: Path, well: str):
    df = pd.read_csv(split_dir / f"{well}__horizontal_well.csv")
    df.insert(0, "well", well)
    df["row_idx"] = np.arange(len(df), dtype=np.int32)
    return df


def load_typewell(split_dir: Path, well: str):
    p = split_dir / f"{well}__typewell.csv"
    if not p.exists():
        return None
    tw = pd.read_csv(p)
    if not {"TVT", "GR"}.issubset(tw.columns) or len(tw) < 2:
        return None
    return tw["TVT"].to_numpy(np.float32), tw["GR"].to_numpy(np.float32)


def load_split(split_dir: Path, imputer: FormationPlaneKNN, exclude_self: bool):
    wells = list_wells(split_dir)
    all_dfs: list[pd.DataFrame] = []
    print(f"   {len(wells)} wells in {split_dir.name}", flush=True)
    t0 = time.perf_counter()
    for i, wid in enumerate(wells):
        sub = load_horizontal(split_dir, wid)
        xy = sub[["X", "Y"]].to_numpy(np.float64)
        self_wid = wid if exclude_self else None
        formation_imputed = imputer.impute(xy, self_wid=self_wid)
        typewell = load_typewell(split_dir, wid)
        feats = _per_well_features(sub, formation_imputed=formation_imputed, typewell=typewell)
        feat_df = pd.DataFrame(feats)
        feat_df.insert(0, "row_idx", sub["row_idx"].to_numpy())
        feat_df.insert(0, "well", wid)
        feat_df["TVT_input"] = sub["TVT_input"].to_numpy()
        if "TVT" in sub.columns:
            feat_df["TVT"] = sub["TVT"].to_numpy()
        all_dfs.append(feat_df)
        if (i + 1) % 100 == 0:
            print(f"   ... {i + 1}/{len(wells)} wells ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"   total {time.perf_counter() - t0:.0f}s", flush=True)
    out = pd.concat(all_dfs, ignore_index=True, sort=False)
    return out


# -------------------- CV --------------------


def make_well_folds(well_ids: np.ndarray, n_splits: int, seed: int):
    rng = np.random.RandomState(seed)
    unique_wells = np.array(sorted(np.unique(well_ids)))
    rng.shuffle(unique_wells)
    fold_of = {w: i % n_splits for i, w in enumerate(unique_wells)}
    well_fold = np.array([fold_of[w] for w in well_ids])
    folds = []
    for k in range(n_splits):
        val = np.where(well_fold == k)[0]
        tr = np.where(well_fold != k)[0]
        folds.append((tr, val))
    return folds


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_hidden(a, b, mask):
    err = a[mask] - b[mask]
    return float(np.sqrt(np.mean(err**2)))


# -------------------- main --------------------


def main() -> None:
    print("=" * 60, flush=True)
    print("ROGII exp003 — tysig (xcorr + multi-scale SC + WLS) + LightGBM", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] build FormationPlaneKNN imputer", flush=True)
    t0 = time.perf_counter()
    imputer = FormationPlaneKNN(TRAIN_DIR, k=10)
    print(f"   {len(imputer.df)} centroids in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n[2/4] feature engineering — train (leave-one-out impute)", flush=True)
    train = load_split(TRAIN_DIR, imputer, exclude_self=True)
    print(f"   train shape: {train.shape}", flush=True)

    print("\n[2/4] feature engineering — test", flush=True)
    test = load_split(TEST_DIR, imputer, exclude_self=False)
    print(f"   test shape: {test.shape}", flush=True)

    feat_cols = [c for c in FEATURE_COLS if c in train.columns and c in test.columns]
    print(
        f"\n   features: {len(feat_cols)} (target: {len(FEATURE_COLS)}; missing: "
        f"{sorted(set(FEATURE_COLS) - set(feat_cols))[:5]}...)",
        flush=True,
    )

    last_known_train = train["last_known_TVT"].to_numpy(np.float32)
    y_residual = (train["TVT"].to_numpy(np.float32) - last_known_train).astype(np.float32)
    train_hidden = train["TVT_input"].isna().to_numpy()
    well_ids = train["well"].to_numpy()
    last_known_test = test["last_known_TVT"].to_numpy(np.float32)
    test_hidden = test["TVT_input"].isna().to_numpy()

    X_train = train[feat_cols].astype(np.float32)
    X_test = test[feat_cols].astype(np.float32)

    print(
        f"   residual std={y_residual.std():.2f}  |  "
        f"train hidden: {train_hidden.sum():,}/{len(X_train):,} ({train_hidden.mean():.1%})  |  "
        f"test hidden: {test_hidden.sum():,}",
        flush=True,
    )

    print("\n[3/4] 5-fold GroupKFold by well", flush=True)
    folds = make_well_folds(well_ids, n_splits=N_SPLITS, seed=SEED)

    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        t_fold = time.perf_counter()
        print(f"\n[fold {k + 1}/{N_SPLITS}]  train={len(tr_idx):,}  val={len(va_idx):,}", flush=True)
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], y_residual[tr_idx], free_raw_data=False)
        ds_va = lgb.Dataset(
            X_train.iloc[va_idx], y_residual[va_idx], reference=ds_tr, free_raw_data=False
        )
        booster = lgb.train(
            LGB_PARAMS,
            ds_tr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_va],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        oof[va_idx] = booster.predict(
            X_train.iloc[va_idx], num_iteration=booster.best_iteration
        )
        test_pred += booster.predict(X_test, num_iteration=booster.best_iteration) / N_SPLITS

        oof_tvt_va = last_known_train[va_idx] + oof[va_idx]
        true_tvt_va = train["TVT"].iloc[va_idx].to_numpy(np.float32)
        rmse_h = rmse_hidden(true_tvt_va, oof_tvt_va, train_hidden[va_idx])
        print(
            f"   best_iter={booster.best_iteration}  TVT hidden RMSE={rmse_h:.4f}  "
            f"({time.perf_counter() - t_fold:.0f}s)",
            flush=True,
        )

    oof_tvt = last_known_train + oof
    cv_hidden = rmse_hidden(train["TVT"].to_numpy(np.float32), oof_tvt, train_hidden)
    print(f"\n==> CV TVT hidden RMSE = {cv_hidden:.4f}", flush=True)

    print("\n[4/4] write submission", flush=True)
    test_tvt = last_known_test + test_pred
    sub = pd.DataFrame(
        {
            "id": [
                f"{w}_{i}"
                for w, i in zip(
                    test.loc[test_hidden, "well"].to_numpy(),
                    test.loc[test_hidden, "row_idx"].astype(int).to_numpy(),
                    strict=True,
                )
            ],
            "tvt": test_tvt[test_hidden],
        }
    )
    out_csv = OUT_DIR / "submission.csv"
    sub.to_csv(out_csv, index=False)
    print(f"   wrote {out_csv} ({len(sub):,} rows)", flush=True)
    print("\n✓ done", flush=True)


if __name__ == "__main__":
    main()
