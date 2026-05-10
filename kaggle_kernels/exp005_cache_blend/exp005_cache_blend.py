#!/usr/bin/env python
# coding: utf-8

# # ROGII exp005 v3 — Cache + karnakbaev artifacts blend + Edge S + Edge R (LB target: ~9.95)
#
# Self-contained Kaggle script that reuses pre-trained model artefacts published in
#   `karnakbaevarthur/rogii-code-helper-dataset` (apache-2.0 license)
# and the matching feature builder published in
#   `karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline` (public LB 10.784).
#
# What this script does:
#   1. Build test features from live `/kaggle/input/.../test/` directory (= safe for private rerun).
#   2. Load LGB×3 + XGB + CatBoost boosters from the artefacts dataset.
#   3. Load Ridge meta and `ensemble_weights.json` (Nelder-Mead) and `postproc_params.json`.
#   4. Apply Nelder-Mead blend (= author's final form) → fade-in postproc → SG smooth → submission.csv.
#   5. **Edge S: dTVT 0.01 grid round-to-grid**, inspired by hengck23 discussion 697431.
#      Submission の最終 tvt を np.round(., 2) で 0.01 ft grid に snap。
#      773/773 wells で dTVT min_step = 0.01 ft が 100% 普遍 (中央実測)。
#   6. **Edge R: test-time online learning (= continued training on test visible TVT_input),
#      inspired by discussion 698002**. test wells の visible region の末尾 K 行を
#      擬似 hidden に変換、 `is_train=True` 経路で feature+target を生成して
#      `lgb.train(..., init_model=base, num_boost_round=200)` で 5 base 各々を fine-tune。
#      online predict と既存 NM blend を w_R=0.5 で重ね合わせ。
#      公開実測: online 10.953 vs no-online 11.323 = -0.370 ft (topic 698002)。
#      leak guard: hidden TVT に絶対触れない (= visible TVT_input のみ参照)。
#
# Source attribution / license:
#   - Pipeline: based on karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline
#     (notebook is public on Kaggle; code reused here under fair-use Kaggle-style sharing.)
#   - Artefacts: `karnakbaevarthur/rogii-code-helper-dataset` (apache-2.0).
#   - Original solution itself credits romantamrazov/rogii-super-solution-lb-top-3 ideas.
#
# This file is for the team's exp005 entry only. We modified path resolution to
# tolerate both `/kaggle/input/rogii-code-helper-dataset/...` (= standard Kaggle attach)
# and the legacy `/kaggle/input/datasets/karnakbaevarthur/rogii-code-helper-dataset/...`.

# ## 0. Imports & Config

# In[ ]:


from __future__ import annotations

import gc
import json
import logging
import multiprocessing
import pickle
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─── Execution Config ────────────────────────────────────────────────────────
# MODE: "train" | "infer" | "cv" | "features_only" | "ensemble_only"
MODE = "infer"

# Active base models
ACTIVE_MODELS = ["lgb0", "lgb1", "lgb2", "xgb", "cb"]

# Debug flags
# Allow env override for local smoke runs (= 5 wells in ~1 min vs 776 wells in ~10 min)
import os as _os_dbg
_ENV_DEBUG_MAX_WELLS = _os_dbg.environ.get("ROGII_DEBUG_MAX_WELLS")
DEBUG_MAX_WELLS    = int(_ENV_DEBUG_MAX_WELLS) if _ENV_DEBUG_MAX_WELLS else None  # set to e.g. 10 for fast iteration
DEBUG_ONE_FOLD     = False
DEBUG_INSPECT_PF   = False
DEBUG_INSPECT_BEAM = False

# ─── Paths ───────────────────────────────────────────────────────────────────
# Env override for local smoke runs (= takes precedence over Kaggle paths).
import os as _os
_ENV_DATA_DIR     = _os.environ.get("ROGII_DATA_DIR")
_ENV_ARTEFACT_DIR = _os.environ.get("ROGII_ARTEFACT_DIR")
_ENV_OUTPUT_DIR   = _os.environ.get("ROGII_OUTPUT_DIR")

_KAGGLE_CANDIDATES = [
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
]
if _ENV_DATA_DIR and Path(_ENV_DATA_DIR).exists():
    DATA_DIR = Path(_ENV_DATA_DIR).resolve()
else:
    DATA_DIR = next((p for p in _KAGGLE_CANDIDATES if (p / "test").exists()),
                    Path("../../data").resolve())
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

if MODE in ("train", "cv", "features_only", "ensemble_only"):
    ARTEFACT_DIR = Path("/kaggle/working/artefacts")
elif _ENV_ARTEFACT_DIR and Path(_ENV_ARTEFACT_DIR).exists():
    ARTEFACT_DIR = Path(_ENV_ARTEFACT_DIR).resolve()
else:
    # Resolve artefacts dir robustly: Kaggle dataset attach standard puts the
    # dataset under `/kaggle/input/<slug>/` (so `artefacts/` lives directly
    # under that). The original kernel hard-codes a legacy nested path.
    _ARTEFACT_CANDIDATES = [
        Path("/kaggle/input/rogii-code-helper-dataset/artefacts"),
        Path("/kaggle/input/datasets/karnakbaevarthur/rogii-code-helper-dataset/artefacts"),
        Path("/kaggle/input/karnakbaevarthur/rogii-code-helper-dataset/artefacts"),
    ]
    ARTEFACT_DIR = next((p for p in _ARTEFACT_CANDIDATES if p.exists()),
                       _ARTEFACT_CANDIDATES[0])
    # Last-ditch search: walk /kaggle/input for a folder containing final_lgb_lgb2.txt
    if not (ARTEFACT_DIR / "final_lgb_lgb2.txt").exists():
        if Path("/kaggle/input").exists():
            for sub in Path("/kaggle/input").rglob("final_lgb_lgb2.txt"):
                ARTEFACT_DIR = sub.parent
                break

OUTPUT_DIR = Path(_ENV_OUTPUT_DIR).resolve() if _ENV_OUTPUT_DIR else Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) if not str(OUTPUT_DIR).startswith("/kaggle") else None
# Do NOT mkdir on a read-only Kaggle input path (only mkdir for /kaggle/working artefacts).
if str(ARTEFACT_DIR).startswith("/kaggle/working") or str(ARTEFACT_DIR).startswith("/tmp"):
    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED     = 42
N_SPLITS = 5
np.random.seed(SEED)

# ─── Parallel ────────────────────────────────────────────────────────────────
NCPU = min(4, multiprocessing.cpu_count())

# ─── Formation columns (geological surfaces) ─────────────────────────────────
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K    = 10   # plane-fit KNN neighbors
DENSE_SPW  = 60   # dense samples per well
DENSE_K    = 20

# ─── Beam configs: (beam_size, move_cost, emit_scale, smooth_r, tag) ─────────
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10,  8.0,  64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
]

# ─── TVT offset anchors (3 families) ─────────────────────────────────────────
ANCH_OFFS = np.array([-120,-80,-40,-20,-10,-5,0,5,10,20,40,80,120], dtype=np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], dtype=np.float32)
SC_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],  dtype=np.float32)

# ─── Particle Filter hyperparameters ─────────────────────────────────────────
PF_N               = 500
ANCC_N             = 500
PF_MOM             = 0.993
PF_VN              = 0.005
PF_PN              = 0.01
PF_GR_SIG_MIN      = 10.0
PF_GR_SIG_MAX      = 60.0
PF_GR_SIG_DEF      = 30.0
PF_INIT_V_STD      = 0.02
PF_INIT_SPR        = 0.5
PF_RESAMP          = 0.5
PF_ROUGH_P         = 0.2
PF_ROUGH_V         = 0.003
PF_GR_WIN          = 5
PF_GR_WT           = 0.3
ANCC_ALPHA         = 0.998
ANCC_RN            = 0.002
ANCC_PN            = 0.005
ANCC_IR            = 0.01
ANCC_IS            = 0.3
ANCC_RP            = 0.1
ANCC_RR            = 0.001

# ─── GPU detection ───────────────────────────────────────────────────────────
def _has_gpu() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=3
        ).returncode == 0
    except Exception:
        return False

_GPU = _has_gpu()
print(f"GPU: {_GPU}  | CPUs: {NCPU}  | MODE: {MODE}")

# ─── Model hyperparameters ───────────────────────────────────────────────────
LGB_BASE = dict(
    boosting_type    = "gbdt",
    learning_rate    = 0.04,
    num_leaves       = 127,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_lambda       = 5.0,
    reg_alpha        = 0.1,
    objective        = "regression",
    verbose          = -1,
    n_jobs           = -1,
    n_estimators     = 5000,
)
if _GPU:
    LGB_BASE["device_type"] = "gpu"
    LGB_BASE["gpu_use_dp"]  = False
    LGB_BASE["max_bin"]     = 255

LGB_SEEDS = [42, 7, 123]

XGB_PARAMS = dict(
    n_estimators          = 5000,
    learning_rate         = 0.04,
    max_depth             = 7,
    min_child_weight      = 20,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.1,
    reg_lambda            = 5.0,
    random_state          = SEED,
    tree_method           = "hist",
    device                = "cuda" if _GPU else "cpu",
    eval_metric           = "rmse",
    early_stopping_rounds = 150,
)

CB_PARAMS = dict(
    iterations            = 5000,
    learning_rate         = 0.04,
    depth                 = 8,
    l2_leaf_reg           = 3.0,
    min_data_in_leaf      = 20,
    loss_function         = "RMSE",
    random_seed           = SEED,
    task_type             = "GPU" if _GPU else "CPU",
    od_type               = "Iter",
    od_wait               = 150,
    verbose               = 0,
)
if _GPU and "GPU" in CB_PARAMS.get("task_type", ""):
    CB_PARAMS["devices"] = "0:1"

EARLY_STOP_ROUNDS = 150
LOG_EVERY         = 500
FINAL_ITER_SCALE  = 1.10

# ─── Edge R knobs (test-time online learning, exp005 v3 新規) ──────────────
# Inspired by Kaggle discussion topic 698002 (online 10.953 vs no-online 11.323 = -0.370 ft).
# test wells の visible region の末尾 K 行を擬似 hidden に変換し、
# `is_train=True` 経路で feature+target を生成 → LGB 5 base に init_model continued training。
EDGE_R_ENABLE          = True
EDGE_R_VISIBLE_TAIL_K  = 100   # visible 内の擬似 hidden サイズ (= 平均 hidden length ~110 と同等)
EDGE_R_MIN_VISIBLE_ROWS = 30    # visible 行数 30 未満の well は skip (= 教師信号不十分)
EDGE_R_NUM_BOOST       = 200    # continued training round 数 (= topic 698002 で 100-300 推奨)
EDGE_R_LR_MUL          = 0.5    # continued training の lr × 0.5 (= 過適応抑制)
EDGE_R_BLEND_W         = 0.5    # online 重ね合わせ weight (= exp005 v3 は固定、 grid なし)

# Meta columns excluded from ML features
_META_COLS = {"well", "id", "prediction_id", "target", "group_id", "row_idx"}

print(f"DATA_DIR={DATA_DIR}")
print(f"ARTEFACT_DIR={ARTEFACT_DIR}")
print(f"Beam configs: {len(BEAMS)} | ANCH_OFFS: {len(ANCH_OFFS)} | BEAM_OFFS: {len(BEAM_OFFS)} | SC_OFFS: {len(SC_OFFS)}")


# ## 1. Artifact Manager

# In[ ]:


def _json_default(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


class ArtifactManager:
    """Single point of control for all pipeline I/O."""

    TRAIN_DF           = "train_df"
    TEST_DF            = "test_df"
    OOF_PREDICTIONS    = "oof_predictions"
    BEST_ITERS         = "best_iters"
    ENSEMBLE_WEIGHTS   = "ensemble_weights"
    FEATURES_LIST      = "features"
    FOLD_METRICS       = "fold_metrics"
    FEATURE_IMPORTANCE = "feature_importance"
    RIDGE_META         = "ridge_meta"

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj: Any, name: str) -> Path:
        path = self.dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        print(f"  ✔ saved  {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
        return path

    def load(self, name: str) -> Any:
        path = self.dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_json(self, obj: Any, name: str) -> Path:
        path = self.dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        print(f"  ✔ saved  {path.name}")
        return path

    def load_json(self, name: str) -> Any:
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path) as f:
            return json.load(f)

    def save_df(self, df: pd.DataFrame, name: str) -> Path:
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✔ saved  {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
        return path

    def load_df(self, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"DataFrame not found: {path}")
        return pd.read_parquet(path)

    def exists(self, name: str, ext: str = ".pkl") -> bool:
        return (self.dir / f"{name}{ext}").exists()

    def list_artefacts(self):
        for p in sorted(self.dir.iterdir()):
            print(f"  {p.name:45s}  {p.stat().st_size/1e6:.2f} MB")


am = ArtifactManager(ARTEFACT_DIR)


# ## 2. Logging & Math Utilities

# In[ ]:


_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.DEBUG)
        log.propagate = False
    return log

@contextmanager
def timer(log, label: str):
    log.info(f"▶  {label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info(f"✔  {label} — {time.perf_counter()-t0:.2f}s")

def section(log, title: str):
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)

log = get_logger("pipeline")

# Math helpers
def nn_idx(arr: np.ndarray, v: float) -> int:
    """Binary-search nearest index in sorted array."""
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def robust_slope(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2: return 0.0
    if np.std(x[m]) < 1e-6: return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def recent_mean_diff(values: np.ndarray, window: int) -> float:
    vals = values[-(window + 1):]
    return float(np.diff(vals).mean()) if len(vals) >= 2 else 0.0


def systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    cum = np.cumsum(weights)
    pos = (np.arange(n) + np.random.uniform()) / n
    return np.searchsorted(cum, pos)


def fill_and_smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    s = pd.Series(values, dtype="float32").interpolate(
        limit_direction="both"
    ).fillna(fallback)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=np.float32)


print("Utilities loaded ✓")


# ## 3. GR Feature Engineering
# #
# GR leads (gr_lead*) are LEGAL: GR is fully observed across the entire
# borehole; only TVT_input is withheld in the hidden section.
# Leads look ahead in depth, not ahead in the target.

# In[ ]:


def compute_gr_features(gr_raw: pd.Series, fallback: float) -> dict:
    """Full-well GR feature dict; all Series aligned to original index."""
    def roll(s, w, fn="mean"):
        return getattr(s.rolling(w, center=True, min_periods=1), fn)()

    gf   = gr_raw.astype("float32").interpolate(limit_direction="both").fillna(fallback)
    grad = gf.diff().fillna(0.0)

    mn5  = roll(gf, 5,  "min");  mx5  = roll(gf, 5,  "max")
    mn21 = roll(gf, 21, "min");  mx21 = roll(gf, 21, "max")

    return dict(
        gr_filled = gf,
        roll3     = roll(gf, 3),
        roll5     = roll(gf, 5),
        roll11    = roll(gf, 11),
        roll21    = roll(gf, 21),
        roll51    = roll(gf, 51),
        roll101   = roll(gf, 101),
        roll151   = roll(gf, 151),
        std5      = roll(gf, 5,  "std").fillna(0),
        std21     = roll(gf, 21, "std").fillna(0),
        min5      = mn5,  max5  = mx5,  range5  = mx5  - mn5,
        min21     = mn21, max21 = mx21, range21 = mx21 - mn21,
        grad      = grad,
        grad2     = grad.diff().fillna(0.0),
        lag1      = gf.shift(1).bfill(),
        lag5      = gf.shift(5).bfill(),
        lag15     = gf.shift(15).bfill(),
        lag30     = gf.shift(30).bfill(),
        lead1     = gf.shift(-1).ffill(),    # legal
        lead5     = gf.shift(-5).ffill(),    # legal
        lead15    = gf.shift(-15).ffill(),   # legal
        lead30    = gf.shift(-30).ffill(),   # legal
        cumsum    = gf.cumsum(),
    )

print("GR feature functions loaded ✓")


# ## 4. Beam Search (5 configs)
# #
# Five priors on geological smoothness:
# vcons (tight), cons (balanced), sm5 (smooth radius=5), loose, vloose.
# Disagreement between beams = path uncertainty → key ML signal.

# In[ ]:


def beam_search(
    gr_h:       np.ndarray,
    tw_tvt:     np.ndarray,
    tw_gr:      np.ndarray,
    start_tvt:  float,
    bs:         int   = 10,
    mc:         float = 20.0,
    es:         float = 144.0,
    r:          int   = 2,
) -> np.ndarray:
    """Run beam search; returns absolute TVT path aligned to gr_h length."""
    tw_tvt = np.asarray(tw_tvt, np.float32)
    tw_gr  = np.asarray(tw_gr,  np.float32)
    T  = len(tw_tvt)
    fb = float(np.nanmean(tw_gr))
    sg = fill_and_smooth_gr(gr_h, fb, r)
    si = nn_idx(tw_tvt, start_tvt)
    ns = len(sg)

    bi = np.full(bs, si, np.int32)
    bc = np.zeros(bs, np.float64)
    bps = np.empty((ns, bs), np.int32)
    bpb = np.empty((ns, bs), np.int32)

    for s, gv in enumerate(sg):
        ci = np.clip(bi[:, None] + np.array([-1, 0, 1]), 0, T - 1)
        em = (gv - tw_gr[ci]) ** 2 / es
        mv = mc * np.array([1, 0, 1])[None, :]
        cc = bc[:, None] + em + mv

        fi = ci.ravel(); fc = cc.ravel(); fp = np.repeat(np.arange(bs), 3)
        ord_ = np.argsort(fc, kind="stable")
        kept = []; seen = set()
        for o in ord_:
            t = int(fi[o])
            if t not in seen:
                seen.add(t); kept.append(o)
            if len(kept) == bs:
                break
        while len(kept) < bs:
            kept.append(kept[-1])
        kept = np.array(kept, np.int32)

        bps[s] = fp[kept]; bpb[s] = fi[kept]
        bi = fi[kept].astype(np.int32); bc = fc[kept]

    path = np.empty(ns, np.int32)
    cb   = int(np.argmin(bc))
    for s in range(ns - 1, -1, -1):
        path[s] = bpb[s, cb]; cb = bps[s, cb]
    return tw_tvt[path]


def compute_beam_features(
    hgr_full:       np.ndarray,
    tw_tvt:         np.ndarray,
    tw_gr:          np.ndarray,
    last_known_tvt: float,
    sel_local:      np.ndarray,
) -> dict:
    """Run all 5 beam configs; return feature dict (indexed by sel_local)."""
    paths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        paths[tag] = beam_search(hgr_full, tw_tvt, tw_gr, last_known_tvt,
                                 bs, mc, es, r)

    lkt   = np.float32(last_known_tvt)
    stack = np.stack([p for p in paths.values()], axis=1)  # (n_hidden, 5)
    beam_ref = (paths["cons"] + paths["sm5"]) / 2.0

    out = {}
    for tag, p in paths.items():
        out[f"beam_{tag}_d"] = (p - lkt).astype(np.float32)[sel_local]

    out["beam_mean_d"]   = (stack.mean(axis=1) - lkt).astype(np.float32)[sel_local]
    out["beam_std_d"]    = stack.std(axis=1).astype(np.float32)[sel_local]
    out["beam_med_d"]    = (np.median(stack, axis=1) - lkt).astype(np.float32)[sel_local]
    out["beam_spread_d"] = (paths["vloose"] - paths["vcons"]).astype(np.float32)[sel_local]
    out["beam_gap_d"]    = (paths["loose"]  - paths["cons"]).astype(np.float32)[sel_local]

    out["tw_gr_at_cons"]  = np.interp(paths["cons"],  tw_tvt, tw_gr).astype(np.float32)[sel_local]
    out["tw_gr_at_loose"] = np.interp(paths["loose"], tw_tvt, tw_gr).astype(np.float32)[sel_local]
    out["gr_minus_tw_cons"]  = (hgr_full - np.interp(paths["cons"],  tw_tvt, tw_gr)).astype(np.float32)[sel_local]
    out["gr_minus_tw_loose"] = (hgr_full - np.interp(paths["loose"], tw_tvt, tw_gr)).astype(np.float32)[sel_local]

    # beam_ref for tw_diff anchor (returned as full array for offset features)
    out["_beam_ref_full"] = beam_ref  # NOT indexed – used for offset features

    return out

print("Beam search functions loaded ✓")


# ## 5. Particle Filters (Z-velocity + ANCC)

# In[ ]:


def _cal_gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return PF_GR_SIG_DEF
    ex = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    return float(np.clip(np.std(kn["GR"].values - ex), PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _z_beta(hw: pd.DataFrame) -> Tuple[float, float, float]:
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 30:
        return -1.0, 0.0, 0.1
    dz   = np.diff(kn["Z"].values); dtvt = np.diff(kn["TVT_input"].values)
    dmd  = np.diff(kn["MD"].values); m    = dmd > 0
    if m.sum() < 10:
        return -1.0, 0.0, 0.1
    vz = dz[m] / dmd[m]; vt = dtvt[m] / dmd[m]
    c, _, _, _ = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
    return float(c[0]), float(c[1]), max(float(np.std(vt - (c[0]*vz + c[1]))), 0.001)


def _init_v(hw: pd.DataFrame) -> float:
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 10:
        return 0.0
    tail = kn.tail(20)
    dt = np.diff(tail["TVT_input"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    return 0.0 if m.sum() < 3 else float(np.median(dt[m] / dm[m]))


def run_pf_z(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,
             N: int = PF_N) -> Tuple[np.ndarray, np.ndarray]:
    """Z-velocity particle filter. Returns (pred_tvts, pred_stds) for hidden rows."""
    tw_s   = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values
    tf_p   = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    tf_s   = interp1d(tw_tvt, tw_s,  bounds_error=False, fill_value=(tw_s[0],  tw_s[-1]))
    tmin, tmax = tw_tvt.min(), tw_tvt.max()
    gs     = _cal_gr_sigma(hw, tw_tvt, tw_gr)
    beta, icpt, zsig = _z_beta(hw)
    kn     = hw[hw["TVT_input"].notna()]
    ev     = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    gr_sm  = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pos    = float(kn["TVT_input"].iloc[-1]) + np.random.normal(0, PF_INIT_SPR, N)
    vel    = _init_v(hw) + np.random.normal(0, PF_INIT_V_STD, N)
    w      = np.ones(N) / N
    md_v   = ev["MD"].values; gr_v = ev["GR"].values; z_v = ev["Z"].values
    pm     = float(kn["MD"].iloc[-1]); pz = float(kn["Z"].iloc[-1])
    pts    = np.empty(len(ev)); std = np.empty(len(ev))

    for i, idx in enumerate(ev.index):
        dm  = max(md_v[i] - pm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve  = beta * dzd + icpt
        vel = PF_MOM * vel + np.random.normal(0, PF_VN, N)
        pos = pos + vel * dm + np.random.normal(0, PF_PN, N)
        pos = np.clip(pos, tmin - 50, tmax + 50)

        if not np.isnan(gr_v[i]):
            ep  = tf_p(pos); lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)
            gs_sm = gr_sm.iloc[hw.index.get_loc(idx)]
            if not np.isnan(gs_sm):
                es_ = tf_s(pos)
                ls  = np.exp(-0.5 * ((gs_sm - es_) / (gs * 1.5)) ** 2)
                lk  = (1 - PF_GR_WT) * lp + PF_GR_WT * ls
            else:
                lk = lp
            lk = np.maximum(lk, 1e-300); w *= lk
            ws = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)

        zs  = max(zsig * 2.0, 0.005)
        lz  = np.exp(-0.5 * ((vel - ve) / zs) ** 2)
        lz  = np.maximum(lz, 1e-300); w *= lz
        ws  = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)

        ne = 1.0 / np.sum(w ** 2)
        if ne < PF_RESAMP * N:
            ix  = systematic_resample(w, N)
            pos = pos[ix] + np.random.normal(0, PF_ROUGH_P, N)
            vel = vel[ix] + np.random.normal(0, PF_ROUGH_V, N)
            w[:] = 1.0 / N

        mu      = float(np.average(pos, weights=w))
        pts[i]  = mu
        std[i]  = float(np.sqrt(np.average((pos - mu) ** 2, weights=w)))
        pm = md_v[i]; pz = z_v[i]

    return pts, std


def run_pf_ancc(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                N: int = ANCC_N) -> Tuple[np.ndarray, np.ndarray]:
    """ANCC (TVT+Z composite) particle filter."""
    tmin, tmax = tw_tvt.min(), tw_tvt.max()
    gs  = _cal_gr_sigma(hw, tw_tvt, tw_gr)
    kn  = hw[hw["TVT_input"].notna()]
    ev  = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])

    ls  = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt  = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm  = np.diff(tail["MD"].values); m = dm > 0
    ir  = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    pos  = ls + np.random.normal(0, ANCC_IS, N)
    rate = ir + np.random.normal(0, ANCC_IR, N)
    w    = np.ones(N) / N

    md_v = ev["MD"].values; z_v = ev["Z"].values; gr_v = ev["GR"].values
    pm   = float(kn["MD"].iloc[-1])
    pts  = np.empty(len(ev)); std = np.empty(len(ev))

    for i in range(len(ev)):
        dm   = max(md_v[i] - pm, 1.0)
        rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RN, N)
        pos  = pos + rate * dm + np.random.normal(0, ANCC_PN, N)
        tvt_e = np.clip(pos - z_v[i], tmin - 50, tmax + 50)
        pos   = tvt_e + z_v[i]

        if not np.isnan(gr_v[i]):
            eg  = np.interp(tvt_e, tw_tvt, tw_gr)
            lk  = np.exp(-0.5 * ((gr_v[i] - eg) / gs) ** 2)
            lk  = np.maximum(lk, 1e-300); w *= lk
            ws  = w.sum(); w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)

        ne = 1.0 / np.sum(w ** 2)
        if ne < PF_RESAMP * N:
            ix   = systematic_resample(w, N)
            pos  = pos[ix]  + np.random.normal(0, ANCC_RP, N)
            rate = rate[ix] + np.random.normal(0, ANCC_RR, N)
            w[:] = 1.0 / N

        tv     = float(np.average(pos - z_v[i], weights=w))
        pts[i] = tv
        std[i] = float(np.sqrt(np.average((pos - z_v[i] - tv) ** 2, weights=w)))
        pm     = md_v[i]

    return pts, std

print("Particle Filters loaded ✓")


# ## 6. Self-Correlation NCC
# #
# KEY INSIGHT from Super Solution: prefix GR can be self-correlated against
# the hidden GR to infer TVT. The NCC finds the prefix window that best
# matches each hidden GR window, then maps back to TVT via known_tvt.

# In[ ]:


def self_corr_tvt(
    kgr:    np.ndarray,
    ktvt:   np.ndarray,
    hgr:    np.ndarray,
    hw:     int = 15,
    stride: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each position in the hidden section, find the known-section window
    with highest NCC against the current hidden window. Returns the TVT
    at the best-match center and the NCC score.
    """
    win = 2 * hw + 1
    nk  = len(kgr); nh = len(hgr)
    if nk < win + 1 or nh == 0:
        return np.full(nh, float(ktvt[-1]), np.float32), np.zeros(nh, np.float32)

    kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
    hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)

    sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
    M   = len(sts)
    if M == 0:
        return np.full(nh, float(ktvt[-1]), np.float32), np.zeros(nh, np.float32)

    # Build template matrix (M, win)
    C  = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
    Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)

    # Build hidden windows (nh, win) with edge padding
    hp = np.pad(hg, hw, mode="edge")
    H  = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
    Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)

    # NCC matrix (nh, M)
    ncc   = Hn @ Cn.T / win
    best  = ncc.argmax(1)
    score = ncc.max(1).astype(np.float32)

    # Map to TVT via center of best-matching window
    ctrs = np.clip(sts[best] + hw, 0, nk - 1)
    return ktvt[ctrs].astype(np.float32), score

print("Self-correlation NCC loaded ✓")


# ## 7. Spatial Imputers
# #
# FormationPlaneKNN: fits a weighted plane through the 6 formation surfaces.
# DenseANCCImputer: IDW from dense ANCC samples across all training wells.
# Both exclude self-well during CV to prevent leakage.

# In[ ]:


class FormationPlaneKNN:
    """
    For each query point, fit a weighted plane to K nearest well centroids
    and predict formation surface depths for all 6 FORMATIONS.
    """

    def __init__(self, well_ids: List[str], data_dir: Path):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                avail = FORMATIONS + ["X", "Y"]
                df = pd.read_csv(p, usecols=[c for c in avail])
            except Exception:
                continue
            # only keep formations that exist in this file
            form_cols = [c for c in FORMATIONS if c in df.columns]
            if not form_cols or "X" not in df.columns or "Y" not in df.columns:
                continue
            df = df[["X", "Y"] + form_cols].dropna()
            if len(df) == 0:
                continue
            row = {"wid": wid,
                   "x": float(df["X"].median()),
                   "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median()) if c in df.columns else np.nan
            rows.append(row)

        self.df = pd.DataFrame(rows)
        if len(self.df) == 0:
            self._empty = True
            return
        self._empty = False
        self.wmap  = {w: i for i, w in enumerate(self.df["wid"])}
        xy         = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree  = cKDTree(xy / self.scale)
        self.xa    = self.df["x"].to_numpy()
        self.ya    = self.df["y"].to_numpy()
        self.fa    = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q: np.ndarray, self_wid: str = None,
               k: int = PLANE_K) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predictions: (N, 6), min_dist: (N,))."""
        if self._empty:
            N = len(xy_q)
            return (np.full((N, len(FORMATIONS)), np.nan, np.float32),
                    np.full(N, np.inf, np.float32))

        q   = np.atleast_2d(xy_q) / self.scale
        nf  = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)

        ord_ = np.argpartition(dist, min(k - 1, nf - 1), axis=1)[:, :k]
        dk   = np.take_along_axis(dist, ord_, axis=1)
        ik   = np.take_along_axis(idx,  ord_, axis=1)
        vk   = np.isfinite(dk)
        w    = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)

        xn = self.xa[ik]; yn = self.ya[ik]
        wx = w * xn;      wy = w * yn

        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1];       A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2];       A[:, 2, 1] = A[:, 1, 2];       A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9

        fn  = self.fa[ik]  # (N, k, 6)
        rhs = np.stack([(wx[:, :, None] * fn).sum(1),
                        (wy[:, :, None] * fn).sum(1),
                        (w[:,  :, None] * fn).sum(1)], axis=1)  # (N, 3, 6)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except: pass

        Xq   = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] +
                Yq[:, None] * coef[:, 1, :] +
                coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0).astype(np.float32)

        min_dist = np.where(vk, dk, np.inf).min(1).astype(np.float32)
        return pred, min_dist


class DenseANCCImputer:
    """IDW interpolation from dense ANCC point cloud (60 pts/well)."""

    def __init__(self, well_ids: List[str], data_dir: Path, spw: int = DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s  = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))

        if not xs:
            self._empty = True
            return
        self._empty = False
        self.xy    = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc  = np.concatenate(anccs).astype(np.float32)
        self.wids  = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree  = cKDTree(self.xy / self.scale)
        self._mean = float(self.ancc.mean())

    def impute(self, xy_q: np.ndarray, self_wid: str = None,
               k: int = DENSE_K, nfetch: int = 3000
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean_ancc, std_ancc, min_dist) each (N,)."""
        if self._empty:
            N = len(xy_q)
            fb = self._mean if hasattr(self, "_mean") else 0.0
            return (np.full(N, fb, np.float32),
                    np.zeros(N, np.float32),
                    np.full(N, np.inf, np.float32))

        xy_q = np.atleast_2d(xy_q)
        q    = xy_q / self.scale
        nf   = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)

        ord_ = np.argpartition(dist, min(k - 1, nf - 1), axis=1)[:, :k]
        dk   = np.take_along_axis(dist, ord_, axis=1)
        ik   = np.take_along_axis(idx,  ord_, axis=1)
        vk   = np.isfinite(dk)
        w    = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw   = w.sum(1); safe = np.where(sw < 1e-9, 1.0, sw)

        an   = self.ancc[ik]
        ap   = (an * w).sum(1) / safe
        ap   = np.where(sw < 1e-9, self._mean, ap)
        var  = ((an - ap[:, None]) ** 2 * w).sum(1) / safe

        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


# Build global imputers (will be rebuilt per-split during CV)
print("Building global spatial imputers from all train wells...")
_t0 = time.time()
_hw_paths   = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
_train_wids = [p.stem.replace("__horizontal_well", "") for p in _hw_paths]

if DEBUG_MAX_WELLS:
    _train_wids = _train_wids[:DEBUG_MAX_WELLS]

_FI_GLOBAL = FormationPlaneKNN(_train_wids, TRAIN_DIR)
_DI_GLOBAL = DenseANCCImputer(_train_wids, TRAIN_DIR)
print(f"  FormationKNN: {len(_FI_GLOBAL.df) if not _FI_GLOBAL._empty else 0} centroids"
      f"  |  DenseANCC: {len(_DI_GLOBAL.ancc) if not _DI_GLOBAL._empty else 0} pts"
      f"  ({time.time()-_t0:.0f}s)")

print("Spatial imputers loaded ✓")


# ## 8. Per-Well Feature Builder
# #
# Orchestrates all feature modules into one flat DataFrame per well.
# Thread-safe: uses module-level globals for imputers.

# In[ ]:


# Thread-safe global imputer references (overridden per CV fold during training)
_FI_REF = _FI_GLOBAL
_DI_REF = _DI_GLOBAL


def affine_cal(kgr: np.ndarray, tw_at_k: np.ndarray, min_pts: int = 20
               ) -> Tuple[float, float]:
    """Fit affine: kgr ≈ a * tw_at_k + b."""
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        bias = float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
        return 1.0, bias
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def build_well_features(
    hw_path:     str,
    tw_path:     str,
    is_train:    bool,
    fi:          FormationPlaneKNN = None,
    di:          DenseANCCImputer  = None,
    debug_pf:    bool = False,
    debug_beam:  bool = False,
) -> Optional[pd.DataFrame]:
    """
    Build complete feature DataFrame for one well.
    fi/di: spatial imputers (use globals if None).
    Returns None if no usable hidden rows.
    """
    global _FI_REF, _DI_REF

    fi_use = fi if fi is not None else _FI_REF
    di_use = di if di is not None else _DI_REF

    hw_path = Path(hw_path)
    wid     = hw_path.stem.replace("__horizontal_well", "")

    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path)
    except Exception as exc:
        print(f"  WARN [{wid}]: {exc}")
        return None

    if "TVT" not in tw.columns or "GR" not in tw.columns:
        return None
    tw = tw.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr  = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    # ── Split known/hidden ───────────────────────────────────────────────
    mask       = hw["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    mask_start = int(np.flatnonzero(mask)[0])
    if mask_start == 0:
        return None

    kn = hw.iloc[:mask_start].copy()
    ev = hw.iloc[mask_start:].copy()

    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        ev = ev[ev["TVT"].notna()].copy()
    if len(ev) == 0 or len(kn) < 10:
        return None

    # ── GR features (full well) ─────────────────────────────────────────
    gr_mean_tw = float(np.nanmean(tw_gr))
    grd        = compute_gr_features(hw["GR"], fallback=gr_mean_tw)

    # Fully filled GR for hidden section
    gr_full = grd["gr_filled"]
    hgr     = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr     = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    nh_full = int(hw["TVT_input"].isna().sum())

    # Indices into hidden section
    sel_full  = ev.index.to_numpy(np.int64)
    sel_local = (sel_full - mask_start).astype(np.int64)

    hgr_sel = hgr[sel_local]

    # ── Anchors ─────────────────────────────────────────────────────────
    lk           = kn.iloc[-1]
    ktvt         = kn["TVT_input"].to_numpy(np.float32)
    kmd          = kn["MD"].to_numpy(np.float32)
    kz           = kn["Z"].to_numpy(np.float32)
    last_tvt     = float(lk["TVT_input"])
    last_md      = float(lk["MD"])
    last_z       = float(lk["Z"])
    last_x       = float(lk["X"])
    last_y       = float(lk["Y"])
    last_gr      = float(lk["GR"]) if not np.isnan(lk["GR"]) else gr_mean_tw

    # ── Typewell at last known ──────────────────────────────────────────
    tw_at_k      = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    tw_gr_at_lk  = float(np.interp(last_tvt, tw_tvt, tw_gr))
    lk_tw_idx    = nn_idx(tw_tvt, last_tvt)
    local_win    = tw_gr[max(0, lk_tw_idx - 5): lk_tw_idx + 6]
    tw_gr_std_local = float(np.std(local_win)) if len(local_win) > 1 else 0.0

    # ── Affine GR calibration ───────────────────────────────────────────
    a_cal, b_cal = affine_cal(kgr, tw_at_k)

    # ── Prefix residuals ────────────────────────────────────────────────
    pr            = kgr - tw_at_k
    prefix_tw_rmse = float(np.sqrt(np.mean(pr ** 2))) if len(pr) else 0.0
    prefix_tw_mae  = float(np.mean(np.abs(pr)))       if len(pr) else 0.0
    prefix_tw_bias = float(pr.mean())                 if len(pr) else 0.0

    # ── Prefix TVT trends ───────────────────────────────────────────────
    pfx_tvt_step20   = recent_mean_diff(ktvt, 20)
    pfx_tvt_step100  = recent_mean_diff(ktvt, 100)
    slp_all  = robust_slope(kmd, ktvt)
    slp_50   = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z    = robust_slope(kz, ktvt)

    # ── Position features ───────────────────────────────────────────────
    hmd      = ev["MD"].to_numpy(np.float32)
    hz       = ev["Z"].to_numpy(np.float32)
    hx       = ev["X"].to_numpy(np.float32)
    hy       = ev["Y"].to_numpy(np.float32)
    md_since = (hmd - last_md).astype(np.float32)
    dz       = (hz - last_z).astype(np.float32)
    dx       = (hx - last_x).astype(np.float32)
    dy       = (hy - last_y).astype(np.float32)
    dxy      = np.sqrt(dx**2 + dy**2).astype(np.float32)
    sdmd     = np.maximum(md_since, 1e-5)
    frac     = (sel_local / max(nh_full - 1, 1)).astype(np.float32)
    frac2    = frac ** 2
    sqrt_frac = np.sqrt(frac)

    # Trajectory derivatives
    mdd      = hw["MD"].diff().replace(0, np.nan)
    dzdmd    = (hw["Z"].diff() / mdd).iloc[sel_full].values.astype(np.float32)
    dxdmd    = (hw["X"].diff() / mdd).iloc[sel_full].values.astype(np.float32)
    dydmd    = (hw["Y"].diff() / mdd).iloc[sel_full].values.astype(np.float32)

    # Slope baselines
    slp_base_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_base_50  = (last_tvt + slp_50  * md_since).astype(np.float32)

    # ── Beam features (5 configs) ───────────────────────────────────────
    np.random.seed(SEED + hash(wid) % 1000)
    bf = compute_beam_features(hgr, tw_tvt, tw_gr, last_tvt, sel_local)
    beam_ref_full = bf.pop("_beam_ref_full")                 # full array
    beam_ref      = beam_ref_full[sel_local]

    # ── Particle filters ────────────────────────────────────────────────
    pf_z, std_z     = run_pf_z(hw, tw_tvt, tw_gr)
    pf_a, std_a     = run_pf_ancc(hw, tw_tvt, tw_gr)

    if len(pf_z) == 0: pf_z = np.full(nh_full, last_tvt, np.float32); std_z = np.ones(nh_full, np.float32)
    if len(pf_a) == 0: pf_a = np.full(nh_full, last_tvt, np.float32); std_a = np.ones(nh_full, np.float32)

    has_z = len(pf_z) == nh_full and not np.any(np.isnan(pf_z))
    pf_z_sel  = pf_z[sel_local].astype(np.float32)  if has_z else np.full(len(sel_local), last_tvt, np.float32)
    std_z_sel = std_z[sel_local].astype(np.float32)  if has_z else np.ones(len(sel_local), np.float32)
    pf_a_sel  = pf_a[sel_local].astype(np.float32)
    std_a_sel = std_a[sel_local].astype(np.float32)

    # ── Self-correlation NCC ────────────────────────────────────────────
    sc_raw, sc_score = self_corr_tvt(kgr, ktvt, hgr, hw=15, stride=3)
    sc_trust         = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref          = (1 - sc_trust) * beam_ref_full + sc_trust * sc_raw
    sc_raw_sel       = sc_raw[sel_local]
    sc_score_sel     = sc_score[sel_local]
    hyb_ref_sel      = hyb_ref[sel_local]

    # ── Spatial features ────────────────────────────────────────────────
    swid    = wid if is_train else None
    xy_ev   = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn   = kn[["X", "Y"]].to_numpy(np.float64)

    form_ev, knn_d  = fi_use.impute(xy_ev, self_wid=swid)    # (nh, 6), (nh,)
    form_kn, _      = fi_use.impute(xy_kn, self_wid=swid)    # (nk, 6)

    # TVT formula from each formation: TVT = -Z + F + b_well
    tvt_formulas = {}
    for fi_idx, fn in enumerate(FORMATIONS):
        b_v   = ktvt + kz - form_kn[:, fi_idx]
        b_all = float(np.nanmedian(b_v))
        b_50  = float(np.nanmedian(b_v[-50:])) if len(b_v) >= 5 else b_all
        tvt_f_all = (-hz + form_ev[:, fi_idx] + b_all).astype(np.float32)
        tvt_f_50  = (-hz + form_ev[:, fi_idx] + b_50).astype(np.float32)
        tvt_formulas[f"tvtF_{fn}_d"]   = (tvt_f_all - last_tvt).astype(np.float32)
        tvt_formulas[f"tvtF50_{fn}_d"] = (tvt_f_50  - last_tvt).astype(np.float32)
        tvt_formulas[f"bw_{fn}"]       = np.float32(b_all)
        tvt_formulas[f"bw50_{fn}"]     = np.float32(b_50)

    # Dense ANCC
    d_ancc, d_std, d_dist   = di_use.impute(xy_ev, self_wid=swid)
    d_kn,   d_std_kn, _     = di_use.impute(xy_kn, self_wid=swid)
    b_vd    = ktvt + kz - d_kn
    b_d     = float(np.nanmedian(b_vd))
    b_d50   = float(np.nanmedian(b_vd[-50:])) if len(b_vd) >= 5 else b_d
    tvt_dense   = (-hz + d_ancc + b_d).astype(np.float32)
    tvt_dense50 = (-hz + d_ancc + b_d50).astype(np.float32)
    res_kn  = ktvt + kz - d_kn
    d_rmse  = float(np.sqrt(np.nanmean(res_kn ** 2)))
    d_bias  = float(np.nanmean(res_kn))

    # ── 3-anchor tw_diff features ───────────────────────────────────────
    # Anchor 1: last_known_tvt
    tw_diff_anch = {f"tda{int(o)}": (hgr_sel - float(np.interp(last_tvt + o, tw_tvt, tw_gr))).astype(np.float32)
                    for o in ANCH_OFFS}
    # Anchor 2: beam_ref
    tw_diff_beam = {f"tdbc{int(o)}": (hgr_sel - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32))
                    for o in BEAM_OFFS}
    # Anchor 3: sc_raw
    tw_diff_sc   = {f"tdsc{int(o)}": (hgr_sel - np.interp(sc_raw_sel + o, tw_tvt, tw_gr).astype(np.float32))
                    for o in SC_OFFS}

    # ── GR rolling features ─────────────────────────────────────────────
    gr_s = pd.Series(gr_full.values)
    gr_rolls = {}
    for w_size in [5, 21, 51, 101]:
        ro = gr_s.rolling(w_size, center=True, min_periods=1)
        gr_rolls[f"grm{w_size}"]  = ro.mean().iloc[sel_full].values.astype(np.float32)
        gr_rolls[f"grs{w_size}"]  = ro.std().fillna(0).iloc[sel_full].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        gr_rolls[f"glag{lag}"]  = gr_s.shift(lag).bfill().iloc[sel_full].values.astype(np.float32)
        gr_rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[sel_full].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[sel_full].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[sel_full].values.astype(np.float32)

    # ── GR cumsum since mask_start ──────────────────────────────────────
    cs_offset = grd["cumsum"].iloc[mask_start - 1] if mask_start > 0 else 0.0
    gr_cumsum  = (grd["cumsum"].iloc[sel_full].values - cs_offset).astype(np.float32)

    # ── Prefix GR stats ─────────────────────────────────────────────────
    prefix_gr_mean   = float(np.nanmean(kgr)) if len(kgr) > 0 else gr_mean_tw
    prefix_gr_std    = float(np.nanstd(kgr))  if len(kgr) > 1 else 0.0
    prefix_gr_last5  = float(np.nanmean(kgr[-5:]))  if len(kgr) >= 5  else prefix_gr_mean
    prefix_gr_last20 = float(np.nanmean(kgr[-20:])) if len(kgr) >= 20 else prefix_gr_mean

    def sc(v): return np.full(len(ev), np.float32(v), np.float32)

    # ── Assemble final DataFrame ────────────────────────────────────────
    out = pd.DataFrame({
        # Meta
        "well"          : wid,
        "id"            : [f"{wid}_{i}" for i in sel_full],
        # Position
        "last_known_tvt": sc(last_tvt),
        "known_len"     : sc(mask_start),
        "hidden_len"    : sc(nh_full),
        "frac"          : frac,
        "frac2"         : frac2,
        "sqrt_frac"     : sqrt_frac,
        "md_since"      : md_since,
        "z"             : hz,
        "dx"            : dx, "dy": dy, "dz": dz, "dxy": dxy,
        "dxdmd"         : dxdmd, "dydmd": dydmd, "dzdmd": dzdmd,
        # GR raw
        "gr"            : hgr_sel,
        "gr_d1"         : gr_d1, "gr_d2": gr_d2,
        "last_known_gr" : sc(last_gr),
        "gr_minus_last" : (hgr_sel - last_gr).astype(np.float32),
        "gr_vs_tw_anc"  : (hgr_sel - tw_gr_at_lk).astype(np.float32),
        "gr_cumsum"     : gr_cumsum,
        # GR vs slope baseline
        "gr_vs_slp_all" : (hgr_sel - np.interp(slp_base_all, tw_tvt, tw_gr).astype(np.float32)),
        **gr_rolls,
        # Prefix stats
        "prefix_gr_mean"    : sc(prefix_gr_mean),
        "prefix_gr_std"     : sc(prefix_gr_std),
        "prefix_gr_last5"   : sc(prefix_gr_last5),
        "prefix_gr_last20"  : sc(prefix_gr_last20),
        "prefix_tw_rmse"    : sc(prefix_tw_rmse),
        "prefix_tw_mae"     : sc(prefix_tw_mae),
        "prefix_tw_bias"    : sc(prefix_tw_bias),
        "pfx_tvt_step20"    : sc(pfx_tvt_step20),
        "pfx_tvt_step100"   : sc(pfx_tvt_step100),
        "slp_all"           : sc(slp_all),
        "slp_50"            : sc(slp_50),
        "slp_z"             : sc(slp_z),
        "slp_base_d_all"    : (slp_base_all - last_tvt).astype(np.float32),
        "slp_base_d_50"     : (slp_base_50  - last_tvt).astype(np.float32),
        # Affine calibration
        "cal_a"             : sc(a_cal),
        "cal_b"             : sc(b_cal),
        # Typewell stats
        "tw_gr_at_lk"       : sc(tw_gr_at_lk),
        "tw_gr_std_local"   : sc(tw_gr_std_local),
        "tw_tvt_range"      : sc(float(tw_tvt[-1] - tw_tvt[0])),
        "tw_gr_mean"        : sc(float(tw_gr.mean())),
        "tw_gr_std"         : sc(float(tw_gr.std())),
        "ktvt_range"        : sc(float(np.ptp(ktvt))),
        "ktvt_std"          : sc(float(ktvt.std())),
        # PF features (Z-velocity model)
        "pf_z_d"            : (pf_z_sel - last_tvt).astype(np.float32),
        "pf_z_std"          : std_z_sel,
        # PF features (ANCC model)
        "pf_a_d"            : (pf_a_sel - last_tvt).astype(np.float32),
        "pf_a_std"          : std_a_sel,
        # PF cross-signals
        "pf_vs_z"           : (pf_a_sel - pf_z_sel).astype(np.float32),
        # Self-correlation NCC
        "sc_d"              : (sc_raw_sel - last_tvt).astype(np.float32),
        "sc_score"          : sc_score_sel,
        "sc_trust"          : sc(sc_trust),
        "hyb_d"             : (hyb_ref_sel - last_tvt).astype(np.float32),
        # Beam features
        **bf,
        # PF vs beam
        "pf_a_vs_beam_cons" : (pf_a_sel - last_tvt - bf.get("beam_cons_d", np.zeros(len(ev)))),
        "pf_z_vs_beam_cons" : (pf_z_sel - last_tvt - bf.get("beam_cons_d", np.zeros(len(ev)))),
        "sc_vs_beam_cons"   : (sc_raw_sel - last_tvt - bf.get("beam_cons_d", np.zeros(len(ev)))),
        # Spatial formation formulas
        **{k: (np.full(len(ev), v, np.float32) if np.isscalar(v) else v)
           for k, v in tvt_formulas.items()},
        "spatial_knn_dist"  : knn_d,
        # Dense ANCC
        "dense_ancc_d"      : (tvt_dense   - last_tvt).astype(np.float32),
        "dense_ancc50_d"    : (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_std"         : d_std,
        "dense_dist"        : d_dist,
        "dense_rmse"        : sc(d_rmse),
        "dense_bias"        : sc(d_bias),
        # PF vs spatial
        "pf_a_vs_dense"     : (pf_a_sel - tvt_dense).astype(np.float32),
        "pf_a_vs_spatialF"  : (pf_a_sel - (-hz + form_ev[:, 0] + float(np.nanmedian(ktvt + kz - form_kn[:, 0])))).astype(np.float32),
        "beam_vs_dense"     : (beam_ref - tvt_dense).astype(np.float32),
        "sc_vs_dense"       : (sc_raw_sel - tvt_dense).astype(np.float32),
        # 3-anchor tw_diff families
        **tw_diff_anch,
        **tw_diff_beam,
        **tw_diff_sc,
    })

    if is_train:
        out["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))

    return out.reset_index(drop=True)

print("Per-well feature builder loaded ✓")


# ## 9. Dataset Builder (parallel joblib)

# In[ ]:


def _build_one(hw_path: str, is_train: bool) -> Optional[pd.DataFrame]:
    """Worker function for parallel execution."""
    wid    = Path(hw_path).stem.replace("__horizontal_well", "")
    tw_p   = str(Path(hw_path).parent / f"{wid}__typewell.csv")
    if not Path(tw_p).exists():
        return None
    try:
        return build_well_features(hw_path, tw_p, is_train=is_train)
    except Exception as exc:
        print(f"  WARN [{wid}]: {exc}")
        return None


def build_dataset(
    data_dir:   Path,
    is_train:   bool,
    max_wells:  int = None,
    n_jobs:     int = NCPU,
) -> pd.DataFrame:
    from tqdm.auto import tqdm  # Ensure tqdm is available

    hw_files = sorted(data_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        hw_files = hw_files[:max_wells]
        print(f"  DEBUG: limited to {max_wells} wells")

    label = "train" if is_train else "test"
    print(f"Building {label} dataset from {len(hw_files)} wells...")

    # Wrap the delayed generator with tqdm
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_one)(str(p), is_train) 
        for p in tqdm(hw_files, desc=f"Feature Engineering ({label})")
    )

    parts = [r for r in results if r is not None]
    n_skip = len(hw_files) - len(parts)
    df = pd.concat(parts, ignore_index=True)
    print(f"  → {df.shape}  wells={df['well'].nunique()}  skipped={n_skip}")
    return df


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    wmap = {w: i for i, w in enumerate(sorted(df["well"].unique()))}
    df["group_id"] = df["well"].map(wmap).astype(np.int32)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS and c != "group_id"]

print("Dataset builder loaded ✓")


# ## 9.5 Edge R — Test-Time Online Learning (exp005 v3 新規)
# #
# `discussion 698002` で公開された "online 10.953 vs no-online 11.323 = -0.370 ft"
# の効果を再現する。 test wells の visible region 末尾 K 行を擬似 hidden に変換し、
# `build_well_features(.., is_train=True)` 経路で (X, y) を生成、
# pretrained LGB に `init_model` で continued training する。
#
# leak guard:
#   - visible region の TVT_input のみ参照、 hidden mask 域 (= host が公開してない部分) は
#     一切触らない (= build_well_features の `ev` filter で自動排除)
#   - 各 well 内で完結、 cross-well leak なし
#   - karnakbaev pretrained は train data で fit 済、 test visible で warm start するのみ

# In[ ]:


def build_visible_dataset(
    data_dir:   Path,
    max_wells:  int = None,
    tail_k:     int = EDGE_R_VISIBLE_TAIL_K,
    min_visible_rows: int = EDGE_R_MIN_VISIBLE_ROWS,
    n_jobs:     int = NCPU,
) -> Optional[pd.DataFrame]:
    """Build a test-visible-as-pseudo-hidden dataset for Edge R online learning.

    各 test well で:
      1. visible rows (= TVT_input.notna()) を抽出
      2. 末尾 `tail_k` 行を擬似 hidden に変換 (= TVT_input → NaN にマスク)
      3. TVT 列を visible の TVT_input から copy (= build_well_features の target 要求を満たす)
      4. tmp file に save し、 build_well_features(is_train=True) で feature 化
      5. 全 well で concat
    visible 行数 < `min_visible_rows + tail_k` の well は skip。
    """
    import tempfile

    hw_files = sorted(data_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        hw_files = hw_files[:max_wells]

    print(f"[Edge R] Building visible-as-pseudo-hidden dataset from {len(hw_files)} wells "
          f"(tail_k={tail_k}, min_visible={min_visible_rows})")

    def _build_one_visible(hw_path_str: str) -> Optional[pd.DataFrame]:
        hw_path = Path(hw_path_str)
        wid     = hw_path.stem.replace("__horizontal_well", "")
        tw_p    = hw_path.parent / f"{wid}__typewell.csv"
        if not tw_p.exists():
            return None
        try:
            hw_raw = pd.read_csv(hw_path)
        except Exception:
            return None
        if "TVT_input" not in hw_raw.columns:
            return None

        visible_mask = hw_raw["TVT_input"].notna().to_numpy()
        n_visible    = int(visible_mask.sum())
        # need: min_visible_rows for kn + tail_k for ev
        if n_visible < (min_visible_rows + tail_k):
            return None

        # visible 行のうち末尾 tail_k 行を擬似 hidden に変換
        visible_idx = np.flatnonzero(visible_mask)
        tail_idx    = visible_idx[-tail_k:]
        # original_tvt_input は temp 用に保存（target に使う）
        orig_tvt_input = hw_raw["TVT_input"].to_numpy(np.float32).copy()

        hw_mod = hw_raw.copy()
        # target 列を host train-style に作る: TVT = original TVT_input (visible のみ filled)
        hw_mod["TVT"] = np.nan
        hw_mod.loc[hw_mod["TVT_input"].notna(), "TVT"] = orig_tvt_input[visible_mask]
        # 擬似 hidden 化: 末尾 tail_k 行の TVT_input を NaN に
        hw_mod.loc[tail_idx, "TVT_input"] = np.nan
        # build_well_features は ev["TVT"].notna() を filter するので、
        # 擬似 hidden 行の TVT (= 元 TVT_input) は残しておく → target に使われる

        # tmp file に save (= build_well_features は csv path を要求)
        tmp_dir = tempfile.gettempdir()
        tmp_hw  = Path(tmp_dir) / f"_edge_r_{wid}__horizontal_well.csv"
        hw_mod.to_csv(tmp_hw, index=False)

        try:
            df = build_well_features(str(tmp_hw), str(tw_p), is_train=True)
        except Exception as exc:
            print(f"  WARN [Edge R {wid}]: {exc}")
            df = None
        finally:
            try: tmp_hw.unlink()
            except Exception: pass
        return df

    from tqdm.auto import tqdm
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_one_visible)(str(p))
        for p in tqdm(hw_files, desc="Edge R (visible-as-ev)")
    )
    parts = [r for r in results if r is not None]
    if not parts:
        print("[Edge R] No usable wells — disabling Edge R")
        return None
    df = pd.concat(parts, ignore_index=True)
    print(f"[Edge R] visible dataset: {df.shape}  wells={df['well'].nunique()}  "
          f"skipped={len(hw_files) - len(parts)}")
    return df


def edge_r_continued_train_lgb(
    base_booster,
    X_R: np.ndarray,
    y_R: np.ndarray,
    base_params: dict,
    num_boost_round: int = EDGE_R_NUM_BOOST,
    lr_mul: float = EDGE_R_LR_MUL,
):
    """Apply LightGBM continued training (warm start) on visible-as-pseudo-hidden data.

    base_params から lr × lr_mul、 early stopping 無効化、 objective 維持で
    `lgb.train(.., init_model=base_booster, num_boost_round=R)` を実行。
    GPU device は continued training で OpenCL/CUDA 不在になりやすいので CPU 強制。
    karnakbaev pretrained は CPU device で fit されている (= 互換性 OK)。
    """
    p = dict(base_params)
    p["learning_rate"] = float(p.get("learning_rate", 0.04)) * float(lr_mul)
    p.pop("n_estimators", None)
    p.pop("early_stopping_rounds", None)
    # Edge R は CPU で続行: GPU continued training は API 不安定 + Kaggle CPU kernel でも問題なし
    p["device_type"] = "cpu"
    p.pop("gpu_use_dp", None)
    ds = lgb.Dataset(X_R, label=y_R)
    online = lgb.train(
        p, ds,
        num_boost_round=int(num_boost_round),
        init_model=base_booster,
        keep_training_booster=False,
    )
    return online


def edge_r_apply(
    test_df:         pd.DataFrame,
    visible_df:      Optional[pd.DataFrame],
    base_models:     dict,
    feature_cols:    List[str],
    nm_w:            dict,
    test_delta_base: np.ndarray,
    log,
) -> Tuple[np.ndarray, dict]:
    """Run Edge R pipeline and return (test_delta_blended, diagnostics).

    Returns:
      test_delta_blended: blended delta (= w_R × online_NM + (1-w_R) × base_NM)
      diagnostics: dict of online metrics
    """
    diag: Dict[str, Any] = {"applied": False}

    if not EDGE_R_ENABLE or visible_df is None or len(visible_df) == 0:
        log.warning("[Edge R] skipped: disabled or empty visible dataset")
        return test_delta_base, diag

    # feature_cols は LGB が知っている columns (= karnakbaev schema)
    missing = [c for c in feature_cols if c not in visible_df.columns]
    if missing:
        log.warning(f"[Edge R] visible_df missing {len(missing)} cols, e.g. {missing[:3]} — filling 0")
        for c in missing:
            visible_df[c] = np.float32(0.0)

    X_R = visible_df[feature_cols].to_numpy(np.float32)
    y_R = visible_df["target"].to_numpy(np.float32)
    log.info(f"[Edge R] X_R shape={X_R.shape}  y_R range=[{y_R.min():.3f}, {y_R.max():.3f}]")

    # NaN/inf guard
    if not np.all(np.isfinite(y_R)):
        n_bad = int((~np.isfinite(y_R)).sum())
        log.warning(f"[Edge R] dropping {n_bad} non-finite y_R rows")
        finite = np.isfinite(y_R)
        X_R, y_R = X_R[finite], y_R[finite]
    if len(y_R) < 100:
        log.warning(f"[Edge R] too few visible-as-pseudo-hidden rows ({len(y_R)}) — disabling")
        return test_delta_base, diag

    # LGB 3 base のみ continued training (= XGB/CB は base predict そのまま)
    X_test_full = test_df[feature_cols].to_numpy(np.float32)
    online_preds = {}
    for seed_idx, seed in enumerate(LGB_SEEDS):
        key = f"lgb{seed_idx}"
        if key not in base_models:
            continue
        try:
            base_params = make_lgb_model(seed)
            online_booster = edge_r_continued_train_lgb(
                base_models[key], X_R, y_R, base_params,
                num_boost_round=EDGE_R_NUM_BOOST,
                lr_mul=EDGE_R_LR_MUL,
            )
            p_online = online_booster.predict(X_test_full).astype(np.float64)
            online_preds[key] = p_online
            log.info(f"[Edge R]   {key} online predict range="
                     f"[{p_online.min():.3f}, {p_online.max():.3f}]")
        except Exception as exc:
            log.warning(f"[Edge R]   {key} continued training failed ({type(exc).__name__}: {exc})")
            online_preds[key] = base_models[key].predict(X_test_full).astype(np.float64)
            log.warning(f"[Edge R]   {key} fallback to base predict")

    if not online_preds:
        log.warning("[Edge R] no online predictions produced — keeping base delta")
        return test_delta_base, diag

    # NM blend は karnakbaev 5 base 想定なので、 online は LGB 3 だけだが
    # NM blend に LGB 3 を online で差替え + XGB/CB を base で残す、 として concat
    online_full = {}
    for k in ACTIVE_MODELS:
        if k in online_preds:
            online_full[k] = online_preds[k]
        elif k in base_models:
            # XGB/CB は base predict (= continued training なし)
            m = base_models[k]
            if k == "cb":
                online_full[k] = m.predict(X_test_full.astype(np.float64)).astype(np.float64)
            else:
                online_full[k] = m.predict(X_test_full).astype(np.float64)

    test_delta_online = apply_ensemble_nm(online_full, nm_w)
    log.info(f"[Edge R] online NM blend range="
             f"[{test_delta_online.min():.3f}, {test_delta_online.max():.3f}]")
    log.info(f"[Edge R] base    NM blend range="
             f"[{test_delta_base.min():.3f}, {test_delta_base.max():.3f}]")

    w = float(EDGE_R_BLEND_W)
    blended = w * test_delta_online + (1.0 - w) * test_delta_base
    diff = blended - test_delta_base
    log.info(f"[Edge R] blended w={w} (online) + {1-w} (base)")
    log.info(f"[Edge R]   diff abs mean: {np.abs(diff).mean():.4f}  max: {np.abs(diff).max():.4f}")

    diag = {
        "applied"       : True,
        "n_visible_rows": int(len(y_R)),
        "n_lgb_online"  : int(sum(1 for k in online_preds if k.startswith("lgb"))),
        "w_R"           : w,
        "diff_abs_mean" : float(np.abs(diff).mean()),
    }
    return blended, diag


print("Edge R (test-time online learning) loaded ✓")


# ## 10. Model Registry (LGB×3 seeds + XGB + CatBoost)

# In[ ]:


def make_lgb_model(seed: int, n_est: int = None):
    p = dict(LGB_BASE)
    p["seed"] = seed
    if n_est is not None:
        p["n_estimators"] = n_est
    return p   # return params dict (use lgb.train API for early stopping)


def make_xgb_model(n_est: int = None):
    p = dict(XGB_PARAMS)
    if n_est is not None:
        p["n_estimators"] = n_est
    return XGBRegressor(**p)


def make_cb_model(n_iter: int = None):
    p = dict(CB_PARAMS)
    if n_iter is not None:
        p["iterations"] = n_iter
    return CatBoostRegressor(**p)


def save_lgb_model(booster, directory: Path, tag: str) -> Path:
    path = directory / f"final_lgb_{tag}.txt"
    booster.save_model(str(path))
    print(f"  ✔ saved {path.name}")
    return path


def save_xgb_model(model, directory: Path) -> Path:
    path = directory / "final_xgb.json"
    model.save_model(str(path))
    print(f"  ✔ saved {path.name}")
    return path


def save_cb_model(model, directory: Path) -> Path:
    path = directory / "final_cb.cbm"
    model.save_model(str(path))
    print(f"  ✔ saved {path.name}")
    return path

print("Model registry loaded ✓")


# ## 11. Cross-Validation (GroupKFold + OOF)
# #
# GroupKFold ensures entire wells stay within a fold.
# OOF predictions are used for Ridge stacking and Nelder-Mead ensemble.

# In[ ]:


def run_cv(
    train_df:      pd.DataFrame,
    feature_cols:  List[str],
    one_fold:      bool = False,
) -> Tuple[dict, dict, pd.DataFrame]:
    """
    Run 5-fold GroupKFold CV.
    Returns (oof_preds dict, best_iters dict, fold_metrics_df).
    """
    X      = train_df[feature_cols]
    y      = train_df["target"]
    groups = train_df["well"]
    n      = len(train_df)

    cv     = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(X, y, groups=groups))
    if one_fold:
        splits = splits[:1]
        print("DEBUG: running only fold 0")

    oof_preds  = {k: np.zeros(n, dtype=np.float64) for k in ACTIVE_MODELS}
    best_iters = {k: [] for k in ACTIVE_MODELS}
    fold_scores= {k: [] for k in ACTIVE_MODELS}

    section(log, f"CROSS-VALIDATION  ({N_SPLITS}-fold GroupKFold)")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        Xtr = X.iloc[tr_idx]; ytr = y.iloc[tr_idx]
        Xva = X.iloc[va_idx]; yva = y.iloc[va_idx]
        w_tr = train_df.iloc[tr_idx]["well"].nunique()
        w_va = train_df.iloc[va_idx]["well"].nunique()
        log.info(f"\n── Fold {fold+1}/{len(splits)}  "
                 f"(train={len(tr_idx):,}  val={len(va_idx):,}  "
                 f"wells_tr={w_tr}  wells_va={w_va}) ──")

        # ── LGB × 3 seeds ────────────────────────────────────────────
        for seed_idx, seed in enumerate(LGB_SEEDS):
            key = f"lgb{seed_idx}"
            p   = make_lgb_model(seed)
            dtr = lgb.Dataset(Xtr, label=ytr)
            dva = lgb.Dataset(Xva, label=yva, reference=dtr)
            m   = lgb.train(
                p, dtr, valid_sets=[dva],
                num_boost_round=p["n_estimators"],
                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                            lgb.log_evaluation(LOG_EVERY)],
            )
            preds = m.predict(Xva, num_iteration=m.best_iteration).astype(np.float64)
            oof_preds[key][va_idx] = preds
            best_iters[key].append(m.best_iteration)
            rmse = root_mean_squared_error(yva, preds)
            fold_scores[key].append(rmse)
            log.info(f"  LGB(s={seed}) iter={m.best_iteration:5d}  fold_RMSE={rmse:.5f}")

        # ── XGBoost ──────────────────────────────────────────────────
        key = "xgb"
        m   = make_xgb_model()
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=LOG_EVERY)
        preds = m.predict(Xva).astype(np.float64)
        oof_preds[key][va_idx] = preds
        bi = int(m.best_iteration)
        best_iters[key].append(bi)
        rmse = root_mean_squared_error(yva, preds)
        fold_scores[key].append(rmse)
        log.info(f"  XGB    iter={bi:5d}  fold_RMSE={rmse:.5f}")

        # ── CatBoost ─────────────────────────────────────────────────
        key = "cb"
        m   = make_cb_model()
        m.fit(Pool(Xtr.values, label=ytr.values),
              eval_set=Pool(Xva.values, label=yva.values),
              use_best_model=True)
        preds = m.predict(Xva.values).astype(np.float64)
        oof_preds[key][va_idx] = preds
        bi = int(m.best_iteration_)
        best_iters[key].append(bi)
        rmse = root_mean_squared_error(yva, preds)
        fold_scores[key].append(rmse)
        log.info(f"  CB     iter={bi:5d}  fold_RMSE={rmse:.5f}")

    # Summary
    section(log, "CV SUMMARY")
    rows = {k: fold_scores[k] + [np.mean(fold_scores[k]), np.std(fold_scores[k])]
            for k in ACTIVE_MODELS}
    idx  = [f"fold_{i+1}" for i in range(len(splits))] + ["mean", "std"]
    fm_df = pd.DataFrame(rows, index=idx)
    try:    log.info("\n" + fm_df.to_markdown(floatfmt=".5f"))
    except: log.info("\n" + fm_df.to_string(float_format=lambda v: f"{v:.5f}"))

    y_arr = y.to_numpy()
    for key in ACTIVE_MODELS:
        oof_rmse = root_mean_squared_error(y_arr, oof_preds[key])
        log.info(f"  OOF RMSE  {key:6s} = {oof_rmse:.5f}")

    return oof_preds, best_iters, fm_df

print("Cross-validation loaded ✓")


# ## 12. Ensemble (Ridge Stacking + Nelder-Mead fallback)

# In[ ]:


def optimise_ridge_ensemble(
    oof_preds:  dict,
    y_true:     np.ndarray,
) -> Tuple[Ridge, dict]:
    """
    Fit a positive-constrained Ridge meta-learner on OOF predictions.
    Returns (ridge_model, weights_dict).
    """
    keys     = list(oof_preds.keys())
    Sx       = np.column_stack([oof_preds[k] for k in keys])

    ridge    = Ridge(alpha=1.0, fit_intercept=False, positive=True)
    ridge.fit(Sx, y_true)

    oof_stk  = ridge.predict(Sx)
    r_stk    = root_mean_squared_error(y_true, oof_stk)
    r_avg    = root_mean_squared_error(y_true, Sx.mean(1))

    coef     = ridge.coef_
    coef_sum = max(coef.sum(), 1e-9)
    weights  = {k: float(coef[i] / coef_sum) for i, k in enumerate(keys)}

    section(log, "RIDGE STACKING ENSEMBLE")
    log.info(f"  Equal-avg OOF RMSE : {r_avg:.5f}")
    log.info(f"  Ridge stk OOF RMSE : {r_stk:.5f}  (Δ={r_avg - r_stk:+.5f})")
    for k, w in weights.items():
        log.info(f"    {k:8s}: {w:.4f}")
    return ridge, weights


def optimise_nelder_mead(
    oof_preds:  dict,
    y_true:     np.ndarray,
    n_restarts: int = 5,
) -> Tuple[dict, float]:
    """Nelder-Mead on OOF. Returns (weights_dict, oof_rmse)."""
    keys    = list(oof_preds.keys())
    mat     = np.column_stack([oof_preds[k] for k in keys])
    n       = len(keys)

    def objective(raw_w):
        w = np.maximum(raw_w, 0.0); w /= (w.sum() + 1e-12)
        return root_mean_squared_error(y_true, mat @ w)

    rng      = np.random.default_rng(SEED)
    best_res = None
    for trial in range(n_restarts):
        w0  = np.ones(n) / n if trial == 0 else rng.dirichlet(np.ones(n))
        res = minimize(objective, w0, method="Nelder-Mead",
                       options={"maxiter": 30_000, "xatol": 1e-12, "fatol": 1e-12})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    w      = np.maximum(best_res.x, 0.0); w /= (w.sum() + 1e-12)
    weights = {k: float(w[i]) for i, k in enumerate(keys)}
    rmse    = root_mean_squared_error(y_true, mat @ w)

    section(log, "NELDER-MEAD ENSEMBLE")
    for k, wv in weights.items():
        log.info(f"  {k:8s}: {wv:.4f}")
    log.info(f"  NM OOF RMSE : {rmse:.5f}")
    return weights, rmse


def apply_ensemble_ridge(
    test_preds: dict,
    ridge:      Ridge,
) -> np.ndarray:
    keys = list(test_preds.keys())
    St   = np.column_stack([test_preds[k] for k in keys])
    return ridge.predict(St).astype(np.float64)


def apply_ensemble_nm(
    test_preds: dict,
    weights:    dict,
) -> np.ndarray:
    keys = list(weights.keys())
    mat  = np.column_stack([test_preds[k] for k in keys])
    w    = np.array([weights[k] for k in keys])
    return (mat @ w).astype(np.float64)

print("Ensemble optimisation loaded ✓")


# ## 13. Final Training (full data)

# In[ ]:


def run_final_training(
    train_df:     pd.DataFrame,
    feature_cols: List[str],
    best_iters:   dict,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Retrain on full data at (mean_best_iter × FINAL_ITER_SCALE).
    Returns (models_dict, feature_importance_df).
    models_dict keys: lgb0, lgb1, lgb2, xgb, cb
    Values: booster/model objects ready for predict.
    """
    section(log, "FINAL TRAINING — FULL DATA")
    X  = train_df[feature_cols]
    y  = train_df["target"].values
    models = {}

    # LGB × 3 seeds
    lgb_fi = None
    for seed_idx, seed in enumerate(LGB_SEEDS):
        key    = f"lgb{seed_idx}"
        n_iter = max(50, int(round(np.mean(best_iters[key]) * FINAL_ITER_SCALE)))
        log.info(f"  {key}  n_iter={n_iter}")
        p = make_lgb_model(seed, n_est=n_iter)
        p.pop("n_estimators", None)
        ds = lgb.Dataset(X, label=y)
        m  = lgb.train(p, ds, num_boost_round=n_iter,
                       callbacks=[lgb.log_evaluation(LOG_EVERY)])
        save_lgb_model(m, ARTEFACT_DIR, key)
        models[key] = m
        if seed_idx == 0:
            lgb_fi = pd.DataFrame({
                "feature"   : feature_cols,
                "importance": m.feature_importance(importance_type="gain"),
            }).sort_values("importance", ascending=False).reset_index(drop=True)

    # XGBoost
    key    = "xgb"
    n_iter = max(50, int(round(np.mean(best_iters[key]) * FINAL_ITER_SCALE)))
    log.info(f"  {key}  n_iter={n_iter}")
    p = dict(XGB_PARAMS)
    p.pop("early_stopping_rounds", None)
    p["n_estimators"] = n_iter
    m = XGBRegressor(**p)
    m.fit(X, y, verbose=LOG_EVERY)
    save_xgb_model(m, ARTEFACT_DIR)
    models[key] = m

    # CatBoost
    key    = "cb"
    n_iter = max(50, int(round(np.mean(best_iters[key]) * FINAL_ITER_SCALE)))
    log.info(f"  {key}  n_iter={n_iter}")
    p = dict(CB_PARAMS)
    p.pop("od_type", None); p.pop("od_wait", None)
    p["iterations"] = n_iter
    m = CatBoostRegressor(**p)
    m.fit(Pool(X.values, label=y), verbose=LOG_EVERY)
    save_cb_model(m, ARTEFACT_DIR)
    models[key] = m

    if lgb_fi is not None:
        log.info("\nTop-25 features (LGB0 gain):")
        log.info("\n" + lgb_fi.head(25).to_string(index=False))

    return models, lgb_fi


def predict_all(models: dict, X: pd.DataFrame) -> dict:
    """Generate predictions from all models."""
    preds = {}
    for key, m in models.items():
        if key.startswith("lgb"):
            p = m.predict(X).astype(np.float64)
        elif key == "xgb":
            p = m.predict(X).astype(np.float64)
        else:  # cb
            p = m.predict(X.values).astype(np.float64)
        preds[key] = p
        log.info(f"  {key:8s} delta range: {p.min():.2f}–{p.max():.2f}")
    return preds

print("Final training loaded ✓")


# ## 14. Post-Processing
# #
# Two stages:
# 1. Fade-in correction: multiply delta by alpha × (1 - exp(-md_since/tau))
#    to smoothly transition from last_known_tvt at the start.
# 2. Savitzky-Golay smoothing per well for temporal continuity.

# In[ ]:


def search_postproc_params(
    train_df:  pd.DataFrame,
    final_oof: np.ndarray,
    y_true:    np.ndarray,
) -> Tuple[float, Optional[float]]:
    """Grid search alpha × tau on OOF delta predictions."""
    base    = train_df["last_known_tvt"].values
    y_abs   = y_true + base  # absolute TVT

    best_cfg, best_r = (None, None), np.inf
    alphas = np.arange(0.60, 1.05, 0.05)
    taus   = [None, 30.0, 60.0, 120.0, 250.0, 500.0]

    for alpha in alphas:
        for tau in taus:
            d = final_oof.copy()
            if tau is not None:
                d *= (1.0 - np.exp(-np.maximum(train_df["md_since"].values, 0.0) / tau))
            d *= alpha
            r = root_mean_squared_error(y_abs, base + d)
            if r < best_r:
                best_r   = r
                best_cfg = (alpha, tau)

    log.info(f"PostProc grid-search:  alpha={best_cfg[0]:.2f}  tau={best_cfg[1]}  "
             f"abs TVT RMSE={best_r:.4f}")
    return best_cfg


def apply_postproc(
    df:    pd.DataFrame,
    delta: np.ndarray,
    alpha: float,
    tau:   Optional[float],
) -> np.ndarray:
    d = delta.copy()
    if tau is not None:
        d *= (1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau))
    return d * alpha


def sg_smooth_per_well(
    df:     pd.DataFrame,
    delta:  np.ndarray,
    sg_w:   int = 17,
    sg_p:   int = 3,
) -> np.ndarray:
    """Savitzky-Golay smooth per well (operates on delta, preserves dtype)."""
    result = delta.copy()
    df_tmp = df[["well"]].copy()
    df_tmp["_delta"] = delta
    df_tmp["_idx"]   = np.arange(len(df))

    for well, g in df_tmp.groupby("well", sort=False):
        v   = g["_delta"].values
        n   = len(v)
        wl  = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        result[g["_idx"].values] = v

    return result.astype(np.float64)

print("Post-processing loaded ✓")


# ## 15. Inference & Submission

# In[ ]:


def build_submission(
    test_df:     pd.DataFrame,
    delta_pred:  np.ndarray,
    sample_sub:  pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    abs_tvt  = delta_pred + test_df["last_known_tvt"].to_numpy()
    pred_map = dict(zip(test_df["id"], abs_tvt))

    sub = sample_sub.copy()
    sub["tvt"] = sub["id"].map(pred_map)
    miss = sub["tvt"].isna().sum()
    if miss > 0:
        log.warning(f"{miss} rows missing prediction — filling with global mean")
        fb = float(test_df["last_known_tvt"].mean())
        sub["tvt"] = sub["tvt"].fillna(fb)

    # ── Edge S: round-to-grid post-process ────────────────────────────────
    # dTVT = 0.01 ft grid 上の階段関数 (= 773/773 wells で 100% 確証)
    # 出典: hengck23 (Kaggle Grandmaster) discussion 697431 msg#8
    #       + docs/research/discussions-deep-v2.dense.md §2.3.1
    # 連続予測を 0.01 grid に snap → MAE が round 量だけ確実に減る
    EDGE_S_ROUND_DECIMALS = 2  # 0.01 ft grid
    if EDGE_S_ROUND_DECIMALS is not None:
        final_tvt_pre = sub["tvt"].astype(float).values.copy()
        sub["tvt"] = np.round(sub["tvt"].astype(float).values, EDGE_S_ROUND_DECIMALS)
        diff = sub["tvt"].values - final_tvt_pre
        log.info(f"[Edge S] applied round-to-{EDGE_S_ROUND_DECIMALS}-decimals")
        log.info(f"[Edge S]   diff abs mean: {np.abs(diff).mean():.6f}")
        log.info(f"[Edge S]   diff abs max:  {np.abs(diff).max():.6f}")
        log.info(f"[Edge S]   unique tvt count: {len(np.unique(sub['tvt']))}")

    sub[["id", "tvt"]].to_csv(output_path, index=False)
    log.info(f"Submission → {output_path}  ({len(sub):,} rows)")
    log.info(f"\n{sub.head(8).to_string(index=False)}")
    return sub

print("Inference & submission loaded ✓")


# ## 16. Main Dispatch

# In[ ]:


def _get_or_build_train_df():
    try:
        log.info("Loading cached train dataset …")
        train_df     = am.load_df(am.TRAIN_DF)
        train_df     = assign_groups(train_df)
        feature_cols = am.load_json(am.FEATURES_LIST)
        return train_df, feature_cols
    except FileNotFoundError:
        pass
    log.info("Cache miss — building train features …")
    with timer(log, "train feature engineering"):
        train_df = build_dataset(
            TRAIN_DIR, is_train=True,
            max_wells=DEBUG_MAX_WELLS,
            n_jobs=NCPU,
        )
    train_df = assign_groups(train_df)
    am.save_df(train_df, am.TRAIN_DF)
    feature_cols = get_feature_columns(train_df)
    am.save_json(feature_cols, am.FEATURES_LIST)
    return train_df, feature_cols


def _get_or_build_test_df() -> pd.DataFrame:
    try:
        log.info("Loading cached test dataset …")
        return am.load_df(am.TEST_DF)
    except FileNotFoundError:
        pass
    with timer(log, "test feature engineering"):
        test_df = build_dataset(
            TEST_DIR, is_train=False,
            max_wells=DEBUG_MAX_WELLS,
            n_jobs=NCPU,
        )
    am.save_df(test_df, am.TEST_DF)
    return test_df


# ─────────────────────────────────────────────────────────────────────────────
section(log, f"ROGII HYBRID PIPELINE  |  MODE={MODE}")
log.info(f"Data dir   : {DATA_DIR}")
log.info(f"Artefacts  : {ARTEFACT_DIR}")
log.info(f"Models     : {ACTIVE_MODELS}")
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────────────

if MODE == "features_only":
    train_df, feature_cols = _get_or_build_train_df()
    log.info(f"Train: {train_df.shape}  features: {len(feature_cols)}")
    test_df = _get_or_build_test_df()
    log.info(f"Test : {test_df.shape}")


elif MODE == "cv":
    train_df, feature_cols = _get_or_build_train_df()
    oof_preds, best_iters, fm_df = run_cv(
        train_df, feature_cols, one_fold=DEBUG_ONE_FOLD
    )
    oof_df = train_df[["well", "id", "target"]].copy()
    for k, p in oof_preds.items(): oof_df[f"oof_{k}"] = p
    am.save_df(oof_df, am.OOF_PREDICTIONS)
    am.save_json(best_iters, am.BEST_ITERS)
    am.save_df(fm_df.reset_index(), am.FOLD_METRICS)
    log.info("CV complete. OOF + best_iters saved.")


elif MODE == "ensemble_only":
    oof_df    = am.load_df(am.OOF_PREDICTIONS)
    y_true    = oof_df["target"].to_numpy()
    oof_preds = {k: oof_df[f"oof_{k}"].to_numpy()
                 for k in ACTIVE_MODELS if f"oof_{k}" in oof_df.columns}
    ridge, ridge_w = optimise_ridge_ensemble(oof_preds, y_true)
    nm_w, nm_rmse  = optimise_nelder_mead(oof_preds, y_true)
    am.save(ridge, am.RIDGE_META)
    am.save_json(ridge_w, "ridge_weights")
    am.save_json(nm_w,    am.ENSEMBLE_WEIGHTS)


elif MODE == "train":
    # ── Step 1: Build / load features ──────────────────────────────────
    train_df, feature_cols = _get_or_build_train_df()
    y = train_df["target"]
    log.info(f"Train features: {train_df.shape}  #features: {len(feature_cols)}")

    # ── Step 2: Cross-validation ────────────────────────────────────────
    oof_preds, best_iters, fm_df = run_cv(
        train_df, feature_cols, one_fold=DEBUG_ONE_FOLD
    )
    oof_df = train_df[["well", "id", "target"]].copy()
    for k, p in oof_preds.items(): oof_df[f"oof_{k}"] = p
    am.save_df(oof_df, am.OOF_PREDICTIONS)
    am.save_json(best_iters, am.BEST_ITERS)
    am.save_df(fm_df.reset_index(), am.FOLD_METRICS)

    # ── Step 3: Ensemble optimisation on OOF ────────────────────────────
    y_arr          = y.to_numpy()
    ridge, ridge_w = optimise_ridge_ensemble(oof_preds, y_arr)
    nm_w, nm_rmse  = optimise_nelder_mead(oof_preds, y_arr)
    am.save(ridge, am.RIDGE_META)
    am.save_json(ridge_w, "ridge_weights")
    am.save_json(nm_w,    am.ENSEMBLE_WEIGHTS)

    # Determine which ensemble is better on OOF
    keys    = list(oof_preds.keys())
    Sx_oof  = np.column_stack([oof_preds[k] for k in keys])
    r_ridge = root_mean_squared_error(y_arr, ridge.predict(Sx_oof))
    r_nm    = nm_rmse
    use_ridge = r_ridge <= r_nm
    log.info(f"Ridge OOF={r_ridge:.5f}  NM OOF={r_nm:.5f}  → using {'Ridge' if use_ridge else 'NM'}")
    final_oof_delta = ridge.predict(Sx_oof) if use_ridge else (Sx_oof @ np.array([nm_w[k] for k in keys]))

    # ── Step 4: Post-processing grid search ─────────────────────────────
    best_alpha, best_tau = search_postproc_params(train_df, final_oof_delta, y_arr)
    am.save_json({"alpha": float(best_alpha), "tau": best_tau, "use_ridge": use_ridge},
                 "postproc_params")

    # ── Step 5: Final training ───────────────────────────────────────────
    models, fi_df = run_final_training(train_df, feature_cols, best_iters)
    if fi_df is not None:
        am.save_df(fi_df, am.FEATURE_IMPORTANCE)

    # ── Step 6: Test predictions ─────────────────────────────────────────
    test_df = _get_or_build_test_df()
    log.info(f"Test features: {test_df.shape}")
    section(log, "INFERENCE")
    test_preds = predict_all(models, test_df[feature_cols])

    if use_ridge:
        test_delta = apply_ensemble_ridge(test_preds, ridge)
    else:
        test_delta = apply_ensemble_nm(test_preds, nm_w)

    # ── Step 7: Post-processing ──────────────────────────────────────────
    test_delta_pp = apply_postproc(test_df, test_delta, best_alpha, best_tau)
    test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)

    # Spike removal: clip per-well to [p1, p99] of training delta range
    train_delta_range = (float(y_arr.min()), float(y_arr.max()))
    test_delta_sg = np.clip(test_delta_sg,
                            train_delta_range[0] * 1.5,
                            train_delta_range[1] * 1.5)

    # ── Step 8: Submission ───────────────────────────────────────────────
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    build_submission(test_df, test_delta_sg, sample, OUTPUT_DIR / "submission.csv")


elif MODE == "infer":
    feature_cols   = am.load_json(am.FEATURES_LIST)
    ridge          = am.load(am.RIDGE_META)
    nm_w           = am.load_json(am.ENSEMBLE_WEIGHTS)
    pp_params      = am.load_json("postproc_params")
    best_alpha     = float(pp_params["alpha"])
    best_tau       = pp_params.get("tau")
    use_ridge      = bool(pp_params.get("use_ridge", True))

    # ← CHANGED: always rebuild test features, never use cached test_df
    with timer(log, "test feature engineering"):
        test_df = build_dataset(TEST_DIR, is_train=False, n_jobs=NCPU)

    # Load model files
    import lightgbm as _lgb2
    from xgboost import XGBRegressor as _XGB2
    from catboost import CatBoostRegressor as _CB2
    models = {}
    for seed_idx in range(len(LGB_SEEDS)):
        key = f"lgb{seed_idx}"
        b   = _lgb2.Booster(model_file=str(ARTEFACT_DIR / f"final_lgb_{key}.txt"))
        models[key] = b
    m_xgb = _XGB2(); m_xgb.load_model(str(ARTEFACT_DIR / "final_xgb.json"))
    models["xgb"] = m_xgb
    m_cb = _CB2();  m_cb.load_model(str(ARTEFACT_DIR / "final_cb.cbm"))
    models["cb"]  = m_cb


    test_preds = predict_all(models, test_df[feature_cols])
    use_ridge = False
    if use_ridge:
        test_delta = apply_ensemble_ridge(test_preds, ridge)
    else:
        test_delta = apply_ensemble_nm(test_preds, nm_w)

    # ── Edge R: test-time online learning (= continued training on test visible) ──
    # 出典: discussion 698002 (online 10.953 vs no-online 11.323 = -0.370 ft)
    # leak guard: visible TVT_input のみ参照、 hidden TVT に絶対触れない
    if EDGE_R_ENABLE:
        section(log, "Edge R: test-time online learning")
        try:
            with timer(log, "Edge R visible dataset build"):
                visible_df = build_visible_dataset(
                    TEST_DIR,
                    max_wells=DEBUG_MAX_WELLS,
                    tail_k=EDGE_R_VISIBLE_TAIL_K,
                    min_visible_rows=EDGE_R_MIN_VISIBLE_ROWS,
                    n_jobs=NCPU,
                )
            with timer(log, "Edge R continued training + blend"):
                test_delta, edge_r_diag = edge_r_apply(
                    test_df, visible_df, models, feature_cols, nm_w,
                    test_delta_base=test_delta, log=log,
                )
            log.info(f"[Edge R] diagnostics: {edge_r_diag}")
        except Exception as _er_e:
            log.error(f"[Edge R] pipeline failed ({type(_er_e).__name__}: {_er_e}); "
                      f"falling back to base test_delta")

    test_delta_pp = apply_postproc(test_df, test_delta, best_alpha, best_tau)
    test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    build_submission(test_df, test_delta_sg, sample, OUTPUT_DIR / "submission.csv")


else:
    log.error(f"Unknown MODE='{MODE}'. "
              f"Choose from: train | infer | cv | features_only | ensemble_only")


section(log, "DONE")


# In[ ]:




