#!/usr/bin/env python
# coding: utf-8

# # ROGII exp006 — TabICL 6th base + PF-lite (LB target: 9 帯)
#
# Self-contained Kaggle GPU script that extends exp005 with:
#   1. **TabICL** (transformer in-context tabular regressor) as a 6th base model
#      via 5-fold OOF on `train_df.parquet` + test prediction. The OOF is then
#      stacked with the 5 existing karnakbaev base OOF (lgb0/lgb1/lgb2/xgb/cb)
#      and Ridge meta is **re-fit** in-kernel on the 6-base matrix.
#   2. **PF-lite** features (state-less likelihood-weighted candidate-TVT
#      ensemble) — implemented as dormant helpers in v1 (= test-side helpers
#      defined but not yet plugged into GBM stack since the karnakbaev train_df
#      lacks PF-lite columns). Activated in v2 once train-side PF-lite is added.
#
# Pipeline:
#   a. Build test features from live `/kaggle/input/.../test/` directory.
#   b. Load karnakbaev artefacts: LGB×3 + XGB + CB boosters, train_df.parquet,
#      oof_predictions.parquet, features.json.
#   c. Install TabICL wheel + load ckpt (private dataset, --no-deps).
#   d. LGB quick-fit on train_df → top-50 features for TabICL.
#   e. TabICL 5-fold OOF on train_df + test prediction (GroupKFold(5) by `well`).
#   f. Build 6-base OOF stack (5 karnakbaev OOF + TabICL OOF) → Ridge meta re-fit.
#   g. Apply Ridge meta to 6-base test predictions → fade-in postproc → SG smooth.
#
# License / source attribution:
#   - karnakbaev artifacts:  `karnakbaevarthur/rogii-code-helper-dataset` (apache-2.0)
#   - TabICL package + ckpt: `thermostatic/rogii-tabicl-v2-public-assets` (apache-2.0;
#                            upstream https://github.com/soda-inria/tabicl)
#   - TabICL stack pipeline: inspired by needless090/score-10-081-score-lb-32-rank
#                            (public Kaggle, LB 10.081 / rank 32)
#   - PF-lite helpers:       inspired by pilkwang/rogii-eda-v4-same-matrix-super-stack
#                            (public Kaggle; v1 keeps PF-lite functions dormant)
#   - Pipeline base:         karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline
#                            (public Kaggle, LB 10.784); reused under fair-use sharing.
#   - Original solution credits romantamrazov/rogii-super-solution-lb-top-3 ideas.
#
# This file is for the team's exp006 entry only.

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
# MODE: "train" | "infer" | "cv" | "features_only" | "ensemble_only" | "infer_tabicl"
#   infer_tabicl: exp006 path = load karnakbaev OOF + TabICL 5-fold OOF
#                 → Ridge meta re-fit on 6-base → final blend.
MODE = "infer_tabicl"

# Active base models (for inference). TabICL is added on top in infer_tabicl.
ACTIVE_MODELS = ["lgb0", "lgb1", "lgb2", "xgb", "cb"]

# ─── TabICL knobs ───────────────────────────────────────────────────────────
# TabICL hyper-parameters; tuned conservatively to fit within 8h GPU cap.
# v1: 1 seed × n_est=4 × ctx=4096 × 5-fold ≈ 30-50 min on T4.
TABICL_ENABLE        = True
TABICL_N_FEATURES    = 50      # top-K by LGB quick-fit feature_importances
TABICL_CTX           = 4096    # in-context training set size per fold/seed
TABICL_N_ESTIMATORS  = 4       # TabICL ensemble member count per fit
TABICL_SEEDS         = [42]    # v1: single seed; expand to [0,1,2,3,4] in v2
TABICL_CHUNK         = 50000   # predict chunk size (memory)
TABICL_USE_AMP       = "auto"  # mixed precision (mandatory for T4 16GB)
TABICL_BATCH_SIZE    = 4
TABICL_QUICK_LGB_N   = 300     # n_estimators for the quick LGB used to rank features
TABICL_QUICK_SAMPLE  = 200_000 # rows sampled for quick LGB (keeps it fast)
TABICL_HARD_TIMEOUT_S = 6 * 3600  # 6h cap inside TabICL phase (rest budget for postproc)

# ─── Stacking knob ──────────────────────────────────────────────────────────
# When TabICL OOF is available, re-fit Ridge meta on (5 karnakbaev base + TabICL).
# Fall back to karnakbaev's Nelder-Mead 5-base blend if TabICL fails.
STACK_REFIT_RIDGE = True

# Debug flags
DEBUG_MAX_WELLS    = None   # set to e.g. 10 for fast iteration
DEBUG_ONE_FOLD     = False
DEBUG_INSPECT_PF   = False
DEBUG_INSPECT_BEAM = False

# ─── Paths ───────────────────────────────────────────────────────────────────
_KAGGLE_CANDIDATES = [
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
]
DATA_DIR = next((p for p in _KAGGLE_CANDIDATES if (p / "test").exists()),
                Path("../../data").resolve())
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

if MODE in ("train", "cv", "features_only", "ensemble_only"):
    ARTEFACT_DIR = Path("/kaggle/working/artefacts")
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

OUTPUT_DIR = Path("/kaggle/working")
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


# ─── PF-lite (state-less candidate-TVT ensemble) ─────────────────────────────
# Adapted from pilkwang/rogii-eda-v4-same-matrix-super-stack (public Kaggle)
#   `weighted_candidate_tvt_features` (line ~1932-2003) and helpers.
#
# v1 status: dormant. These helpers are defined but not connected to the
# karnakbaev train_df (which lacks PF-lite columns), so they cannot affect the
# GBM stack. v2 will compute PF-lite for both train and test and add them to
# the TabICL feature pool.

def pf_lite_feature_names() -> List[str]:
    return [
        "pf_lite_tvt", "pf_lite_delta", "pf_lite_std",
        "pf_lite_weight_sum", "pf_lite_candidate_count",
        "pf_lite_vs_dense", "pf_lite_vs_plane", "pf_lite_vs_sc",
        "pf_lite_vs_beam_cons", "pf_lite_gr_abs_resid",
    ]


def typewell_gr_at_tvt(tw_tvt: np.ndarray, tw_gr: np.ndarray,
                       tvt_values: np.ndarray) -> np.ndarray:
    """Linear interpolate GR at given TVT positions; out-of-range → NaN."""
    if len(tw_tvt) < 2:
        return np.full(len(tvt_values), np.nan)
    return np.interp(np.asarray(tvt_values, dtype=float),
                     tw_tvt, tw_gr, left=np.nan, right=np.nan)


def weighted_candidate_tvt_features(
    tw_tvt:           np.ndarray,
    tw_gr:            np.ndarray,
    tail_gr:          np.ndarray,
    tail_z:           np.ndarray,
    candidate_tvt:    Dict[str, np.ndarray],
    last_known_tvt:   float,
    dense_ancc:       Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Likelihood-weighted ensemble of candidate TVT estimates.

    Returns 10 features per row (6 of which are activated in v1 if connected):
      pf_lite_tvt, pf_lite_delta, pf_lite_std, pf_lite_weight_sum,
      pf_lite_candidate_count, pf_lite_vs_dense, pf_lite_vs_plane,
      pf_lite_vs_sc, pf_lite_vs_beam_cons, pf_lite_gr_abs_resid.
    """
    names = list(candidate_tvt)
    n = len(tail_gr)
    if not names:
        return {nm: np.full(n, np.nan, dtype=np.float32)
                for nm in pf_lite_feature_names()}

    cand = np.column_stack(
        [np.asarray(candidate_tvt[nm], dtype=float) for nm in names])
    finite = np.isfinite(cand)
    tw_at_cand = typewell_gr_at_tvt(
        tw_tvt, tw_gr, cand.reshape(-1)).reshape(cand.shape)
    gr_abs = np.abs(np.asarray(tail_gr, dtype=float)[:, None] - tw_at_cand)
    gr_penalty = np.clip(
        np.nan_to_num(gr_abs, nan=60.0, posinf=60.0, neginf=60.0) / 20.0,
        0.0, 6.0)
    weights = np.exp(-gr_penalty) * finite

    if dense_ancc is not None:
        d = np.asarray(dense_ancc, dtype=float)
        d_valid = np.isfinite(d)
        if d_valid.any():
            ancc_abs = np.abs((cand + np.asarray(tail_z, dtype=float)[:, None])
                              - d[:, None])
            ancc_penalty = np.clip(
                np.nan_to_num(ancc_abs, nan=120.0, posinf=120.0, neginf=120.0)
                / 35.0, 0.0, 6.0)
            d_factor = 0.20 + 0.80 * np.exp(-ancc_penalty)
            d_factor = np.where(d_valid[:, None], d_factor, 1.0)
            weights *= d_factor

    denom = weights.sum(axis=1)
    safe  = np.where(denom <= 1e-8, 1.0, denom)
    pf_tvt = np.sum(np.where(finite, cand, 0.0) * weights, axis=1) / safe

    # Fallback to candidate median when weights collapse
    cand_masked = np.where(finite, cand, np.nan)
    finite_count = finite.sum(axis=1)
    fallback_tvt = np.full(n, np.nan, dtype=float)
    fallback_std = np.full(n, np.nan, dtype=float)
    rows = finite_count > 0
    if rows.any():
        fallback_tvt[rows] = np.nanmedian(cand_masked[rows], axis=1)
        fallback_std[rows] = np.nanstd(cand_masked[rows], axis=1)
    low = denom <= 1e-8
    pf_tvt[low] = fallback_tvt[low]

    var = np.sum(((cand - pf_tvt[:, None]) ** 2) * weights, axis=1) / safe
    pf_std = np.sqrt(np.maximum(var, 0.0))
    pf_std[low] = fallback_std[low]

    gr_masked = np.where(finite, gr_abs, np.inf)
    best_gr_abs = np.min(gr_masked, axis=1)
    best_gr_abs[~np.isfinite(best_gr_abs)] = np.nan

    out: Dict[str, np.ndarray] = {
        "pf_lite_tvt"            : pf_tvt.astype(np.float32),
        "pf_lite_delta"          : (pf_tvt - float(last_known_tvt)).astype(np.float32),
        "pf_lite_std"            : pf_std.astype(np.float32),
        "pf_lite_weight_sum"     : denom.astype(np.float32),
        "pf_lite_candidate_count": finite.sum(axis=1).astype(np.float32),
        "pf_lite_gr_abs_resid"   : best_gr_abs.astype(np.float32),
    }

    def _delta_from(name: str) -> np.ndarray:
        arr = candidate_tvt.get(name)
        if arr is None:
            return np.full(n, np.nan, dtype=np.float32)
        return (pf_tvt - np.asarray(arr, dtype=float)).astype(np.float32)

    out["pf_lite_vs_dense"]     = _delta_from("dense")
    out["pf_lite_vs_plane"]     = _delta_from("plane")
    out["pf_lite_vs_sc"]        = _delta_from("selfcorr")
    out["pf_lite_vs_beam_cons"] = _delta_from("beam_cons")
    return out


print("PF-lite helpers loaded ✓ (v1: dormant, not yet connected to GBM stack)")


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

    test_delta_pp = apply_postproc(test_df, test_delta, best_alpha, best_tau)
    test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    build_submission(test_df, test_delta_sg, sample, OUTPUT_DIR / "submission.csv")


elif MODE == "infer_tabicl":
    # ── exp006 path: TabICL 6th base + Ridge re-fit on 6-base OOF ───────────
    feature_cols   = am.load_json(am.FEATURES_LIST)
    pp_params      = am.load_json("postproc_params")
    best_alpha     = float(pp_params["alpha"])
    best_tau       = pp_params.get("tau")

    section(log, "exp006: build test features (live test wells)")
    with timer(log, "test feature engineering"):
        test_df = build_dataset(TEST_DIR, is_train=False,
                                max_wells=DEBUG_MAX_WELLS, n_jobs=NCPU)
    log.info(f"Test features: {test_df.shape}")

    # ── Step A: load karnakbaev base boosters & predict on test ─────────────
    section(log, "exp006: load 5 base boosters + predict test")
    import lightgbm as _lgb2
    from xgboost import XGBRegressor as _XGB2
    from catboost import CatBoostRegressor as _CB2
    base_models = {}
    for seed_idx in range(len(LGB_SEEDS)):
        key = f"lgb{seed_idx}"
        base_models[key] = _lgb2.Booster(
            model_file=str(ARTEFACT_DIR / f"final_lgb_{key}.txt"))
    m_xgb = _XGB2(); m_xgb.load_model(str(ARTEFACT_DIR / "final_xgb.json"))
    base_models["xgb"] = m_xgb
    m_cb = _CB2(); m_cb.load_model(str(ARTEFACT_DIR / "final_cb.cbm"))
    base_models["cb"] = m_cb
    test_preds = predict_all(base_models, test_df[feature_cols])
    log.info(f"  base test_preds keys: {list(test_preds.keys())}")
    for k, v in test_preds.items():
        log.info(f"    {k}: shape={v.shape}  range=[{v.min():.3f}, {v.max():.3f}]")

    # ── Step B: load karnakbaev train_df + OOF + run TabICL ─────────────────
    tabicl_oof = None
    tabicl_test = None
    tabicl_ok = False
    train_df_loaded = False

    if TABICL_ENABLE:
        section(log, "exp006: load karnakbaev train_df + OOF for TabICL")
        try:
            train_df_path = ARTEFACT_DIR / "train_df.parquet"
            oof_pred_path = ARTEFACT_DIR / "oof_predictions.parquet"
            if not train_df_path.exists():
                raise FileNotFoundError(f"train_df missing: {train_df_path}")
            if not oof_pred_path.exists():
                raise FileNotFoundError(f"oof_predictions missing: {oof_pred_path}")

            with timer(log, "load train_df.parquet"):
                train_df_kb = pd.read_parquet(train_df_path)
            log.info(f"  train_df: shape={train_df_kb.shape}")
            with timer(log, "load oof_predictions.parquet"):
                oof_df_kb = pd.read_parquet(oof_pred_path)
            log.info(f"  oof_df: shape={oof_df_kb.shape}  cols={list(oof_df_kb.columns)[:10]}")

            # Sanity: required columns
            required = set(feature_cols) | {"target", "well", "id"}
            missing  = required - set(train_df_kb.columns)
            if missing:
                raise RuntimeError(
                    f"train_df is missing {len(missing)} required cols, e.g. {sorted(list(missing))[:5]}")

            # Align OOF to train_df row order via id
            if "id" in oof_df_kb.columns:
                merge_keys = ["id"]
            elif {"well", "target"}.issubset(oof_df_kb.columns):
                merge_keys = ["well", "target"]
            else:
                merge_keys = None

            if merge_keys is not None:
                # Reindex OOF to train_df row order
                _tmp = train_df_kb[merge_keys].merge(
                    oof_df_kb, on=merge_keys, how="left", suffixes=("", "_oof"))
                kb_oof_cols = [f"oof_{k}" for k in ACTIVE_MODELS]
                missing_oof = [c for c in kb_oof_cols if c not in _tmp.columns]
                if missing_oof:
                    raise RuntimeError(f"OOF parquet missing cols: {missing_oof}")
                kb_oof = {k: _tmp[f"oof_{k}"].to_numpy(np.float32)
                          for k in ACTIVE_MODELS}
                kb_oof_nan = {k: int(np.isnan(v).sum()) for k, v in kb_oof.items()}
                log.info(f"  karnakbaev OOF NaN counts: {kb_oof_nan}")
                if any(n > 0 for n in kb_oof_nan.values()):
                    log.warning("OOF has NaN after merge — fallback to NM blend")
                    raise RuntimeError("kb_oof_nan_after_merge")
            else:
                raise RuntimeError("Cannot align OOF parquet to train_df rows (no id/well-target)")

            # ── Step C: install TabICL wheel + load ckpt ────────────────────
            section(log, "exp006: install TabICL package + load ckpt")
            tabicl_root_candidates = [
                Path("/kaggle/input/rogii-tabicl-v2-public-assets"),
                Path("/kaggle/input/datasets/thermostatic/rogii-tabicl-v2-public-assets"),
                Path("/kaggle/input/thermostatic/rogii-tabicl-v2-public-assets"),
            ]
            tabicl_root = next((p for p in tabicl_root_candidates if p.exists()), None)
            if tabicl_root is None:
                # last-ditch: glob search
                hits = list(Path("/kaggle/input").rglob("tabicl-*.whl"))
                if hits:
                    tabicl_root = hits[0].parent
            if tabicl_root is None:
                raise FileNotFoundError(
                    "TabICL dataset not attached — expected slug "
                    "`thermostatic/rogii-tabicl-v2-public-assets`")
            log.info(f"  tabicl_root: {tabicl_root}")
            wheels = sorted(tabicl_root.rglob("tabicl-*.whl"))
            ckpts  = sorted(tabicl_root.rglob("tabicl-regressor*.ckpt"))
            if not wheels:
                raise FileNotFoundError("TabICL wheel not found under tabicl_root")
            if not ckpts:
                raise FileNotFoundError("TabICL ckpt not found under tabicl_root")
            wheel_path = wheels[0]
            ckpt_path  = ckpts[0]
            log.info(f"  wheel: {wheel_path}")
            log.info(f"  ckpt : {ckpt_path}")
            _cmd = [sys.executable, "-m", "pip", "install",
                    "--no-index", "--no-deps", str(wheel_path)]
            log.info(f"  install: {' '.join(_cmd)}")
            subprocess.run(_cmd, check=True)
            from tabicl import TabICLRegressor

            # ── Step D: LGB quick-fit → top-K features for TabICL ───────────
            section(log, "exp006: LGB quick-fit for TabICL feature selection")
            X_kb = train_df_kb[feature_cols].copy()
            y_kb = train_df_kb["target"].to_numpy(np.float32)
            g_kb = train_df_kb["well"].to_numpy()
            rng_q = np.random.RandomState(42)
            n_q   = min(TABICL_QUICK_SAMPLE, len(train_df_kb))
            idx_q = rng_q.choice(len(train_df_kb), size=n_q, replace=False)
            try:
                _quick = lgb.LGBMRegressor(
                    n_estimators=TABICL_QUICK_LGB_N, learning_rate=0.05,
                    num_leaves=63, n_jobs=-1, verbosity=-1,
                    device_type="gpu" if _GPU else "cpu")
            except Exception:
                _quick = lgb.LGBMRegressor(
                    n_estimators=TABICL_QUICK_LGB_N, learning_rate=0.05,
                    num_leaves=63, n_jobs=-1, verbosity=-1)
            with timer(log, "quick LGB fit (200k rows)"):
                _quick.fit(X_kb.iloc[idx_q].values, y_kb[idx_q])
            _imp = pd.Series(_quick.feature_importances_,
                             index=feature_cols).sort_values(ascending=False)
            tabicl_feats = _imp.head(TABICL_N_FEATURES).index.tolist()
            log.info(f"  top-{TABICL_N_FEATURES} features: {tabicl_feats[:5]} ...")

            # ── Step E: TabICL 5-fold OOF ───────────────────────────────────
            section(log, "exp006: TabICL 5-fold OOF + test prediction")
            X_top  = X_kb[tabicl_feats].values.astype(np.float32)
            Xt_top = test_df[tabicl_feats].values.astype(np.float32)
            cv     = GroupKFold(n_splits=5)
            splits = list(cv.split(X_top, y_kb, g_kb))
            tabicl_oof = np.zeros(len(X_top), dtype=np.float32)
            tabicl_test = np.zeros(len(Xt_top), dtype=np.float32)
            n_seeds = len(TABICL_SEEDS)
            tabicl_t0 = time.perf_counter()

            for fold, (tr_idx, va_idx) in enumerate(splits):
                fold_va_pred = np.zeros(len(va_idx), dtype=np.float32)
                fold_test    = np.zeros(len(Xt_top), dtype=np.float32)
                for sd in TABICL_SEEDS:
                    elapsed = time.perf_counter() - tabicl_t0
                    if elapsed > TABICL_HARD_TIMEOUT_S:
                        log.warning(
                            f"  fold {fold} seed {sd}: TabICL hard timeout "
                            f"({elapsed:.0f}s > {TABICL_HARD_TIMEOUT_S}s) — break")
                        break
                    ctx_n  = min(TABICL_CTX, len(tr_idx))
                    ctx_idx = np.random.RandomState(sd*1000 + fold).choice(
                        tr_idx, size=ctx_n, replace=False)
                    X_ctx = X_top[ctx_idx]; y_ctx = y_kb[ctx_idx]
                    X_va  = X_top[va_idx]
                    t_fit = time.perf_counter()
                    reg = TabICLRegressor(
                        model_path=str(ckpt_path), device="cuda",
                        random_state=sd, n_estimators=TABICL_N_ESTIMATORS,
                        n_jobs=1, verbose=False,
                        use_amp=TABICL_USE_AMP, batch_size=TABICL_BATCH_SIZE)
                    reg.fit(X_ctx, y_ctx)
                    pred_va = np.empty(len(va_idx), dtype=np.float32)
                    for i in range(0, len(va_idx), TABICL_CHUNK):
                        end = min(i + TABICL_CHUNK, len(va_idx))
                        pred_va[i:end] = reg.predict(X_va[i:end]).astype(np.float32)
                    fold_va_pred += pred_va / n_seeds
                    pred_te = np.empty(len(Xt_top), dtype=np.float32)
                    for i in range(0, len(Xt_top), TABICL_CHUNK):
                        end = min(i + TABICL_CHUNK, len(Xt_top))
                        pred_te[i:end] = reg.predict(Xt_top[i:end]).astype(np.float32)
                    fold_test += pred_te / n_seeds
                    log.info(f"  fold{fold} seed{sd}: "
                             f"[{time.perf_counter()-t_fit:.1f}s]  "
                             f"va_range=[{pred_va.min():.3f},{pred_va.max():.3f}]")
                tabicl_oof[va_idx] = fold_va_pred
                tabicl_test += fold_test / 5
                rmse_fold = root_mean_squared_error(y_kb[va_idx], fold_va_pred)
                log.info(f"  fold{fold} OOF RMSE: {rmse_fold:.4f}")

            tabicl_overall = root_mean_squared_error(y_kb, tabicl_oof)
            log.info(f"TabICL OOF RMSE: {tabicl_overall:.4f}")
            log.info(f"TabICL test pred: shape={tabicl_test.shape} "
                     f"range=[{tabicl_test.min():.3f},{tabicl_test.max():.3f}]")
            tabicl_ok = True
            train_df_loaded = True

            # ── Step F: build 6-base OOF stack & re-fit Ridge ──────────────
            if STACK_REFIT_RIDGE:
                section(log, "exp006: 6-base Ridge meta re-fit")
                stack_keys = ACTIVE_MODELS + ["tabicl"]
                Sx = np.column_stack(
                    [kb_oof[k] for k in ACTIVE_MODELS] + [tabicl_oof])
                St = np.column_stack(
                    [test_preds[k] for k in ACTIVE_MODELS] + [tabicl_test])
                ridge_ex = Ridge(alpha=1.0, fit_intercept=False, positive=True)
                ridge_ex.fit(Sx, y_kb)
                oof_blend = ridge_ex.predict(Sx)
                r_avg     = root_mean_squared_error(y_kb, Sx.mean(axis=1))
                r_stk     = root_mean_squared_error(y_kb, oof_blend)
                wts = ridge_ex.coef_ / max(ridge_ex.coef_.sum(), 1e-9)
                log.info(f"  Simple 6-avg OOF RMSE: {r_avg:.4f}")
                log.info(f"  Ridge stk OOF RMSE   : {r_stk:.4f}")
                log.info(f"  Ridge weights        : "
                         f"{dict(zip(stack_keys, np.round(wts, 4)))}")
                if r_stk < r_avg:
                    test_delta = ridge_ex.predict(St).astype(np.float64)
                    log.info("  → using Ridge re-fit blend")
                else:
                    test_delta = St.mean(axis=1).astype(np.float64)
                    log.info("  → Ridge stk worse than avg, falling back to mean")
            else:
                # No re-fit: simple mean of 6 bases
                St = np.column_stack(
                    [test_preds[k] for k in ACTIVE_MODELS] + [tabicl_test])
                test_delta = St.mean(axis=1).astype(np.float64)
                log.info("  STACK_REFIT_RIDGE=False → simple 6-avg blend")

        except Exception as e:
            log.error(f"TabICL path failed: {type(e).__name__}: {e}")
            log.warning("Falling back to karnakbaev 5-base NM blend")
            tabicl_ok = False

    if not tabicl_ok:
        # Fallback path = exp005 NM 5-base blend
        section(log, "exp006 fallback: karnakbaev 5-base Nelder-Mead blend")
        nm_w = am.load_json(am.ENSEMBLE_WEIGHTS)
        test_delta = apply_ensemble_nm(test_preds, nm_w)
        log.info(f"  NM blend test_delta range=[{test_delta.min():.3f},"
                 f"{test_delta.max():.3f}]")

    # ── Step G: post-processing ────────────────────────────────────────────
    section(log, "exp006: post-processing + SG smooth")
    test_delta_pp = apply_postproc(test_df, test_delta, best_alpha, best_tau)
    test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)

    # Spike clamp using y_kb range if available, else conservative ±60 ft
    if train_df_loaded:
        clip_lo = float(y_kb.min()) * 1.5
        clip_hi = float(y_kb.max()) * 1.5
    else:
        clip_lo, clip_hi = -60.0, 60.0
    test_delta_sg = np.clip(test_delta_sg, clip_lo, clip_hi)
    log.info(f"  test_delta_sg range=[{test_delta_sg.min():.3f},"
             f"{test_delta_sg.max():.3f}]  clip=[{clip_lo:.3f},{clip_hi:.3f}]")

    # ── Step H: submission ─────────────────────────────────────────────────
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    build_submission(test_df, test_delta_sg, sample,
                     OUTPUT_DIR / "submission.csv")


else:
    log.error(f"Unknown MODE='{MODE}'. "
              f"Choose from: train | infer | infer_tabicl | cv | "
              f"features_only | ensemble_only")


section(log, "DONE")


# In[ ]:




