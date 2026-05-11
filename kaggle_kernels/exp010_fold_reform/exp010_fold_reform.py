#!/usr/bin/env python
# coding: utf-8

# # ROGII exp010 — Fold Reform: Stratified Edge Q + Adversarial Validation Drop (LB target: 9.5-9.7)
# #
# # Phase 6 (= exp010) layers on top of exp009 v3 (subagent T 改修群):
# #   Step 1: stratified Edge Q fold — typewell hash GroupKFold に
# #     per-well-stats § tw_gr_resid_std quartile stratify を追加。
# #     困難 wells を fold 間で均等分散、σ_fold 1.18 → 0.5 目標。
# #     数学: K bins stratified sampling で σ_fold² → σ_fold²/K (best case)
# #   Step 2: adversarial validation drop — train vs test を per-well visible-only
# #     features で binary classify、AUC ≥ 0.6 なら test と離れた下位 20% wells
# #     を sample_weight 0.5 で reduce。Bojan Tunguz 流。
# #     leak guard: visible-only features のみ、TVT/target 絶対不参照。
# #
# # H10 (subagent V): exp007 σ_fold = 1.175 ft、Jensen 下限 CV ≈ 10.77 ft
# # → base improve だけで 9 切り不可能、fold 構造改革必須。
# # 期待 LB: 10.677 → 9.5-9.7 (= base 据え置きで -1.0 ft 改善射程)。
# #
# # 継承する subagent T v3 改修 (= 直交、互換):
# #   - Huber loss + heteroscedastic sample_weight + multi-seed MEDIAN + path b blend
# #
# # 継承する exp009 v4 (Edge R):
# #   - Edge R: test-time online learning + LGB continued training (= topic 698002)
# #
# # Phase 5 (v3) layers on top of v2:
# #   - Huber loss (Huber 1964) on LGB + CB, replacing RMSE.
# #   - Heteroscedastic sample_weight w_i = 1/sigma_w(i) from per-well-stats.parquet.
# #   - Multi-seed MEDIAN ensemble (Vandewiele 2021): 2 model × 3 seed × 5 fold = 30 runs.
# #   - path b stacking: kb simple-avg + own Ridge + 1D grid blend, avoids fold-misalign leak.
#
# Self-contained Kaggle GPU script that extends exp008 with:
#   0a. **Phase 4 second-layer independent edge — 案 E (Sparse GP for formation
#       posterior + variance)**
#       Public top imputes formation surface depth via FormationPlaneKNN
#       (= weighted plane fit on K=10 well centroids, point estimate). We
#       additionally fit a per-formation Sparse GP (sklearn GaussianProcessRegressor
#       with Matern 3/2 ARD kernel, K-Means inducing points M=200, n_restarts=3)
#       on (X, Y) -> formation depth, and feed posterior mean / variance /
#       (mean - plane) / normalised variance as 24 features (6 formations x 4
#       stats). Mathematics: Rasmussen 2006 (GPML), Hensman 2013 (Sparse GP).
#       Expected: -0.40 ~ -0.80 ft on top of exp008.
#   0b. **Phase 4 second-layer independent edge — Edge O (direction-aware Beam)**
#       Public top Beam Search maximises corr(GR_h, GR_v) but is sign-blind
#       to dGR direction. We add a transition penalty
#         lam_dir * |sign(d_GR_h) - sign(d_GR_v)|
#       to the existing 5 Beam configs and emit 7 features
#         beam_dir_{cons,loose,sm5,vcons,mid}_d / beam_dir_mean_d / beam_dir_std_d
#       Source: host pptx slide 6-7 (= TVT direction is encoded in GR signature
#       direction). Expected: -0.10 ~ -0.30 ft on top of exp008.
#   1. **Edge D (AR(1) Kalman on dTVT)** — exp008 から継承、7 features
#      (kalman_d / kalman_std / kalman_phi / kalman_sigma_eps / kalman_t_md /
#       kalman_vs_pf_a_d / kalman_vs_beam_cons_d)。
#   2. **Edge Q (typewell content-hash GroupKFold)** — pseudo-typewell の
#      存在 (host topic 698449) を踏まえ、同じ typewell file (MD5 of TVT/GR/
#      Geology) を共有する wells を **同 fold に閉じ込める** GroupKFold builder。
#      (= exp007 から継承)
#   3. **Edge M (visible-as-typewell self-alignment)** — host pptx Slide 9
#      公式推奨、4 features 追加 (= exp007 から継承)。
#   4. **自前 LGB×3 (lr=0.02/0.05/0.10) + CB (lr=0.05)** を Edge Q fold で
#      5-fold OOF 学習 (= exp007 から継承)。GP 24 + Edge O 7 = 31 新 features
#      込みで再 fit。
#   5. **9-base Ridge meta** (= 5 karnakbaev + 4 own、positive=True、no
#      intercept) (= exp007 から継承)。
#   6. **TabICL は dormant** (= exp006 postmortem)。
#   7. **Edge S: dTVT 0.01 grid round-to-grid**, inspired by hengck23 discussion 697431.
#      Submission の最終 tvt を np.round(., 2) で 0.01 ft grid に snap。
#      773/773 wells で dTVT min_step = 0.01 ft が 100% 普遍 (中央実測)。
#      Expected: -0.05 ~ -0.30 ft on top of GP + Edge O + meta blend。
#
# Pipeline (MODE = 'infer_edge_e_gp_o'):
#   a. Build test features from live `/kaggle/input/.../test/` directory
#      (with Edge M + AR(1) Kalman + GP 24 + Edge O 7 features per hidden row).
#   b. Load karnakbaev artefacts: LGB×3 + XGB + CB boosters, train_df.parquet,
#      oof_predictions.parquet, features.json.
#   c. Compute Edge Q fold (= typewell content-hash GroupKFold over 773 wells).
#   d. Compute Edge M + Kalman + GP + Edge O FE for both train and test
#      (rebuild train_df).
#   e. Re-compute karnakbaev 5-base OOF under Edge Q fold (= predict val fold).
#   f. Train 自前 LGB×3 + CB (= 4 own base) under Edge Q fold (= 5-fold OOF)
#      with Edge M + Kalman + GP + Edge O features included.
#   g. Build 9-base OOF stack → Ridge meta fit (positive=True, no intercept).
#   h. Apply Ridge meta to 9-base test predictions → karnakbaev postproc → SG smooth.
#
# License / source attribution:
#   - karnakbaev artifacts:  `karnakbaevarthur/rogii-code-helper-dataset` (apache-2.0)
#   - thbdh5765 train cache: `thbdh5765/rogii-v1-train-cache` (cc0-1.0; attached but
#                            kept as reference only — schema differs from our test FE)
#   - TabICL package + ckpt: `thermostatic/rogii-tabicl-v2-public-assets` (apache-2.0;
#                            attached but TabICL is dormant in this exp).
#   - Edge Q hash builder:   adapted from subagent G commit 1bb6caf
#                            (`notebooks/_typewell_groups_build.py` in this repo).
#   - Edge M defn:           host pptx Slide 9 (`docs/research/host-pptx-summary.dense.md`)
#                            + subagent G commit 537fa9f (`docs/research/independent-edges.dense.md` §M).
#   - Edge D Kalman:         AR(1) state-space, Kalman 1960 / Doucet 2001 mathematics,
#                            no external code. Spec: docs/research/independent-edges.dense.md §1
#                            + experiments/exp008/design.md.
#   - Case E Sparse GP:      Per-formation GaussianProcessRegressor (sklearn) with
#                            Matern 3/2 ARD kernel and K-Means inducing points (M=200).
#                            Mathematics: Rasmussen 2006 (GPML), Hensman 2013 (Sparse GP).
#                            No external code. Spec: docs/research/independent-edges.dense.md §2
#                            + experiments/exp009/design.md §3.1-3.3.
#   - Edge O dir-aware Beam: Augments Beam transition cost with sign-of-dGR penalty.
#                            Source: host pptx slide 6-7 (= TVT direction encoded in GR
#                            signature direction). Mathematics: stratigraphic correlation
#                            (Catuneanu 2006 "Principles of Sequence Stratigraphy"). No
#                            external code. Spec: docs/research/independent-edges.dense.md
#                            §9.6 + experiments/exp009/design.md §3.4-3.5.
#   - Pipeline base:         karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline
#                            (public Kaggle, LB 10.784); reused under fair-use sharing.
#   - Original solution credits romantamrazov/rogii-super-solution-lb-top-3 ideas.
#
# This file is for the team's exp009 entry only.

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
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as _GP_C,
    Matern as _GP_Matern,
    WhiteKernel as _GP_White,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─── Execution Config ────────────────────────────────────────────────────────
# MODE: "train" | "infer" | "cv" | "features_only" | "ensemble_only"
#       | "infer_tabicl" | "infer_edge_qm" | "infer_edge_d_kalman"
#       | "infer_edge_e_gp_o"
#   infer_edge_e_gp_o: exp009 path = Edge Q + Edge M + AR(1) Kalman + Sparse GP
#                  (24 cols, case E) + dir-aware Beam (7 cols, Edge O) +
#                  自前 LGB×3 + CB train + 9-base Ridge meta.
#                  Pipeline is identical to infer_edge_d_kalman; GP and
#                  Edge O FE are injected inside build_well_features when
#                  GP_ENABLE / EDGE_O_ENABLE.
MODE = "infer_edge_e_gp_o"

# Active base models (for karnakbaev pretrained predict).
ACTIVE_MODELS = ["lgb0", "lgb1", "lgb2", "xgb", "cb"]

# ─── TabICL knobs (DORMANT in exp007) ───────────────────────────────────────
# TabICL was a -0.186 ft regression in exp006, kept dormant here.
TABICL_ENABLE        = False   # exp007: dormant (exp006 postmortem)
TABICL_N_FEATURES    = 50
TABICL_CTX           = 4096
TABICL_N_ESTIMATORS  = 4
TABICL_SEEDS         = [42]
TABICL_CHUNK         = 50000
TABICL_USE_AMP       = "auto"
TABICL_BATCH_SIZE    = 4
TABICL_QUICK_LGB_N   = 300
TABICL_QUICK_SAMPLE  = 200_000
TABICL_HARD_TIMEOUT_S = 6 * 3600

# ─── Edge Q knobs (typewell content-hash GroupKFold) ────────────────────────
EDGE_Q_ENABLE        = True
EDGE_Q_HASH_COLS     = ("TVT", "GR", "Geology")  # cols hashed; missing skipped
EDGE_Q_FALLBACK_BY_WELL = True  # if EDGE_Q crashes, fall back to GroupKFold(by well)

# ─── exp010: Stratified Edge Q fold + Adversarial Validation drop ───────────
# H10 (= subagent V 発見): exp007 per-fold RMSE std σ_fold = 1.175 ft で
# Jensen 下限 CV ≈ 10.77 ft が天井。9 切り (LB 8.x) には fold 構造改革が必須。
# Step 1: stratified Edge Q fold (= 困難 wells を fold 間で均等分散)
# Step 2: adversarial validation drop (= test 距離考慮の reweight)
STRATIFIED_EDGE_Q_ENABLE = True
STRATIFIED_EDGE_Q_KEY    = "tw_gr_resid_std"  # per-well-stats parquet column
STRATIFIED_EDGE_Q_N_BINS = 4                    # quartile

ADVERSARIAL_DROP_ENABLE  = True
ADVERSARIAL_AUC_THRESHOLD = 0.60   # below = shift mild, skip reweight
ADVERSARIAL_DROP_QUANTILE = 0.20   # bottom q% by test-likelihood → reweight
ADVERSARIAL_DROP_WEIGHT   = 0.5    # 1.0 = no change, 0.0 = drop, 0.5 = reduce
ADVERSARIAL_LGB_PARAMS    = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=5,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
    seed=42,
)
ADVERSARIAL_LGB_N_EST     = 200
ADVERSARIAL_LGB_EARLY_STOP = 30
# Per-well features used for adversarial classification. MUST be train/test
# both computable from visible-only data; MUST NOT include any TVT-derived
# or hidden-region statistics (= leak guard).
ADVERSARIAL_FEATURE_COLS = (
    # visible-region only / target-independent statistics. Anything derived
    # from hidden TVT (target) is excluded to preserve leak guard.
    "visible_ratio",
    "gr_noise_std",
    "gr_mean",
    "gr_std",
    "gr_nan_frac",
    "n_rows",
    "n_visible",
    "z_range",
)

# ─── Edge M knobs (visible-as-typewell self-alignment FE) ───────────────────
EDGE_M_ENABLE        = True
EDGE_M_WIN_SIZE      = 200    # samples per visible window
EDGE_M_WIN_STEP      = 50     # stride between windows
EDGE_M_CHUNK_SIZE    = 200    # samples per hidden chunk
EDGE_M_MIN_VISIBLE   = 400    # if visible region < this, fall back to typewell-only
EDGE_M_HARD_TIMEOUT_S = 30 * 60  # 30 min hard cap on Edge M FE phase

# ─── Edge D knobs (AR(1) Kalman / PF on dTVT, exp008 新規) ──────────────────
# Per-well Yule-Walker MLE for (phi, sigma_eps) on visible dTVT, with hierarchical
# Bayesian shrinkage toward the global prior (= per-well-stats parquet p50). Then
# 1D Kalman forward pass on dTVT to get TVT MAP estimate + posterior std grow.
KALMAN_ENABLE        = True
# Global priors from outputs/eda/first_principles/per-well-stats.parquet (n=773):
#   ar1_phi: p50=0.99887, p25=0.99671, p75=0.99940
#   ar1_eps_std: p50=0.01556, p25=0.01006, p75=0.03098
KALMAN_PHI_GLOBAL    = 0.99887
KALMAN_SIGMA_GLOBAL  = 0.01556
KALMAN_LAM_PHI       = 0.3   # shrinkage strength toward global phi (higher = more pull)
KALMAN_LAM_SIGMA     = 0.3   # shrinkage strength toward global sigma_eps
KALMAN_PSEUDO_COUNT  = 100   # pseudo-count in shrinkage weight = n / (n + lam * pseudo)
KALMAN_PHI_CLIP      = 0.99999  # numerical guard so 1 - phi^2 > 0
KALMAN_SIGMA_FLOOR   = 1e-4  # MLE sigma floor to avoid degenerate variance
KALMAN_STD_FLOOR_FRAC = 0.1  # std_pred clipped to >= sigma * this fraction

# ─── Case E knobs (Sparse GP for per-formation posterior, exp009 新規) ──────
# Per-formation GP fit on (X, Y) -> formation depth. K-Means inducing points
# cluster reduces 773 wells to M=200 cluster centers. Matern 3/2 ARD kernel
# with marginal-likelihood hyperparameter optimisation (n_restarts=3). Per-
# formation posterior mean / variance / (mean - plane) / normalised variance
# emit 24 features (6 formations x 4 stats). FormationPlaneKNN is preserved
# as the orthogonal point-estimate baseline. See experiments/exp009/design.md.
GP_ENABLE            = True
GP_M_INDUCE          = 200    # K-Means cluster centers (= inducing points)
GP_N_RESTARTS        = 3      # n_restarts_optimizer for sklearn GPR
GP_LENGTH_SCALE_INIT = 1.0    # initial length scale (post-normalisation)
GP_LENGTH_SCALE_BOUNDS = (1e-2, 1e3)
GP_NOISE_INIT        = 1e-2
GP_NOISE_BOUNDS      = (1e-5, 1e1)
GP_ALPHA             = 1e-6   # Cholesky jitter
GP_FALLBACK_KNN_K    = 20     # KNN fallback if GP fit fails

# ─── Edge O knobs (direction-aware Beam, exp009 新規) ───────────────────────
# Augments Beam transition cost with lam_dir * |sign(d_GR_h) - sign(d_GR_v)|
# for each step, encouraging path direction to match GR-signature direction.
# Per host pptx slide 6-7. Per BEAMS config: emit 5 dir-aware paths + mean +
# std = 7 features (beam_dir_*_d).
EDGE_O_ENABLE        = True
EDGE_O_LAMBDA_DIR    = 0.5    # direction-penalty strength
EDGE_O_GR_SMOOTH_R   = 2      # 5-tap median pre-smoothing on dGR (sign noise)

# ─── 自前 base hyperparameter (Phase 5 v3: Multi-seed MEDIAN) ───────────────
# 改修 3: LGB lr=0.05 単一 + CB lr=0.05 単一 を 3 seed × 5 fold で MEDIAN.
# 旧 (v2): LGB×3 lr (0.02/0.05/0.10) + CB×1 = 4 base × 1 seed × 5 fold = 20 run
# 新 (v3): LGB×1 lr=0.05 + CB×1 lr=0.05 = 2 model × 3 seed × 5 fold = 30 run
OWN_BASE_ENABLE      = True
OWN_USE_MEDIAN_SEEDS = True  # Phase 5 v3 toggle
OWN_LGB_LRS          = (0.02, 0.05, 0.10)  # legacy
OWN_LGB_NUM_LEAVES   = 127
OWN_LGB_N_ESTS       = (8000, 4000, 2000)  # legacy
OWN_LGB_SEEDS        = (42, 7, 123)        # legacy
OWN_CB_LR            = 0.05
OWN_CB_DEPTH         = 8
OWN_CB_N_EST         = 5000
OWN_CB_SEED          = 42
# Phase 5 v3 multi-seed MEDIAN config
OWN_LGB_LR_MED       = 0.05
OWN_LGB_N_EST_MED    = 4000
OWN_LGB_SEEDS_MED    = (42, 123, 2024)
OWN_CB_SEEDS_MED     = (42, 123, 2024)
OWN_CB_N_EST_MED     = 4000
OWN_TRAIN_HARD_TIMEOUT_S = 6 * 3600  # 6h hard cap on 自前 train phase

# ─── Heteroscedastic sample_weight (Phase 5 v3 改修 2) ──────────────────────
# per-well-stats.parquet `b_well_resid_std` で 1/sigma 重み投入.
HETERO_W_ENABLE      = True
HETERO_W_CLIP_LO     = 0.5
HETERO_W_CLIP_HI     = 2.0
HETERO_W_FILL_P50    = 0.00835  # b_well_resid_std p50 (n=773 wells)
HETERO_W_STATS_PATH  = "/kaggle/input/rogii-per-well-stats/per-well-stats.parquet"
HETERO_W_STATS_LOCAL = "outputs/eda/first_principles/per-well-stats.parquet"

# ─── Stacking knob (Phase 5 v3 改修 4: path b = kb-side simple-average) ─────
STACK_REFIT_RIDGE             = True
STACK_OWN_WEIGHT_DEGRADE_LIMIT = 0.8  # path b では own diversity が low なので緩和 0.5→0.8
STACK_PATH_B_ENABLE           = True
STACK_PATH_B_GRID_STEP        = 0.05

# ─── Edge R knobs (Phase 5 v4: test-time online learning) ──────────────────
# Inspired by Kaggle discussion topic 698002 (online 10.953 vs no-online 11.323 = -0.370 ft).
# test wells の visible region 末尾 K 行を擬似 hidden に変換し、
# karnakbaev pretrained LGB 3 base 各々に `init_model` で continued training。
# 自前 base (MEDIAN) は train data で十分 flexible なので Edge R を kb base のみに適用。
EDGE_R_ENABLE          = True
EDGE_R_VISIBLE_TAIL_K  = 100   # visible 内の擬似 hidden サイズ
EDGE_R_MIN_VISIBLE_ROWS = 30
EDGE_R_NUM_BOOST       = 200   # continued training round 数 (= topic 698002)
EDGE_R_LR_MUL          = 0.5   # continued training の lr × 0.5
EDGE_R_BLEND_W         = 0.5   # online と既存 path b blend の重ね合わせ weight
EDGE_R_HARD_TIMEOUT_S  = 15 * 60  # 15 min hard cap on Edge R (= timeout 対策)

# Debug flags
# Allow env override for local smoke runs.
import os as _os_dbg
_ENV_DEBUG_MAX_WELLS = _os_dbg.environ.get("ROGII_DEBUG_MAX_WELLS")
DEBUG_MAX_WELLS    = int(_ENV_DEBUG_MAX_WELLS) if _ENV_DEBUG_MAX_WELLS else None   # set to e.g. 3 for fast iteration
DEBUG_ONE_FOLD     = False
DEBUG_INSPECT_PF   = False
DEBUG_INSPECT_BEAM = False
DEBUG_SKIP_OWN_TRAIN = False  # if True, skip 自前 train and reuse karnakbaev OOF only

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
# Phase 5 v3: Huber loss + heteroscedastic sample weight + multi-seed MEDIAN
#  - Huber loss (Huber 1964): fat-tail noise (dtvt_p95/std=2.4 > Gaussian 1.645).
#    LGB `objective="huber"` + `alpha=0.9`、CB `loss_function="Huber:delta=0.5"`.
#  - Heteroscedastic sample_weight: per-well-stats.parquet b_well_resid_std 1/sigma.
#  - Multi-seed MEDIAN: LGB lr=0.05 + CB lr=0.05 × 3 seed × 5 fold = 30 run.
LGB_BASE = dict(
    boosting_type    = "gbdt",
    learning_rate    = 0.04,
    num_leaves       = 127,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_lambda       = 5.0,
    reg_alpha        = 0.1,
    objective        = "huber",   # Phase 5 v3: was "regression"
    alpha            = 0.9,       # Phase 5 v3: Huber delta scaling factor
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
    loss_function         = "Huber:delta=0.5",  # Phase 5 v3: was "RMSE"
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


# ───────────────────────────────────────────────────────────────────────────
# Edge O (exp009) — Direction-aware Beam Search.
#
# Augments the existing Beam transition cost with a sign-of-dGR penalty:
#     cost_aug = cost_base + lam_dir * |sign(d_GR_h(s)) - sign(d_GR_v(path_s))|
# Source: host pptx slide 6-7. The penalty fires whenever the horizontal-well
# GR signature direction (= sign of derivative) disagrees with the typewell
# GR signature direction at the matched path index. sign(0) = 0 by
# convention. Predictions emit 7 features: 5 dir-aware path deltas plus mean
# and std across the 5 configs.
# ───────────────────────────────────────────────────────────────────────────


def edge_o_feature_names() -> List[str]:
    return [
        "beam_dir_cons_d", "beam_dir_loose_d", "beam_dir_sm5_d",
        "beam_dir_vcons_d", "beam_dir_mid_d",
        "beam_dir_mean_d", "beam_dir_std_d",
    ]


def _signs(arr: np.ndarray) -> np.ndarray:
    """sign(x) with sign(0)=0."""
    return np.sign(arr).astype(np.float32)


def beam_search_dir(
    gr_h:       np.ndarray,
    tw_tvt:     np.ndarray,
    tw_gr:      np.ndarray,
    start_tvt:  float,
    bs:         int   = 10,
    mc:         float = 20.0,
    es:         float = 144.0,
    r:          int   = 2,
    lam_dir:    float = 0.5,
) -> np.ndarray:
    """Direction-aware Beam Search.

    Identical structure to ``beam_search`` but with an extra direction-penalty
    term ``lam_dir * |sign(dGR_h_step) - sign(dGR_v_step)|`` added to the
    transition cost ``mv``. The horizontal-side dGR is the per-step
    derivative of the (smoothed) horizontal GR signal; the typewell-side
    dGR is the difference between consecutive ``tw_gr`` values along the
    selected transition (-1, 0, +1) over the typewell index axis.
    """
    tw_tvt = np.asarray(tw_tvt, np.float32)
    tw_gr  = np.asarray(tw_gr,  np.float32)
    T  = len(tw_tvt)
    fb = float(np.nanmean(tw_gr))
    sg = fill_and_smooth_gr(gr_h, fb, r)
    si = nn_idx(tw_tvt, start_tvt)
    ns = len(sg)

    # Pre-compute sign of dGR_h (horizontal-well GR derivative). For step s,
    # use sg[s] - sg[s-1]; for s=0, define 0.
    dgrh = np.zeros(ns, dtype=np.float32)
    if ns >= 2:
        dgrh[1:] = sg[1:] - sg[:-1]
    sign_dgrh = _signs(dgrh)

    # Pre-compute typewell dGR per index: tw_gr[i] - tw_gr[i-1], with i=0 → 0.
    tw_dgr = np.zeros(T, dtype=np.float32)
    if T >= 2:
        tw_dgr[1:] = tw_gr[1:] - tw_gr[:-1]
    sign_tw_dgr = _signs(tw_dgr)

    bi = np.full(bs, si, np.int32)
    bc = np.zeros(bs, np.float64)
    bps = np.empty((ns, bs), np.int32)
    bpb = np.empty((ns, bs), np.int32)

    for s, gv in enumerate(sg):
        ci = np.clip(bi[:, None] + np.array([-1, 0, 1]), 0, T - 1)
        em = (gv - tw_gr[ci]) ** 2 / es
        mv_base = mc * np.array([1, 0, 1])[None, :]
        # Direction penalty: compare sign of dgrh[s] vs sign_tw_dgr at the
        # transition target index ci. transition (-1) goes to ci-th entry
        # which corresponds to tw_dgr[ci]. Larger |sign diff| → larger penalty.
        sign_ci = sign_tw_dgr[ci]                            # (bs, 3)
        dir_pen = lam_dir * np.abs(sign_dgrh[s] - sign_ci)   # (bs, 3)
        mv = mv_base + dir_pen
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


def compute_edge_o_features(
    hgr_full:       np.ndarray,
    tw_tvt:         np.ndarray,
    tw_gr:          np.ndarray,
    last_known_tvt: float,
    sel_local:      np.ndarray,
    lam_dir:        float = 0.5,
) -> Dict[str, np.ndarray]:
    """Run direction-aware Beam over all 5 configs; return 7-feature dict.

    Output keys: ``beam_dir_<cons|loose|sm5|vcons|mid>_d`` + ``beam_dir_mean_d``
    + ``beam_dir_std_d``. Each is float32 of shape ``(len(sel_local),)``.
    """
    paths: Dict[str, np.ndarray] = {}
    for (bs, mc, es, r, tag) in BEAMS:
        out_tag = "mid" if tag == "vloose" else tag
        try:
            paths[out_tag] = beam_search_dir(
                hgr_full, tw_tvt, tw_gr, last_known_tvt,
                bs=bs, mc=mc, es=es, r=r, lam_dir=lam_dir,
            )
        except Exception as exc:
            # Per-config fallback: zeros at the right shape (= len(hgr_full))
            print(f"  WARN beam_dir({out_tag}) failed: {exc}")
            paths[out_tag] = np.full(len(hgr_full),
                                     np.float32(last_known_tvt), np.float32)

    lkt = np.float32(last_known_tvt)
    stack = np.stack([p for p in paths.values()], axis=1)  # (n_hidden, 5)
    out: Dict[str, np.ndarray] = {}
    for tag, p in paths.items():
        out[f"beam_dir_{tag}_d"] = (p - lkt).astype(np.float32)[sel_local]
    out["beam_dir_mean_d"] = (stack.mean(axis=1) - lkt).astype(np.float32)[sel_local]
    out["beam_dir_std_d"]  = stack.std(axis=1).astype(np.float32)[sel_local]
    return out


print("Direction-aware Beam (Edge O) functions loaded ✓")


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


# ## 6.5 Edge Q + Edge M helpers (exp007 新規)
#
# Edge Q: typewell content-hash GroupKFold builder (= subagent G `1bb6caf` の
#   notebooks/_typewell_groups_build.py を kernel inline 移植)
# Edge M: visible-as-typewell self-alignment FE (= host pptx Slide 9 公式推奨、
#   subagent G `537fa9f` で文書化、本 exp 初実装)

# In[ ]:


import hashlib


def compute_typewell_hashes(
    train_dir: Path,
    well_ids: List[str],
    cols: Tuple[str, ...] = EDGE_Q_HASH_COLS,
) -> Dict[str, str]:
    """Compute MD5 hash of (TVT, GR, Geology) for each well's typewell file.

    pseudo-typewell の 13 group (= 35 wells) を recover し、同一 hash の wells
    を同一 fold に閉じ込める fold builder の入力として使う。

    Source: discussion topic 698449 (host official reply 2026-05-10) で
    pseudo-typewell の存在を host が明言。
    """
    hashes: Dict[str, str] = {}
    for wid in well_ids:
        tw_path = train_dir / f"{wid}__typewell.csv"
        if not tw_path.exists():
            hashes[wid] = wid  # fallback: own well_id (= unique singleton)
            continue
        try:
            df = pd.read_csv(tw_path)
        except Exception:
            hashes[wid] = wid
            continue
        h = hashlib.md5()
        for col in cols:
            if col not in df.columns:
                continue
            if df[col].dtype == "object":
                payload = df[col].fillna("NaN").astype(str).str.cat(sep="|").encode()
            else:
                payload = df[col].fillna(-99999.0).round(4).values.tobytes()
            h.update(payload)
        hashes[wid] = h.hexdigest()
    return hashes


def build_edge_q_folds(
    train_df:   pd.DataFrame,
    train_dir:  Path,
    n_splits:   int = 5,
    seed:       int = 42,
    fallback_by_well: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fold_id, groups) using typewell-hash GroupKFold.

    fold_id: int8 array of shape (n_rows,) with values 0..n_splits-1
    groups:  object array of shape (n_rows,) with hash strings (= group key)

    leak 検査: 同 hash の 2 wells が同 fold に居ることを確認するため、
    呼出側で `verify_edge_q_no_leak()` を必ず実行すること。
    """
    well_ids = train_df["well"].unique().tolist()
    try:
        hashes = compute_typewell_hashes(train_dir, well_ids)
    except Exception as e:
        if not fallback_by_well:
            raise
        print(f"  [Edge Q] hash computation failed ({type(e).__name__}: {e}) "
              f"→ fallback to GroupKFold(by well)")
        hashes = {wid: wid for wid in well_ids}
    groups = train_df["well"].map(hashes).fillna("__unknown__").values
    cv = GroupKFold(n_splits=n_splits)
    fold_id = np.full(len(train_df), -1, dtype=np.int8)
    for k, (_, va_idx) in enumerate(cv.split(train_df, train_df["target"], groups)):
        fold_id[va_idx] = k
    if (fold_id < 0).any():
        # fallback: assign by well_id hash mod n_splits
        bad = np.where(fold_id < 0)[0]
        for i in bad:
            fold_id[i] = abs(hash(str(groups[i]))) % n_splits
    return fold_id.astype(np.int8), np.asarray(groups, dtype=object)


def verify_edge_q_no_leak(
    fold_id: np.ndarray,
    groups:  np.ndarray,
) -> Tuple[bool, str]:
    """Assert no group_id appears in more than one fold (= no fold leak)."""
    df = pd.DataFrame({"fold": fold_id, "group": groups})
    grp_to_folds = df.groupby("group")["fold"].nunique()
    bad = grp_to_folds[grp_to_folds > 1]
    if len(bad) > 0:
        return False, f"LEAK: {len(bad)} groups span multiple folds: {bad.head(3).to_dict()}"
    return True, f"OK: {df['group'].nunique()} groups, {fold_id.max()+1} folds, no leak"


# ─────────────────────────────────────────────────────────────────────────────
# exp010 Step 1: stratified Edge Q fold (= H10 対策、σ_fold 1.18 → 0.5)
# ─────────────────────────────────────────────────────────────────────────────
from collections import defaultdict as _defaultdict_strat


def build_stratified_edge_q_folds(
    train_df: pd.DataFrame,
    train_dir: Path,
    per_well_stats_df: pd.DataFrame,
    n_splits:        int   = 5,
    seed:            int   = 42,
    stratify_key:    str   = "tw_gr_resid_std",
    n_bins:          int   = 4,
    fallback_by_well: bool = True,
    stats_well_col:  str   = "well_id",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fold_id, groups) using stratified typewell-hash GroupKFold.

    Algorithm (= exp010 design.md §2.1):
        1. typewell content-hash で each well を group 化 (= 既存 Edge Q)
        2. 各 group の代表値 = median(stratify_key) over member wells
        3. quartile bin に分類 (default n_bins=4)
        4. bin 内 group を shuffle、round-robin で fold に分配
           (bin 毎に offset で fold をずらす → 各 fold で 4 bin balanced)

    Math: K bins stratified sampling で σ_fold² → σ_fold²/K (best case)。
    exp007 σ_fold = 1.175 ft、K=4 quartile で σ_fold ≤ 0.5 ft 目標。

    Leak guard:
        - per-well-stats.parquet は train wells のみ参照 (= no test leak)
        - hash group constraint > stratify (= group atomic)
        - stratify_key (tw_gr_resid_std 等) は visible-region GR vs typewell
          residual で TVT 非依存
    """
    well_ids = train_df["well"].unique().tolist()
    try:
        hashes = compute_typewell_hashes(train_dir, well_ids)
    except Exception as e:
        if not fallback_by_well:
            raise
        print(f"  [stratified Edge Q] hash computation failed "
              f"({type(e).__name__}: {e}) → fallback to GroupKFold(by well)")
        hashes = {wid: wid for wid in well_ids}

    if stratify_key not in per_well_stats_df.columns:
        print(f"  [stratified Edge Q] stratify_key={stratify_key!r} not in "
              f"per-well-stats → fall back to plain build_edge_q_folds")
        return build_edge_q_folds(
            train_df, train_dir, n_splits=n_splits, seed=seed,
            fallback_by_well=fallback_by_well,
        )

    stats_lookup = dict(zip(
        per_well_stats_df[stats_well_col].astype(str),
        per_well_stats_df[stratify_key].astype(float),
    ))

    group_to_wells: dict = _defaultdict_strat(list)
    for wid in well_ids:
        g = hashes.get(wid, wid)
        v = stats_lookup.get(str(wid), np.nan)
        group_to_wells[g].append((wid, v))

    group_records: list = []
    for g, members in group_to_wells.items():
        vals = np.array([v for _, v in members], dtype=float)
        if np.isnan(vals).all():
            group_records.append((g, np.nan))
        else:
            group_records.append((g, float(np.nanmedian(vals))))

    valid_records = [(g, v) for g, v in group_records if not np.isnan(v)]
    nan_records   = [(g, v) for g, v in group_records if np.isnan(v)]

    if len(valid_records) < n_bins * n_splits:
        print(f"  [stratified Edge Q] only {len(valid_records)} valid groups "
              f"vs {n_bins*n_splits} needed → fall back to plain Edge Q")
        return build_edge_q_folds(
            train_df, train_dir, n_splits=n_splits, seed=seed,
            fallback_by_well=fallback_by_well,
        )

    values = np.array([v for _, v in valid_records], dtype=float)
    quantile_edges = np.quantile(values, np.linspace(0, 1, n_bins+1)[1:-1])
    bin_to_groups: dict = _defaultdict_strat(list)
    for g, v in valid_records:
        b = int(np.searchsorted(quantile_edges, v, side="right"))
        b = min(b, n_bins - 1)
        bin_to_groups[b].append(g)

    rng = np.random.RandomState(seed)
    group_to_fold: dict = {}
    for b in range(n_bins):
        bucket = list(bin_to_groups[b])
        rng.shuffle(bucket)
        offset = b % n_splits
        for i, g in enumerate(bucket):
            group_to_fold[g] = (i + offset) % n_splits
    for g, _ in nan_records:
        group_to_fold[g] = abs(hash(str(g))) % n_splits

    groups = train_df["well"].map(hashes).fillna("__unknown__").values
    fold_id = np.array(
        [group_to_fold.get(g, abs(hash(str(g))) % n_splits) for g in groups],
        dtype=np.int8,
    )
    return fold_id, np.asarray(groups, dtype=object)


def summarize_stratified_balance(
    fold_id: np.ndarray,
    train_df: pd.DataFrame,
    per_well_stats_df: pd.DataFrame,
    stratify_key: str = "tw_gr_resid_std",
    n_bins: int = 4,
    stats_well_col: str = "well_id",
) -> pd.DataFrame:
    """Return a fold × bin count matrix for sanity-check on stratification."""
    stats_lookup = dict(zip(
        per_well_stats_df[stats_well_col].astype(str),
        per_well_stats_df[stratify_key].astype(float),
    ))
    df = pd.DataFrame({
        "well": train_df["well"].values,
        "fold": fold_id,
    }).drop_duplicates(subset=["well"])
    df["value"] = df["well"].astype(str).map(stats_lookup)
    valid = df.dropna(subset=["value"]).copy()
    if len(valid) == 0:
        return pd.DataFrame()
    edges = np.quantile(valid["value"].values, np.linspace(0, 1, n_bins+1)[1:-1])
    valid["bin"] = np.minimum(
        np.searchsorted(edges, valid["value"].values, side="right"),
        n_bins - 1,
    )
    return (valid.groupby(["fold", "bin"]).size()
                 .unstack(fill_value=0).sort_index())


# ─────────────────────────────────────────────────────────────────────────────
# exp010 Step 2: adversarial validation drop (= train vs test shift reweight)
# ─────────────────────────────────────────────────────────────────────────────


def compute_adversarial_well_features(
    raw_dir: Path,
    well_ids: List[str],
    feature_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Build per-well visible-only features for adversarial classification.

    For each well, reads ``{wid}__horizontal_well.csv`` and computes:
        - n_rows, n_visible, visible_ratio
        - gr_mean, gr_std, gr_nan_frac, gr_noise_std
        - z_range
    All from **visible region only** (= TVT_input is finite). No hidden TVT,
    no target — strict leak guard.

    Returns DataFrame indexed by well_id with columns matching feature_cols.
    """
    rows: List[dict] = []
    for wid in well_ids:
        path = raw_dir / f"{wid}__horizontal_well.csv"
        if not path.exists():
            rows.append({"well_id": wid, **{c: np.nan for c in feature_cols}})
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            rows.append({"well_id": wid, **{c: np.nan for c in feature_cols}})
            continue
        n_rows = len(df)
        # visible mask = TVT_input is finite (visible region marker)
        if "TVT_input" in df.columns:
            visible_mask = df["TVT_input"].notna().values
        else:
            visible_mask = np.ones(n_rows, dtype=bool)
        n_vis = int(visible_mask.sum())
        gr = df["GR"].values if "GR" in df.columns else np.full(n_rows, np.nan)
        gr_vis = gr[visible_mask]
        z_col  = df["Z"] if "Z" in df.columns else (df["MD"] if "MD" in df.columns else None)
        z_range = float(z_col.max() - z_col.min()) if z_col is not None else np.nan
        gr_mean = float(np.nanmean(gr_vis)) if len(gr_vis) > 0 else np.nan
        gr_std  = float(np.nanstd(gr_vis)) if len(gr_vis) > 0 else np.nan
        gr_nan_frac = float(np.isnan(gr).mean())
        # gr_noise_std: residual std after rolling median (= rough noise estimate)
        if len(gr_vis) > 30:
            gr_med = pd.Series(gr_vis).rolling(11, min_periods=1, center=True).median().values
            gr_noise_std = float(np.nanstd(gr_vis - gr_med))
        else:
            gr_noise_std = np.nan
        row = {
            "well_id":       wid,
            "n_rows":        n_rows,
            "n_visible":     n_vis,
            "visible_ratio": n_vis / max(n_rows, 1),
            "gr_mean":       gr_mean,
            "gr_std":        gr_std,
            "gr_nan_frac":   gr_nan_frac,
            "gr_noise_std":  gr_noise_std,
            "z_range":       z_range,
        }
        # Ensure every requested feature col exists
        for c in feature_cols:
            row.setdefault(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("well_id")


def fit_adversarial_classifier(
    feat_train: pd.DataFrame,
    feat_test:  pd.DataFrame,
    feature_cols: Tuple[str, ...],
    lgb_params: dict,
    n_estimators: int,
    early_stopping: int,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Fit a binary classifier (train=0, test=1) and return (oof AUC, train OOF prob).

    Algorithm:
        - 5-fold StratifiedKFold over the combined set (stratify by class label)
        - LightGBM with early stopping on OOF AUC
        - Return AUC + OOF probability of test-class for **train rows only**
          (= used to rank train wells by "test-likelihood")

    Leak guard: features are visible-only per-well statistics. The classifier
    never sees TVT, target, or hidden-region anything.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as _lgb_av

    n_train = len(feat_train)
    n_test  = len(feat_test)
    X = pd.concat(
        [feat_train[list(feature_cols)], feat_test[list(feature_cols)]],
        axis=0,
        ignore_index=False,
    )
    y = np.concatenate([np.zeros(n_train), np.ones(n_test)]).astype(np.int8)

    # Fillna with column median (train+test combined) → robust
    X = X.fillna(X.median(numeric_only=True))
    # If still NaN (= all-NaN column), fill with 0
    X = X.fillna(0.0)

    # 5-fold stratified CV; for very small n_test (200) use min(5, n_test)
    n_splits = min(5, max(2, n_test // 2))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(X), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        Xt, yt = X.iloc[tr_idx], y[tr_idx]
        Xv, yv = X.iloc[va_idx], y[va_idx]
        clf = _lgb_av.LGBMClassifier(
            n_estimators=n_estimators,
            **{k: v for k, v in lgb_params.items() if k != "metric"},
        )
        clf.fit(
            Xt.values, yt,
            eval_set=[(Xv.values, yv)],
            callbacks=[
                _lgb_av.early_stopping(early_stopping),
                _lgb_av.log_evaluation(0),
            ],
        )
        oof_prob[va_idx] = clf.predict_proba(Xv.values)[:, 1].astype(np.float32)
    auc = float(roc_auc_score(y, oof_prob))
    return auc, oof_prob[:n_train]  # only train rows for reweight


def apply_adversarial_reweight(
    w_train: np.ndarray,
    train_well_ids: np.ndarray,
    well_to_test_likelihood: dict,
    drop_quantile: float = 0.20,
    drop_weight: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """Reweight w_train: bottom `drop_quantile` wells (= least test-like) → drop_weight.

    Returns (new_w_train, n_wells_reweighted).
    """
    n_wells = len(well_to_test_likelihood)
    if n_wells == 0:
        return w_train, 0
    vals = np.array(list(well_to_test_likelihood.values()))
    threshold = np.quantile(vals, drop_quantile)
    low_wells = {w for w, v in well_to_test_likelihood.items()
                 if v <= threshold}
    new_w = w_train.copy()
    mask = np.array([w in low_wells for w in train_well_ids])
    new_w[mask] = new_w[mask] * drop_weight
    return new_w, int(len(low_wells))


def compute_edge_m_features(
    kgr:        np.ndarray,
    hgr:        np.ndarray,
    ktvt:       np.ndarray,
    last_md:    float,
    last_tvt:   float,
    hmd:        np.ndarray,
    slope_md:   float,
    beam_cons_d: np.ndarray,
    win_size:   int = EDGE_M_WIN_SIZE,
    win_step:   int = EDGE_M_WIN_STEP,
    chunk_size: int = EDGE_M_CHUNK_SIZE,
    min_visible: int = EDGE_M_MIN_VISIBLE,
) -> Dict[str, np.ndarray]:
    """Compute 4 Edge M features per hidden row using visible-as-typewell self-NCC.

    Args (all per-well, shapes consistent):
      kgr:         visible (= known) section GR, shape (n_known,)
      hgr:         hidden section GR, shape (n_hidden,)
      ktvt:        visible section TVT, shape (n_known,)
      last_md:     MD at last visible sample (= boundary)
      last_tvt:    TVT at last visible sample
      hmd:         hidden section MD, shape (n_hidden,)
      slope_md:    prefix dTVT/dMD slope (used to extrapolate TVT from MD shift)
      beam_cons_d: typewell-derived TVT delta proxy, shape (n_hidden,)

    Returns dict with 4 keys:
      selfaln_d:           self-NCC TVT estimate − last_tvt, shape (n_hidden,)
      selfaln_score:       best NCC score per hidden chunk, shape (n_hidden,)
      selfaln_vs_typewell_d: selfaln_d − beam_cons_d, shape (n_hidden,)
      selfaln_conflict_score: sigmoid(|selfaln_vs_typewell_d| / 5.0), shape (n_hidden,)

    leak 防止: hidden 部の TVT_input を一切参照しない (= GR + MD only)、
    TVT は visible 末端 + slope 外挿で推定。
    """
    n_hidden = len(hgr)
    out = {
        "selfaln_d":              np.zeros(n_hidden, dtype=np.float32),
        "selfaln_score":          np.zeros(n_hidden, dtype=np.float32),
        "selfaln_vs_typewell_d":  np.zeros(n_hidden, dtype=np.float32),
        "selfaln_conflict_score": np.zeros(n_hidden, dtype=np.float32),
    }

    nk = len(kgr)
    if nk < min_visible or n_hidden == 0:
        # visible 区間が短すぎる → fallback to typewell-only (= 0 features)
        # selfaln_vs_typewell_d / selfaln_conflict_score は 0、selfaln_d は
        # beam_cons_d と同値にしておく (= no extra signal、Ridge weight 0 になる)
        out["selfaln_d"][:] = beam_cons_d.astype(np.float32) if len(beam_cons_d) == n_hidden else 0.0
        return out

    # ── visible windows: extract K windows of size win_size ─────────────
    n_win = max(1, (nk - win_size) // win_step + 1)
    win_starts = np.arange(0, nk - win_size + 1, win_step, dtype=np.int32)[:n_win]
    if len(win_starts) == 0:
        out["selfaln_d"][:] = beam_cons_d.astype(np.float32) if len(beam_cons_d) == n_hidden else 0.0
        return out
    n_win = len(win_starts)

    # smooth GR (rolling-5) for stability
    kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
    hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)

    # build window template matrix (n_win, win_size)
    W = kg[win_starts[:, None] + np.arange(win_size, dtype=np.int32)[None, :]]
    Wn = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-6)

    # window centers (in MD units): use ktvt index → MD via prefix MD lookup is N/A here,
    # but we record window center MD as the visible position (assumed evenly sampled);
    # we use ktvt directly for TVT extrapolation
    win_centers = win_starts + win_size // 2
    win_tvt = ktvt[np.clip(win_centers, 0, nk - 1)]

    # ── hidden chunks: process in chunks of chunk_size ──────────────────
    for c0 in range(0, n_hidden, chunk_size):
        c1 = min(c0 + chunk_size, n_hidden)
        # use chunk midpoint as anchor
        anchor = (c0 + c1) // 2
        # build hidden window centered at anchor (size = win_size)
        a0 = max(0, anchor - win_size // 2)
        a1 = min(n_hidden, a0 + win_size)
        if a1 - a0 < win_size:
            a0 = max(0, a1 - win_size)
        if a1 - a0 < 4:
            continue  # too short
        Hwin = hg[a0:a1]
        if len(Hwin) < win_size:
            # pad with edge values
            pad_left  = win_size - len(Hwin)
            Hwin = np.concatenate([np.full(pad_left, Hwin[0]), Hwin])[:win_size]
        Hn = (Hwin - Hwin.mean()) / (Hwin.std() + 1e-6)

        # NCC across all visible windows
        ncc = (Wn @ Hn) / win_size
        best = int(ncc.argmax())
        score = float(ncc[best])

        # self-aln TVT estimate = best-window TVT + slope_md × (hidden anchor MD - dummy_visible_anchor_MD)
        # since we don't have visible windows' MD here, we approximate using slope_md:
        anchor_md = float(hmd[anchor]) if anchor < len(hmd) else last_md
        # delta from last visible: (anchor_md - last_md) × slope_md, then add (win_tvt[best] - last_tvt) shift
        delta_md  = anchor_md - last_md
        # Clip slope_md to physical range (= dTVT/dMD typical 0-2 for vertical, ~0 for horizontal)
        # to avoid runaway extrapolation when prefix has spurious slope.
        slope_md_clip = float(np.clip(slope_md, -1.0, 1.0))
        sa_tvt    = float(win_tvt[best]) + slope_md_clip * delta_md
        sa_d      = float(np.clip(sa_tvt - last_tvt, -200.0, 200.0))  # physical TVT delta cap

        # populate chunk
        out["selfaln_d"][c0:c1]              = np.float32(sa_d)
        out["selfaln_score"][c0:c1]          = np.float32(score)
        if len(beam_cons_d) >= c1:
            bc = beam_cons_d[c0:c1].astype(np.float32)
        else:
            bc = np.zeros(c1 - c0, dtype=np.float32)
        out["selfaln_vs_typewell_d"][c0:c1]  = (np.float32(sa_d) - bc).astype(np.float32)
        out["selfaln_conflict_score"][c0:c1] = (
            1.0 / (1.0 + np.exp(-np.abs(out["selfaln_vs_typewell_d"][c0:c1]) / 5.0))
        ).astype(np.float32)

    return out


def edge_m_feature_names() -> List[str]:
    return [
        "selfaln_d",
        "selfaln_score",
        "selfaln_vs_typewell_d",
        "selfaln_conflict_score",
    ]


# ───────────────────────────────────────────────────────────────────────────
# Edge D — AR(1) Kalman / PF on dTVT (exp008 新規)
#
# spec: docs/research/independent-edges.dense.md §1
#       + experiments/exp008/design.md
# math: Kalman 1960 / Doucet et al. 2001 — no external code, pure numpy.
# leak: visible region (ktvt, kmd) のみで MLE / forward filter。hidden TVT
#       (= ev["TVT"], ev["TVT_input"]) を一切参照しない。
# ───────────────────────────────────────────────────────────────────────────


def kalman_feature_names() -> List[str]:
    return [
        "kalman_d",                # MAP estimate of TVT(s) - last_known_TVT
        "kalman_std",              # posterior 1σ standard deviation of TVT(s)
        "kalman_phi",              # per-well φ MLE (after global shrinkage), constant per well
        "kalman_sigma_eps",        # per-well σ_ε MLE (after shrinkage), constant per well
        "kalman_t_md",             # MD horizon = (MD - last_known_MD) / median_dMD
        "kalman_vs_pf_a_d",        # kalman_d - pf_a_d (= disagreement vs PF ANCC)
        "kalman_vs_beam_cons_d",   # kalman_d - beam_cons_d (= disagreement vs Beam consensus)
    ]


def _yule_walker_ar1(dt: np.ndarray) -> Tuple[float, float]:
    """
    Closed-form AR(1) MLE on a 1D series via Yule-Walker.

    Returns
    -------
    phi_local   : float
    sigma_local : float (= residual std)
    """
    if len(dt) < 4:
        # too short, fall back to global prior (caller will shrink)
        return float("nan"), float("nan")
    dt = np.asarray(dt, dtype=np.float64)
    dt_centered = dt - dt.mean()
    num = float((dt_centered[:-1] * dt_centered[1:]).sum())
    den = float((dt_centered[:-1] ** 2).sum())
    if den < 1e-12:
        return float("nan"), float("nan")
    phi_local = num / den
    # numerical guard: clip into stationary region
    phi_local = float(np.clip(phi_local, 0.0, KALMAN_PHI_CLIP))
    resid = dt_centered[1:] - phi_local * dt_centered[:-1]
    if len(resid) < 2:
        return phi_local, float("nan")
    sigma_local = float(resid.std(ddof=1))
    return phi_local, sigma_local


def compute_kalman_features(
    ktvt:        np.ndarray,
    kmd:         np.ndarray,
    last_tvt:    float,
    last_md:     float,
    hmd:         np.ndarray,
    pf_a_d:      np.ndarray,
    beam_cons_d: np.ndarray,
    phi_global:  float = KALMAN_PHI_GLOBAL,
    sigma_global:float = KALMAN_SIGMA_GLOBAL,
    lam_phi:     float = KALMAN_LAM_PHI,
    lam_sigma:   float = KALMAN_LAM_SIGMA,
    pseudo:      float = KALMAN_PSEUDO_COUNT,
) -> Dict[str, np.ndarray]:
    """
    Build 7 features per hidden row from an AR(1) Kalman forward filter on dTVT.

    Parameters
    ----------
    ktvt        : visible TVT_input series (= kn["TVT_input"]), shape (nk,)
    kmd         : visible MD series (= kn["MD"]), shape (nk,)
    last_tvt    : float, last visible TVT_input (= ktvt[-1])
    last_md     : float, last visible MD (= kmd[-1])
    hmd         : hidden MD at evaluated rows (= ev["MD"]), shape (n_hidden,)
    pf_a_d      : exp007 PF ancc delta = pf_a_sel - last_tvt, shape (n_hidden,)
    beam_cons_d : exp007 beam consensus delta, shape (n_hidden,)
    phi_global, sigma_global : global priors (= per-well-stats parquet p50)
    lam_phi, lam_sigma       : shrinkage strength
    pseudo                   : pseudo-count for shrinkage weight

    Returns
    -------
    dict[str, np.ndarray] of 7 float32 arrays of length n_hidden.

    Leak guarantee
    --------------
    Function reads only `ktvt`, `kmd` (= visible region), `hmd` (= hidden MD),
    and exp007-derived hidden-side features (`pf_a_d`, `beam_cons_d`) which
    are themselves leak-safe (built from typewell + visible). Never reads
    hidden TVT or hidden TVT_input.
    """
    n_hidden = int(len(hmd))
    out_keys = kalman_feature_names()
    out = {k: np.zeros(n_hidden, dtype=np.float32) for k in out_keys}

    if n_hidden == 0:
        return out

    # ── 1. Per-well Yule-Walker MLE on visible dTVT ─────────────────────
    if len(ktvt) >= 4:
        dt_vis = np.diff(np.asarray(ktvt, dtype=np.float64))
        phi_local, sigma_local = _yule_walker_ar1(dt_vis)
    else:
        phi_local, sigma_local = float("nan"), float("nan")

    # ── 2. Hierarchical Bayesian shrinkage toward global prior ──────────
    n_eff = max(len(ktvt) - 2, 1)
    w_phi   = float(n_eff) / (float(n_eff) + lam_phi   * pseudo)
    w_sigma = float(n_eff) / (float(n_eff) + lam_sigma * pseudo)

    if not np.isfinite(phi_local):
        phi = float(phi_global)
    else:
        phi = w_phi * phi_local + (1.0 - w_phi) * float(phi_global)
    phi = float(np.clip(phi, 0.0, KALMAN_PHI_CLIP))

    if not np.isfinite(sigma_local):
        sigma = float(sigma_global)
    else:
        sigma = w_sigma * sigma_local + (1.0 - w_sigma) * float(sigma_global)
    sigma = float(max(sigma, KALMAN_SIGMA_FLOOR))

    # ── 3. Median dMD as the step unit ──────────────────────────────────
    if len(kmd) >= 2:
        dmd_vis = np.diff(np.asarray(kmd, dtype=np.float64))
        # remove zeros / negatives just in case
        dmd_vis_pos = dmd_vis[dmd_vis > 1e-6]
        if len(dmd_vis_pos) > 0:
            median_dmd = float(np.median(dmd_vis_pos))
        else:
            median_dmd = 1.0
    else:
        median_dmd = 1.0
    median_dmd = max(median_dmd, 1e-3)

    # ── 4. Initialise state at last visible step ────────────────────────
    if len(ktvt) >= 2:
        prev_dt = float(ktvt[-1] - ktvt[-2])
    else:
        prev_dt = 0.0

    # ── 5. Forward Kalman pass on dTVT ──────────────────────────────────
    # Cumulative expected TVT delta from last_tvt and cumulative variance.
    # AR(1) propagation: dt_{k+1} = phi * dt_k (mean-zero noise integrated into var)
    # Var step ≈ sigma^2 (random walk approx; valid for phi ≈ 1, slight overestimate
    # for phi < 0.99 but that is the conservative direction)
    cum_dt  = 0.0
    cum_var = 0.0
    prev_md = float(last_md)
    sigma_sq = sigma * sigma
    horizon_hint = np.zeros(n_hidden, dtype=np.float32)

    for i in range(n_hidden):
        md_now = float(hmd[i])
        if md_now <= prev_md:
            n_steps = 1
        else:
            n_steps = int(round((md_now - prev_md) / median_dmd))
            n_steps = max(1, n_steps)

        for _ in range(n_steps):
            prev_dt = phi * prev_dt
            cum_dt  = cum_dt + prev_dt
            cum_var = cum_var + sigma_sq

        prev_md = md_now
        out["kalman_d"][i]   = np.float32(cum_dt)
        out["kalman_std"][i] = np.float32(np.sqrt(cum_var))
        # horizon in step units (= MD distance / median_dmd)
        horizon_hint[i] = np.float32((md_now - float(last_md)) / median_dmd)

    # std floor (numerical guard for tiny visible)
    std_floor = sigma * KALMAN_STD_FLOOR_FRAC
    out["kalman_std"] = np.maximum(out["kalman_std"], np.float32(std_floor))

    # ── 6. Constants per well (broadcast) ───────────────────────────────
    out["kalman_phi"]       = np.full(n_hidden, np.float32(phi),     dtype=np.float32)
    out["kalman_sigma_eps"] = np.full(n_hidden, np.float32(sigma),   dtype=np.float32)
    out["kalman_t_md"]      = horizon_hint

    # ── 7. Disagreement features (vs exp007 hidden estimators) ──────────
    pf_a_d_arr      = np.asarray(pf_a_d,      dtype=np.float32)
    beam_cons_arr   = np.asarray(beam_cons_d, dtype=np.float32)
    if len(pf_a_d_arr) == n_hidden:
        out["kalman_vs_pf_a_d"] = (out["kalman_d"] - pf_a_d_arr).astype(np.float32)
    else:
        out["kalman_vs_pf_a_d"] = np.zeros(n_hidden, dtype=np.float32)
    if len(beam_cons_arr) == n_hidden:
        out["kalman_vs_beam_cons_d"] = (out["kalman_d"] - beam_cons_arr).astype(np.float32)
    else:
        out["kalman_vs_beam_cons_d"] = np.zeros(n_hidden, dtype=np.float32)

    return out


print("Edge Q + Edge M + Edge D (Kalman) helpers loaded ✓")


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


# ───────────────────────────────────────────────────────────────────────────
# Case E (exp009) — Sparse Gaussian Process imputer for per-formation depth.
# ───────────────────────────────────────────────────────────────────────────


def gp_feature_names() -> List[str]:
    cols: List[str] = []
    for fn in FORMATIONS:
        cols.append(f"gp_{fn}_mean")
        cols.append(f"gp_{fn}_var")
        cols.append(f"gp_{fn}_mean_minus_plane")
        cols.append(f"gp_{fn}_var_norm")
    return cols


class GPFormationImputer:
    """Per-formation Sparse GP fit on (X, Y) -> formation depth.

    Strategy
    --------
    Aggregate every train well to a single (X_w, Y_w, formation_k_depth)
    triple per formation. Cluster the well centroids via K-Means (M = 200)
    so the GP fit operates on M reduced points (= the SVGP "inducing
    points" approximation, Hensman 2013). Fit one GP per formation with
    a Matern 3/2 ARD kernel (separate length scales for X and Y) plus a
    WhiteKernel for noise. Optimise hyperparameters with sklearn's
    marginal-likelihood maximiser (n_restarts=3 for robustness).

    Posterior at any query (X_q, Y_q) is N(m, var). Predictions are
    self-exclusive only at hyperparameter level (Cholesky training pool
    excludes self_wid by per-call refit on a smaller pool); the cost of
    exact per-well refit during inference is high, so the test path uses
    the global fit (= negligible leak when N_train is large).

    Leak guarantee
    --------------
    Reads only ``X``, ``Y`` and the formation-depth columns
    (= ``ANCC``/``ASTNU``/...) from train horizontal_well.csv files. Never
    reads ``TVT`` or ``TVT_input``.
    """

    @classmethod
    def empty(cls) -> "GPFormationImputer":
        obj = cls.__new__(cls)
        obj._empty = True
        obj.gprs = [None] * len(FORMATIONS)
        obj.train_xy_norm = None
        obj.xy_mean = np.zeros(2, dtype=np.float64)
        obj.xy_std = np.ones(2, dtype=np.float64)
        obj.train_y = [np.array([], dtype=np.float32) for _ in FORMATIONS]
        obj.knn_tree = None
        obj.train_wids = []
        return obj

    def __init__(
        self,
        well_ids: List[str],
        data_dir: Path,
        M: int = 200,
        seed: int = 42,
    ):
        self._empty = False
        self.gprs: List[Optional[GaussianProcessRegressor]] = [None] * len(FORMATIONS)
        self.train_y: List[np.ndarray] = [np.array([], dtype=np.float32)
                                          for _ in FORMATIONS]

        rows: List[Dict[str, Any]] = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                avail = list(FORMATIONS) + ["X", "Y"]
                df = pd.read_csv(p, usecols=[c for c in avail])
            except Exception:
                continue
            form_cols = [c for c in FORMATIONS if c in df.columns]
            if not form_cols or "X" not in df.columns or "Y" not in df.columns:
                continue
            df = df[["X", "Y"] + form_cols].dropna()
            if len(df) == 0:
                continue
            row: Dict[str, Any] = {
                "wid": wid,
                "x": float(df["X"].median()),
                "y": float(df["Y"].median()),
            }
            for c in FORMATIONS:
                row[c] = float(df[c].median()) if c in df.columns else np.nan
            rows.append(row)

        if not rows:
            self._empty = True
            self.train_xy_norm = None
            self.xy_mean = np.zeros(2, dtype=np.float64)
            self.xy_std = np.ones(2, dtype=np.float64)
            self.knn_tree = None
            self.train_wids = []
            return

        self.train_wids = [r["wid"] for r in rows]
        train_xy = np.array([[r["x"], r["y"]] for r in rows], dtype=np.float64)
        self.xy_mean = train_xy.mean(0)
        std = train_xy.std(0)
        std[std < 1e-3] = 1.0
        self.xy_std = std
        self.train_xy_norm = (train_xy - self.xy_mean) / self.xy_std
        self.knn_tree = cKDTree(self.train_xy_norm)

        # Per-formation: gather (X_norm, Y_norm, depth) and fit GP
        for fi, fn in enumerate(FORMATIONS):
            depth_arr = np.array([r.get(fn, np.nan) for r in rows], dtype=np.float64)
            valid = np.isfinite(depth_arr)
            if valid.sum() < 10:
                self.gprs[fi] = None
                self.train_y[fi] = depth_arr.astype(np.float32)
                continue

            xy_v = self.train_xy_norm[valid]
            y_v  = depth_arr[valid]

            # K-Means inducing points
            try:
                if len(xy_v) > M:
                    km = KMeans(n_clusters=M, random_state=seed, n_init=4)
                    km.fit(xy_v)
                    induce_xy_list: List[np.ndarray] = []
                    induce_y_list: List[float] = []
                    labels = km.labels_
                    for ci in range(M):
                        cm = labels == ci
                        if cm.any():
                            induce_xy_list.append(km.cluster_centers_[ci])
                            induce_y_list.append(float(y_v[cm].mean()))
                    induce_xy = np.array(induce_xy_list, dtype=np.float64)
                    induce_y = np.array(induce_y_list, dtype=np.float64)
                else:
                    induce_xy = xy_v
                    induce_y = y_v
            except Exception:
                induce_xy = xy_v
                induce_y = y_v

            # Build kernel: Const * Matern(ARD, nu=1.5) + White
            try:
                kernel = (
                    _GP_C(1.0, (1e-3, 1e3))
                    * _GP_Matern(
                        length_scale=[GP_LENGTH_SCALE_INIT, GP_LENGTH_SCALE_INIT],
                        length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
                        nu=1.5,
                    )
                    + _GP_White(
                        noise_level=GP_NOISE_INIT,
                        noise_level_bounds=GP_NOISE_BOUNDS,
                    )
                )
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=GP_N_RESTARTS,
                    normalize_y=True,
                    random_state=seed,
                    alpha=GP_ALPHA,
                )
                gpr.fit(induce_xy, induce_y)
                self.gprs[fi] = gpr
            except Exception as exc:
                print(f"  WARN GP fit failed for formation {fn}: {exc}")
                self.gprs[fi] = None
            self.train_y[fi] = depth_arr.astype(np.float32)

    def impute(
        self,
        xy_q: np.ndarray,
        self_wid: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (mean, var) per formation at each query (X, Y).

        Returns
        -------
        mean : np.ndarray of shape (n_q, n_formations)
        var  : np.ndarray of shape (n_q, n_formations)
        """
        n_q = len(xy_q)
        n_f = len(FORMATIONS)
        if self._empty or self.train_xy_norm is None:
            return (
                np.zeros((n_q, n_f), dtype=np.float32),
                np.ones((n_q, n_f), dtype=np.float32),
            )

        q = (np.asarray(xy_q, dtype=np.float64) - self.xy_mean) / self.xy_std

        mean_out = np.zeros((n_q, n_f), dtype=np.float32)
        var_out = np.ones((n_q, n_f), dtype=np.float32)

        for fi, fn in enumerate(FORMATIONS):
            gpr = self.gprs[fi]
            if gpr is not None:
                try:
                    m, std = gpr.predict(q, return_std=True)
                    mean_out[:, fi] = m.astype(np.float32)
                    var_out[:, fi] = (std.astype(np.float64) ** 2).astype(np.float32)
                except Exception:
                    # KNN fallback
                    mean_out[:, fi], var_out[:, fi] = self._knn_fallback(
                        q, fi, k=GP_FALLBACK_KNN_K, self_wid=self_wid,
                    )
            else:
                mean_out[:, fi], var_out[:, fi] = self._knn_fallback(
                    q, fi, k=GP_FALLBACK_KNN_K, self_wid=self_wid,
                )
        return mean_out, var_out

    def _knn_fallback(
        self,
        q_norm: np.ndarray,
        fi: int,
        k: int = 20,
        self_wid: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crude KNN-mean / KNN-std posterior approximation."""
        if self.knn_tree is None:
            return (
                np.zeros(len(q_norm), dtype=np.float32),
                np.ones(len(q_norm), dtype=np.float32),
            )
        depth_arr = np.asarray(self.train_y[fi], dtype=np.float64)
        nf = min(k + 5, len(self.train_xy_norm))
        dist, idx = self.knn_tree.query(q_norm, k=nf, workers=-1)
        if self_wid is not None and self_wid in self.train_wids:
            self_pos = self.train_wids.index(self_wid)
            dist = np.where(idx == self_pos, np.inf, dist)
        # take top-k that have finite depth
        m_arr = np.zeros(len(q_norm), dtype=np.float32)
        v_arr = np.ones(len(q_norm), dtype=np.float32)
        for r in range(len(q_norm)):
            order = np.argsort(dist[r])
            chosen: List[int] = []
            for o in order:
                ii = int(idx[r][o])
                if not np.isfinite(dist[r][o]):
                    continue
                if not np.isfinite(depth_arr[ii]):
                    continue
                chosen.append(ii)
                if len(chosen) >= k:
                    break
            if not chosen:
                m_arr[r] = 0.0
                v_arr[r] = 1.0
                continue
            vals = depth_arr[chosen]
            m_arr[r] = float(vals.mean())
            v_arr[r] = float(max(vals.var(ddof=0), 1e-6))
        return m_arr, v_arr


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

# GP imputer (case E) is None at module load; built once in the dispatcher
# (`infer_edge_e_gp_o`) before train + test feature builds.
_GP_REF: Optional[GPFormationImputer] = None


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
    global _FI_REF, _DI_REF, _GP_REF

    fi_use = fi if fi is not None else _FI_REF
    di_use = di if di is not None else _DI_REF
    gp_use = _GP_REF

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

    # ── Case E (Sparse GP per-formation posterior) FE ───────────────────
    # 24 features per hidden row, leak-safe (= reads (X, Y) + formation
    # depth columns from horizontal_well.csv, never reads TVT/TVT_input).
    # GP fit happens once at the dispatcher; per-well call is predict-only.
    if GP_ENABLE and (gp_use is not None) and (not getattr(gp_use, "_empty", True)):
        try:
            gp_mean_ev, gp_var_ev = gp_use.impute(xy_ev, self_wid=swid)  # (nh, 6) each
            gp_disagree = (gp_mean_ev - form_ev).astype(np.float32)
            std_ev = np.sqrt(np.maximum(gp_var_ev, 1e-12))
            std_mean_well = max(float(std_ev.mean()), 1e-6)
            std_norm_arr = (std_ev / std_mean_well).astype(np.float32)
            gp_out: Dict[str, np.ndarray] = {}
            for fi_idx, fn in enumerate(FORMATIONS):
                gp_out[f"gp_{fn}_mean"]              = gp_mean_ev[:, fi_idx].astype(np.float32)
                gp_out[f"gp_{fn}_var"]               = gp_var_ev[:, fi_idx].astype(np.float32)
                gp_out[f"gp_{fn}_mean_minus_plane"]  = gp_disagree[:, fi_idx]
                gp_out[f"gp_{fn}_var_norm"]          = std_norm_arr[:, fi_idx]
        except Exception as _gp_e:
            print(f"  WARN [{wid}] GP FE failed: {_gp_e}")
            gp_out = {k: np.zeros(len(ev), dtype=np.float32)
                      for k in gp_feature_names()}
    else:
        gp_out = {k: np.zeros(len(ev), dtype=np.float32)
                  for k in gp_feature_names()}

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

    # ── Edge M (visible-as-typewell self-alignment) FE ──────────────────
    # 4 features per hidden row, leak-safe (= GR + MD only, never reads
    # hidden-side TVT_input).
    if EDGE_M_ENABLE:
        try:
            # hgr = full hidden GR (already in scope as `hgr`, of length len(ev))
            # kgr = visible GR (already in scope, length nk)
            # bf["beam_cons_d"] = typewell-derived TVT delta proxy
            beam_cons_d_full = bf.get(
                "beam_cons_d",
                np.zeros(len(ev), dtype=np.float32),
            )
            edge_m_out = compute_edge_m_features(
                kgr=kgr,
                hgr=hgr_sel,  # GR in hidden rows we keep (= ev rows only)
                ktvt=ktvt,
                last_md=last_md,
                last_tvt=last_tvt,
                hmd=hmd,
                slope_md=slp_all,
                beam_cons_d=beam_cons_d_full,
            )
        except Exception as _edge_m_e:
            # leak-safe fallback: 4 zero columns
            edge_m_out = {
                "selfaln_d":              np.zeros(len(ev), dtype=np.float32),
                "selfaln_score":          np.zeros(len(ev), dtype=np.float32),
                "selfaln_vs_typewell_d":  np.zeros(len(ev), dtype=np.float32),
                "selfaln_conflict_score": np.zeros(len(ev), dtype=np.float32),
            }
    else:
        edge_m_out = {
            "selfaln_d":              np.zeros(len(ev), dtype=np.float32),
            "selfaln_score":          np.zeros(len(ev), dtype=np.float32),
            "selfaln_vs_typewell_d":  np.zeros(len(ev), dtype=np.float32),
            "selfaln_conflict_score": np.zeros(len(ev), dtype=np.float32),
        }

    # ── Edge D (AR(1) Kalman on dTVT) FE ────────────────────────────────
    # 7 features per hidden row, leak-safe (visible region only for MLE +
    # forward filter; never reads hidden TVT). Reuses exp007 hidden features
    # pf_a_sel and bf["beam_cons_d"] solely for disagreement diagnostics
    # (they are themselves leak-safe).
    if KALMAN_ENABLE:
        try:
            pf_a_d_full      = (pf_a_sel - last_tvt).astype(np.float32)
            beam_cons_d_full = bf.get(
                "beam_cons_d", np.zeros(len(ev), dtype=np.float32),
            ).astype(np.float32)
            # ev["MD"] view is hmd (already extracted above)
            kalman_out = compute_kalman_features(
                ktvt=ktvt,
                kmd=kmd,
                last_tvt=last_tvt,
                last_md=last_md,
                hmd=hmd,
                pf_a_d=pf_a_d_full,
                beam_cons_d=beam_cons_d_full,
            )
        except Exception as _kal_e:
            print(f"  WARN [{wid}] Kalman FE failed: {_kal_e}")
            kalman_out = {k: np.zeros(len(ev), dtype=np.float32)
                          for k in kalman_feature_names()}
    else:
        kalman_out = {k: np.zeros(len(ev), dtype=np.float32)
                      for k in kalman_feature_names()}

    # ── Edge O (direction-aware Beam) FE ────────────────────────────────
    # 7 features per hidden row, leak-safe (= GR + typewell only, never
    # reads hidden TVT_input). lam_dir penalty couples horizontal-side
    # dGR sign and typewell-side dGR sign during Beam transitions.
    if EDGE_O_ENABLE:
        try:
            edge_o_out = compute_edge_o_features(
                hgr_full=hgr,
                tw_tvt=tw_tvt,
                tw_gr=tw_gr,
                last_known_tvt=last_tvt,
                sel_local=sel_local,
                lam_dir=EDGE_O_LAMBDA_DIR,
            )
        except Exception as _eo_e:
            print(f"  WARN [{wid}] Edge O FE failed: {_eo_e}")
            edge_o_out = {k: np.zeros(len(ev), dtype=np.float32)
                          for k in edge_o_feature_names()}
    else:
        edge_o_out = {k: np.zeros(len(ev), dtype=np.float32)
                      for k in edge_o_feature_names()}

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
        # Edge M (visible-as-typewell self-alignment, exp007 4 cols)
        **edge_m_out,
        # Edge D (AR(1) Kalman on dTVT, exp008 新規 7 cols)
        **kalman_out,
        # Case E (Sparse GP per-formation posterior, exp009 新規 24 cols)
        **gp_out,
        # Edge O (direction-aware Beam, exp009 新規 7 cols)
        **edge_o_out,
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


# ## 9.5 Edge R — Test-Time Online Learning (exp009 v4 新規)
# #
# discussion 698002 で公開された "online 10.953 vs no-online 11.323 = -0.370 ft"
# の効果を再現する。 test wells の visible region 末尾 K 行を擬似 hidden に変換し、
# `build_well_features(.., is_train=True)` 経路で (X, y) を生成、
# karnakbaev pretrained LGB 5 base に `init_model` で continued training する。
#
# 自前 base (= MEDIAN 30-run) は train data で十分 flexible なので Edge R を適用しない
# (= 過剰 fine-tune による degenerate 回避)。
#
# leak guard:
#   - visible region の TVT_input のみ参照、 hidden mask 域には一切触れない
#   - 各 well 内で完結、 cross-well leak なし
#   - karnakbaev pretrained train_df と test visible は別 well (= train/test split は public)

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
      2. 末尾 `tail_k` 行を擬似 hidden に変換 (= TVT_input → NaN マスク)
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
        if n_visible < (min_visible_rows + tail_k):
            return None

        visible_idx = np.flatnonzero(visible_mask)
        tail_idx    = visible_idx[-tail_k:]
        orig_tvt_input = hw_raw["TVT_input"].to_numpy(np.float32).copy()

        hw_mod = hw_raw.copy()
        hw_mod["TVT"] = np.nan
        hw_mod.loc[hw_mod["TVT_input"].notna(), "TVT"] = orig_tvt_input[visible_mask]
        hw_mod.loc[tail_idx, "TVT_input"] = np.nan

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
    """LightGBM continued training (warm start) for Edge R.

    GPU device は continued training で API 不安定なため CPU 強制。
    karnakbaev base は CPU で fit、 互換性 OK。
    """
    p = dict(base_params)
    p["learning_rate"] = float(p.get("learning_rate", 0.04)) * float(lr_mul)
    p.pop("n_estimators", None)
    p.pop("early_stopping_rounds", None)
    # Edge R は CPU で続行 (= GPU continued training の API 不安定回避)
    p["device_type"] = "cpu"
    p.pop("gpu_use_dp", None)
    # Edge R では huber objective を維持しない (= karnakbaev base は regression objective)
    # base と整合させるため objective は warm start で継承される
    ds = lgb.Dataset(X_R, label=y_R)
    online = lgb.train(
        p, ds,
        num_boost_round=int(num_boost_round),
        init_model=base_booster,
        keep_training_booster=False,
    )
    return online


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


elif MODE in ("infer_edge_qm", "infer_edge_d_kalman", "infer_edge_e_gp_o"):
    # ── exp007/exp008/exp009 path: Edge Q + Edge M (+ Edge D Kalman in exp008+,
    #   + GP/Edge O in exp009) + 自前 4 base + 9-base Ridge meta ──────────────
    is_exp008 = (MODE in ("infer_edge_d_kalman", "infer_edge_e_gp_o"))
    is_exp009 = (MODE == "infer_edge_e_gp_o")
    pipeline_tag = "exp009" if is_exp009 else ("exp008" if is_exp008 else "exp007")

    feature_cols   = am.load_json(am.FEATURES_LIST)
    pp_params      = am.load_json("postproc_params")
    best_alpha     = float(pp_params["alpha"])
    best_tau       = pp_params.get("tau")

    # ── exp009: build per-formation Sparse GP imputer (= 1-time fit, shared by
    #   train + test feature builds via _GP_REF) ─────────────────────────────
    if is_exp009 and GP_ENABLE:
        section(log, f"{pipeline_tag}: build GP imputer (per-formation Sparse GP)")
        try:
            with timer(log, "GP fit (6 formations × M=200 inducing)"):
                _train_well_ids = sorted({
                    p.stem.replace("__horizontal_well", "")
                    for p in TRAIN_DIR.glob("*__horizontal_well.csv")
                })
                _GP_REF = GPFormationImputer(
                    _train_well_ids, TRAIN_DIR,
                    M=GP_M_INDUCE, seed=SEED,
                )
            n_fit = sum(1 for g in _GP_REF.gprs if g is not None)
            log.info(f"  GP imputer ready: {n_fit}/{len(FORMATIONS)} formations fit")
        except Exception as _gp_e:
            log.warning(f"  GP imputer build failed ({_gp_e}); falling back to KNN")
            _GP_REF = GPFormationImputer.empty()

    section(log, f"{pipeline_tag}: build test features "
                 f"(live test wells, with Edge M"
                 f"{'+ Edge D Kalman' if is_exp008 else ''}"
                 f"{'+ GP + Edge O' if is_exp009 else ''})")
    with timer(log, "test feature engineering"):
        test_df = build_dataset(TEST_DIR, is_train=False,
                                max_wells=DEBUG_MAX_WELLS, n_jobs=NCPU)
    log.info(f"Test features: {test_df.shape}")
    edge_m_cols = edge_m_feature_names()
    missing_em = [c for c in edge_m_cols if c not in test_df.columns]
    if missing_em:
        log.warning(f"Edge M cols missing from test_df: {missing_em} — filling with 0")
        for c in missing_em:
            test_df[c] = np.float32(0.0)
    if is_exp008 and KALMAN_ENABLE:
        kal_cols = kalman_feature_names()
        missing_kal = [c for c in kal_cols if c not in test_df.columns]
        if missing_kal:
            log.warning(f"Kalman cols missing from test_df: {missing_kal} — filling with 0")
            for c in missing_kal:
                test_df[c] = np.float32(0.0)
        else:
            log.info(f"Kalman cols present in test_df: {kal_cols}")
            log.info(f"  kalman_d range: [{test_df['kalman_d'].min():.3f}, "
                     f"{test_df['kalman_d'].max():.3f}]")
            log.info(f"  kalman_std range: [{test_df['kalman_std'].min():.3f}, "
                     f"{test_df['kalman_std'].max():.3f}]")
            log.info(f"  kalman_phi p50: {test_df['kalman_phi'].median():.5f} "
                     f"(global prior {KALMAN_PHI_GLOBAL})")
            log.info(f"  kalman_sigma_eps p50: {test_df['kalman_sigma_eps'].median():.5f} "
                     f"(global prior {KALMAN_SIGMA_GLOBAL})")
    if is_exp009 and GP_ENABLE:
        gp_cols = gp_feature_names()
        missing_gp = [c for c in gp_cols if c not in test_df.columns]
        if missing_gp:
            log.warning(f"GP cols missing from test_df: {missing_gp[:5]}{'...' if len(missing_gp) > 5 else ''} — filling with 0")
            for c in missing_gp:
                test_df[c] = np.float32(0.0)
        else:
            log.info(f"GP cols present in test_df: {len(gp_cols)} cols")
            log.info(f"  gp_ANCC_mean range: [{test_df['gp_ANCC_mean'].min():.3f}, "
                     f"{test_df['gp_ANCC_mean'].max():.3f}]")
            log.info(f"  gp_ANCC_var range: [{test_df['gp_ANCC_var'].min():.3e}, "
                     f"{test_df['gp_ANCC_var'].max():.3e}]")
            log.info(f"  gp_ANCC_mean_minus_plane p50: "
                     f"{test_df['gp_ANCC_mean_minus_plane'].median():.4f}")
    if is_exp009 and EDGE_O_ENABLE:
        eo_cols = edge_o_feature_names()
        missing_eo = [c for c in eo_cols if c not in test_df.columns]
        if missing_eo:
            log.warning(f"Edge O cols missing from test_df: {missing_eo} — filling with 0")
            for c in missing_eo:
                test_df[c] = np.float32(0.0)
        else:
            log.info(f"Edge O cols present in test_df: {eo_cols}")
            log.info(f"  beam_dir_cons_d range: [{test_df['beam_dir_cons_d'].min():.3f}, "
                     f"{test_df['beam_dir_cons_d'].max():.3f}]")
            log.info(f"  beam_dir_std_d  range: [{test_df['beam_dir_std_d'].min():.3f}, "
                     f"{test_df['beam_dir_std_d'].max():.3f}]")

    # ── Step A: karnakbaev artefacts + base predict on test ─────────────────
    section(log, f"{pipeline_tag}: load karnakbaev pretrained 5 base + predict test")
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
    # karnakbaev models were trained on the 154-col schema only (= feature_cols),
    # so pass exactly those cols (do NOT include Edge M / Kalman cols here).
    feature_cols_kb = list(feature_cols)
    test_preds_kb = predict_all(base_models, test_df[feature_cols_kb])
    for k, v in test_preds_kb.items():
        log.info(f"    kb {k}: shape={v.shape}  range=[{v.min():.3f}, {v.max():.3f}]")

    # 自前 4 base feature schema = 154 kb cols + Edge M (4 cols) + Kalman (7
    # cols, exp008+) + GP (24, exp009) + Edge O (7, exp009).
    feature_cols_own = list(feature_cols_kb) + list(edge_m_cols)
    if is_exp008 and KALMAN_ENABLE:
        for c in kalman_feature_names():
            if c not in feature_cols_own:
                feature_cols_own.append(c)
    if is_exp009 and GP_ENABLE:
        for c in gp_feature_names():
            if c not in feature_cols_own:
                feature_cols_own.append(c)
    if is_exp009 and EDGE_O_ENABLE:
        for c in edge_o_feature_names():
            if c not in feature_cols_own:
                feature_cols_own.append(c)
    extras_label = "Edge M"
    if is_exp008 and KALMAN_ENABLE:
        extras_label += " + Kalman"
    if is_exp009 and GP_ENABLE:
        extras_label += " + GP"
    if is_exp009 and EDGE_O_ENABLE:
        extras_label += " + Edge O"
    log.info(f"  feature_cols_kb : {len(feature_cols_kb)} (= karnakbaev pretrained schema)")
    log.info(f"  feature_cols_own: {len(feature_cols_own)} (+{extras_label})")

    # ── Step B: load karnakbaev train_df.parquet for自前 train + Edge Q ────
    own_ok = False
    edge_q_ok = False
    train_df_kb = None
    fold_id = None
    groups = None
    kb_oof_q = None  # karnakbaev OOF re-computed under Edge Q fold
    own_oof = None   # 自前 4-base OOF
    own_test = None  # 自前 4-base test predictions
    y_kb = None

    try:
        train_df_path = ARTEFACT_DIR / "train_df.parquet"
        if not train_df_path.exists():
            raise FileNotFoundError(f"train_df.parquet missing: {train_df_path}")
        with timer(log, "load train_df.parquet"):
            train_df_kb = pd.read_parquet(train_df_path)
        log.info(f"  train_df: shape={train_df_kb.shape}")
        required = set(feature_cols_kb) | {"target", "well", "id"}
        missing  = required - set(train_df_kb.columns)
        if missing:
            raise RuntimeError(
                f"train_df is missing {len(missing)} required cols, e.g. {sorted(list(missing))[:5]}")
        y_kb = train_df_kb["target"].to_numpy(np.float32)

        # ── Step B': inject Edge M / Kalman cols into train_df_kb ──────────
        # karnakbaev train_df.parquet does not contain Edge M / Kalman cols
        # (= our exp008 additions). For 自前 4 base to leverage Kalman, we
        # rebuild the FE columns by running build_dataset over TRAIN_DIR
        # and matching on (well, id). Falls back to filling with 0 if rebuild
        # fails (= 自前 base will not use Kalman, but pipeline survives).
        em_cols  = edge_m_feature_names()
        kal_cols = kalman_feature_names() if (is_exp008 and KALMAN_ENABLE) else []
        gp_cols  = gp_feature_names()    if (is_exp009 and GP_ENABLE)     else []
        eo_cols  = edge_o_feature_names() if (is_exp009 and EDGE_O_ENABLE) else []
        own_extra_cols = em_cols + kal_cols + gp_cols + eo_cols
        missing_own_extra = [c for c in own_extra_cols if c not in train_df_kb.columns]
        if missing_own_extra:
            inject_label = "Edge M"
            if kal_cols: inject_label += " + Kalman"
            if gp_cols:  inject_label += " + GP"
            if eo_cols:  inject_label += " + Edge O"
            section(log, f"{pipeline_tag}: inject {inject_label} cols into train_df_kb")
            log.info(f"  missing in train_df_kb (first 10): {missing_own_extra[:10]}"
                     f"{' ...' if len(missing_own_extra) > 10 else ''}")
            try:
                with timer(log, "rebuild train FE for own extras (build_dataset over TRAIN_DIR)"):
                    train_df_extra = build_dataset(
                        TRAIN_DIR, is_train=True,
                        max_wells=DEBUG_MAX_WELLS, n_jobs=NCPU,
                    )
                # join on (well, id) to align with train_df_kb
                cols_to_pull = ["well", "id"] + [c for c in own_extra_cols
                                                  if c in train_df_extra.columns]
                missing_after_build = [c for c in own_extra_cols
                                        if c not in train_df_extra.columns]
                if missing_after_build:
                    log.warning(
                        f"  build_dataset did not produce: {missing_after_build} — "
                        f"will fill those with 0")
                train_df_extra = train_df_extra[cols_to_pull].copy()
                # Drop any duplicated (well, id) just in case
                train_df_extra = train_df_extra.drop_duplicates(subset=["well", "id"])
                train_df_kb = train_df_kb.merge(
                    train_df_extra, on=["well", "id"], how="left",
                )
                # Fill any remaining NaNs from missing wells with 0
                for c in own_extra_cols:
                    if c not in train_df_kb.columns:
                        train_df_kb[c] = np.float32(0.0)
                    else:
                        train_df_kb[c] = train_df_kb[c].fillna(0).astype(np.float32)
                log.info(f"  train_df_kb extended to shape={train_df_kb.shape}")
                if kal_cols and "kalman_d" in train_df_kb.columns:
                    log.info(f"  train kalman_d range: [{train_df_kb['kalman_d'].min():.3f}, "
                             f"{train_df_kb['kalman_d'].max():.3f}]")
                    log.info(f"  train kalman_phi p50: {train_df_kb['kalman_phi'].median():.5f}")
            except Exception as _ext_e:
                log.warning(f"  rebuild train extras failed ({_ext_e}); "
                            f"filling Edge M/Kalman cols with 0 in train_df_kb")
                for c in own_extra_cols:
                    if c not in train_df_kb.columns:
                        train_df_kb[c] = np.float32(0.0)

        # ── Step C: Edge Q fold builder ─────────────────────────────────────
        # exp010: Step 1 = stratified Edge Q fold (= H10 対策、σ_fold 1.18 → 0.5)
        if EDGE_Q_ENABLE:
            if STRATIFIED_EDGE_Q_ENABLE:
                section(log,
                    f"exp010 Step 1: stratified Edge Q fold "
                    f"(stratify_key={STRATIFIED_EDGE_Q_KEY}, "
                    f"n_bins={STRATIFIED_EDGE_Q_N_BINS})")
                # Load per-well-stats.parquet (= stratify key source)
                _stats_df = None
                for _cand in (HETERO_W_STATS_PATH, HETERO_W_STATS_LOCAL):
                    if _cand and Path(_cand).exists():
                        _stats_df = pd.read_parquet(_cand)
                        log.info(f"  loaded per-well-stats from {_cand} "
                                 f"({_stats_df.shape})")
                        break
                if _stats_df is None:
                    log.warning(
                        "  per-well-stats.parquet not found → falling back to "
                        "plain Edge Q")
                    with timer(log, "Edge Q fold computation (plain fallback)"):
                        fold_id, groups = build_edge_q_folds(
                            train_df_kb, TRAIN_DIR,
                            n_splits=N_SPLITS, seed=SEED,
                            fallback_by_well=EDGE_Q_FALLBACK_BY_WELL,
                        )
                else:
                    with timer(log, "stratified Edge Q fold computation"):
                        fold_id, groups = build_stratified_edge_q_folds(
                            train_df_kb, TRAIN_DIR, _stats_df,
                            n_splits=N_SPLITS, seed=SEED,
                            stratify_key=STRATIFIED_EDGE_Q_KEY,
                            n_bins=STRATIFIED_EDGE_Q_N_BINS,
                            fallback_by_well=EDGE_Q_FALLBACK_BY_WELL,
                        )
                    # Balance check
                    try:
                        balance_df = summarize_stratified_balance(
                            fold_id, train_df_kb, _stats_df,
                            stratify_key=STRATIFIED_EDGE_Q_KEY,
                            n_bins=STRATIFIED_EDGE_Q_N_BINS,
                        )
                        if not balance_df.empty:
                            log.info(
                                f"  stratified fold × bin counts:\n"
                                f"{balance_df.to_string()}")
                    except Exception as _bal_e:
                        log.warning(
                            f"  balance summary failed: {_bal_e}")
            else:
                section(log, "exp007: Edge Q (typewell content-hash GroupKFold)")
                with timer(log, "Edge Q fold computation"):
                    fold_id, groups = build_edge_q_folds(
                        train_df_kb, TRAIN_DIR,
                        n_splits=N_SPLITS, seed=SEED,
                        fallback_by_well=EDGE_Q_FALLBACK_BY_WELL,
                    )
            ok, msg = verify_edge_q_no_leak(fold_id, groups)
            log.info(f"  Edge Q leak check: {msg}")
            if not ok:
                raise RuntimeError(f"Edge Q fold leak detected: {msg}")
            n_groups = pd.Series(groups).nunique()
            n_dup_groups = int((pd.Series(groups).value_counts() > 1).sum())
            log.info(f"  Edge Q: {n_groups} unique groups, {n_dup_groups} duplicate groups (= multi-well)")
            # Log per-fold row counts (= sanity check on balance)
            _fold_size = pd.Series(fold_id).value_counts().sort_index().to_dict()
            log.info(f"  Edge Q per-fold rows: {_fold_size}")
            edge_q_ok = True
        else:
            section(log, "exp007: Edge Q disabled → naive GroupKFold by well")
            cv = GroupKFold(n_splits=N_SPLITS)
            wells_arr = train_df_kb["well"].to_numpy()
            fold_id = np.full(len(train_df_kb), -1, dtype=np.int8)
            for k, (_, va_idx) in enumerate(
                cv.split(train_df_kb, y_kb, wells_arr)
            ):
                fold_id[va_idx] = k
            groups = wells_arr.astype(object)

        # ── Step D: karnakbaev OOF refit under Edge Q fold ────────────────
        # = predict train_df_kb on val fold using the pretrained karnakbaev
        #   models (lgb0/lgb1/lgb2/xgb/cb), so karnakbaev OOF and own OOF
        #   share the same fold partition (= Ridge weight constellation
        #   consistency, exp006 postmortem 対策).
        section(log, f"{pipeline_tag}: karnakbaev OOF refit under Edge Q fold")
        kb_oof_q = {k: np.zeros(len(train_df_kb), dtype=np.float32)
                    for k in ACTIVE_MODELS}
        with timer(log, "kb OOF refit (5 fold predict)"):
            for k in ACTIVE_MODELS:
                preds_full = base_models[k].predict(
                    train_df_kb[feature_cols_kb].values
                ) if k != "xgb" else base_models[k].predict(
                    train_df_kb[feature_cols_kb]
                )
                # Note: this is a TRAIN-set predict using a model FIT on the
                # SAME train data → not a real OOF. We use it only as a proxy
                # to align fold partitions; karnakbaev's true OOF (oof_predictions.parquet)
                # is preferred when fold partitions match. Here we keep the train-set
                # predict per fold as a placeholder for stack consistency.
                kb_oof_q[k] = preds_full.astype(np.float32)
            log.info(f"  kb OOF refit done (note: train-set predict, not true OOF)")
        log.info(
            "  Stack consistency note: kb_oof_q is karnakbaev train-set predict, "
            "used only to align fold partitions with own OOF for Ridge meta. "
            "True OOF requires re-fitting karnakbaev models per Edge Q fold "
            "(skipped for runtime; deferred to v2)."
        )

        # ── Step E: 自前 base train under Edge Q fold ──────────────────────
        # Phase 5 v3: heteroscedastic sample_weight + Multi-seed MEDIAN
        if OWN_BASE_ENABLE and not DEBUG_SKIP_OWN_TRAIN:
            section(log, f"{pipeline_tag}: 自前 base train (Edge Q fold) v3 "
                         f"(MEDIAN={OWN_USE_MEDIAN_SEEDS}, hetero_w={HETERO_W_ENABLE})")
            own_t0 = time.perf_counter()

            # ── Step E.1: build heteroscedastic sample_weight (per-well sigma) ──
            w_train_full = np.ones(len(train_df_kb), dtype=np.float32)
            if HETERO_W_ENABLE:
                try:
                    stats_path = None
                    for cand in (HETERO_W_STATS_PATH, HETERO_W_STATS_LOCAL):
                        if cand and Path(cand).exists():
                            stats_path = cand; break
                    if stats_path is None:
                        raise FileNotFoundError(
                            f"per-well-stats not found; falling back to ones")
                    stats_df = pd.read_parquet(stats_path)
                    if "well_id" not in stats_df.columns or "b_well_resid_std" not in stats_df.columns:
                        raise RuntimeError(
                            f"per-well-stats schema unexpected: cols={stats_df.columns.tolist()[:6]}")
                    sigma_map = dict(zip(
                        stats_df["well_id"].astype(str),
                        stats_df["b_well_resid_std"].astype(np.float32),
                    ))
                    sigma_arr = np.array([
                        sigma_map.get(str(w), np.nan) for w in train_df_kb["well"].values
                    ], dtype=np.float32)
                    n_nan = int(np.isnan(sigma_arr).sum())
                    if n_nan > 0:
                        log.warning(
                            f"  hetero_w: {n_nan} wells missing in per-well-stats; "
                            f"filling with p50={HETERO_W_FILL_P50}")
                        sigma_arr = np.where(np.isnan(sigma_arr),
                                              HETERO_W_FILL_P50, sigma_arr)
                    w_train_full = 1.0 / np.maximum(sigma_arr, 1e-6)
                    w_train_full = w_train_full / float(w_train_full.mean())
                    w_train_full = np.clip(
                        w_train_full,
                        HETERO_W_CLIP_LO, HETERO_W_CLIP_HI,
                    ).astype(np.float32)
                    log.info(
                        f"  hetero_w: shape={w_train_full.shape} "
                        f"min={w_train_full.min():.3f} max={w_train_full.max():.3f} "
                        f"mean={w_train_full.mean():.3f} "
                        f"(stats from {stats_path})")
                except Exception as _hw_e:
                    log.warning(f"  hetero_w build failed ({_hw_e}); using ones")
                    w_train_full = np.ones(len(train_df_kb), dtype=np.float32)
            else:
                log.info("  HETERO_W_ENABLE=False → using uniform weights")

            # ── Step E.1b: exp010 Step 2 = adversarial validation reweight ──
            # train vs test の distribution shift を LightGBM binary classify、
            # AUC ≥ threshold なら test と離れた train wells を sample_weight
            # 0.5 で reduce。leak guard: visible-only per-well features のみ。
            adv_auc = None
            adv_n_dropped = 0
            if ADVERSARIAL_DROP_ENABLE:
                section(log, "exp010 Step 2: adversarial validation reweight")
                try:
                    train_well_ids = list(train_df_kb["well"].unique())
                    # Test wells = whatever build_dataset(TEST_DIR) sees. Use
                    # test_df["well"].unique() to ensure exact match.
                    test_well_ids = list(test_df["well"].unique())
                    log.info(f"  adv: {len(train_well_ids)} train wells vs "
                             f"{len(test_well_ids)} test wells")
                    with timer(log, "adv: build per-well features (train)"):
                        feat_tr = compute_adversarial_well_features(
                            TRAIN_DIR, train_well_ids,
                            ADVERSARIAL_FEATURE_COLS,
                        )
                    with timer(log, "adv: build per-well features (test)"):
                        feat_te = compute_adversarial_well_features(
                            TEST_DIR, test_well_ids,
                            ADVERSARIAL_FEATURE_COLS,
                        )
                    log.info(f"  adv features: train={feat_tr.shape} "
                             f"test={feat_te.shape}")
                    # Leak guard assertion: features should NOT contain any
                    # target / TVT-derived col by construction
                    _forbidden = ("target", "TVT", "TVT_input", "tvt_range")
                    for _f in _forbidden:
                        assert _f not in ADVERSARIAL_FEATURE_COLS, (
                            f"adv leak: {_f} must not be in features")
                    with timer(log, "adv: fit binary LGB"):
                        adv_auc, train_test_likelihood = fit_adversarial_classifier(
                            feat_tr, feat_te,
                            ADVERSARIAL_FEATURE_COLS,
                            ADVERSARIAL_LGB_PARAMS,
                            ADVERSARIAL_LGB_N_EST,
                            ADVERSARIAL_LGB_EARLY_STOP,
                            seed=SEED,
                        )
                    log.info(f"  adv AUC (5-fold OOF) = {adv_auc:.4f}")
                    if adv_auc >= ADVERSARIAL_AUC_THRESHOLD:
                        well_to_likelihood = dict(zip(
                            feat_tr.index.astype(str), train_test_likelihood))
                        train_well_arr = train_df_kb["well"].astype(str).to_numpy()
                        w_train_full, adv_n_dropped = apply_adversarial_reweight(
                            w_train_full,
                            train_well_arr,
                            well_to_likelihood,
                            drop_quantile=ADVERSARIAL_DROP_QUANTILE,
                            drop_weight=ADVERSARIAL_DROP_WEIGHT,
                        )
                        log.info(
                            f"  adv: AUC={adv_auc:.4f} ≥ {ADVERSARIAL_AUC_THRESHOLD} "
                            f"→ reweighted {adv_n_dropped} wells "
                            f"(bottom {ADVERSARIAL_DROP_QUANTILE*100:.0f}% by "
                            f"test-likelihood) × {ADVERSARIAL_DROP_WEIGHT}")
                        log.info(
                            f"  w_train_full after adv: "
                            f"min={w_train_full.min():.3f} "
                            f"max={w_train_full.max():.3f} "
                            f"mean={w_train_full.mean():.3f}")
                    else:
                        log.info(
                            f"  adv: AUC={adv_auc:.4f} < {ADVERSARIAL_AUC_THRESHOLD} "
                            f"(= shift mild) → no reweight applied")
                except Exception as _adv_e:
                    log.warning(
                        f"  adv reweight failed ({type(_adv_e).__name__}: {_adv_e}); "
                        f"continuing with hetero_w only")
            else:
                log.info("  ADVERSARIAL_DROP_ENABLE=False → skipping adv reweight")

            # Make sure all extra cols exist in test_df too
            for c in feature_cols_own:
                if c not in test_df.columns:
                    log.warning(f"  test_df missing own feature {c} — filling 0")
                    test_df[c] = np.float32(0.0)
                if c not in train_df_kb.columns:
                    log.warning(f"  train_df_kb missing own feature {c} — filling 0")
                    train_df_kb[c] = np.float32(0.0)
            X_kb_full = train_df_kb[feature_cols_own]
            X_test_full = test_df[feature_cols_own]
            n_train = len(train_df_kb)
            n_test  = len(test_df)

            if OWN_USE_MEDIAN_SEEDS:
                own_keys = ["lgb_own_med", "cb_own_med"]
                lgb_schedule = [
                    (OWN_LGB_LR_MED, OWN_LGB_N_EST_MED, sd)
                    for sd in OWN_LGB_SEEDS_MED
                ]
                cb_seeds = list(OWN_CB_SEEDS_MED)
                cb_n_iter = OWN_CB_N_EST_MED
            else:
                own_keys = ["lgb_own0", "lgb_own1", "lgb_own2", "cb_own"]
                lgb_schedule = list(zip(
                    OWN_LGB_LRS, OWN_LGB_N_ESTS, OWN_LGB_SEEDS))
                cb_seeds = [OWN_CB_SEED]
                cb_n_iter = OWN_CB_N_EST

            own_oof  = {k: np.zeros(n_train, dtype=np.float32) for k in own_keys}
            own_test = {k: np.zeros(n_test, dtype=np.float32) for k in own_keys}

            try:
                for fold in range(N_SPLITS):
                    elapsed = time.perf_counter() - own_t0
                    if elapsed > OWN_TRAIN_HARD_TIMEOUT_S:
                        log.warning(
                            f"  fold {fold}: own train hard timeout "
                            f"({elapsed:.0f}s > {OWN_TRAIN_HARD_TIMEOUT_S}s) — break")
                        break
                    tr_idx = np.where(fold_id != fold)[0]
                    va_idx = np.where(fold_id == fold)[0]
                    log.info(f"  fold {fold}: tr={len(tr_idx):,} va={len(va_idx):,}")
                    Xt  = X_kb_full.iloc[tr_idx]; yt = y_kb[tr_idx]
                    Xv  = X_kb_full.iloc[va_idx]; yv = y_kb[va_idx]
                    w_tr = w_train_full[tr_idx]; w_va = w_train_full[va_idx]

                    # ── LGB seed loop ─────────────────────────────────────
                    lgb_val_stack = []
                    lgb_test_stack = []
                    for j, (lr, ne, sd) in enumerate(lgb_schedule):
                        key_legacy = f"lgb_own{j}"
                        params = dict(LGB_BASE)
                        params.update(dict(
                            learning_rate=lr,
                            num_leaves=OWN_LGB_NUM_LEAVES,
                            n_estimators=ne,
                            random_state=sd,
                            seed=sd,
                        ))
                        model = lgb.LGBMRegressor(**params)
                        try:
                            model.fit(
                                Xt.values, yt,
                                sample_weight=w_tr,
                                eval_set=[(Xv.values, yv)],
                                eval_sample_weight=[w_va],
                                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS),
                                           lgb.log_evaluation(0)],
                            )
                        except Exception as _lgb_e:
                            log.warning(f"  fold{fold} lgb seed={sd}: GPU LGB failed ({_lgb_e}), retry CPU")
                            params["device_type"] = "cpu"
                            model = lgb.LGBMRegressor(**params)
                            model.fit(
                                Xt.values, yt,
                                sample_weight=w_tr,
                                eval_set=[(Xv.values, yv)],
                                eval_sample_weight=[w_va],
                                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS),
                                           lgb.log_evaluation(0)],
                            )
                        pred_v = model.predict(Xv.values).astype(np.float32)
                        pred_t = model.predict(X_test_full.values).astype(np.float32)
                        if OWN_USE_MEDIAN_SEEDS:
                            lgb_val_stack.append(pred_v)
                            lgb_test_stack.append(pred_t)
                        else:
                            own_oof[key_legacy][va_idx]  = pred_v
                            own_test[key_legacy]        += pred_t / N_SPLITS
                        log.info(
                            f"    fold{fold} lgb seed={sd} lr={lr} "
                            f"ne_used={getattr(model, 'best_iteration_', ne)}: "
                            f"va RMSE={root_mean_squared_error(yv, pred_v):.4f}")
                        del model
                        gc.collect()

                    if OWN_USE_MEDIAN_SEEDS and lgb_val_stack:
                        med_v = np.median(np.stack(lgb_val_stack, axis=0), axis=0).astype(np.float32)
                        med_t = np.median(np.stack(lgb_test_stack, axis=0), axis=0).astype(np.float32)
                        own_oof["lgb_own_med"][va_idx]  = med_v
                        own_test["lgb_own_med"]        += med_t / N_SPLITS
                        log.info(
                            f"    fold{fold} lgb_own_med (MEDIAN over {len(lgb_val_stack)} seed): "
                            f"va RMSE={root_mean_squared_error(yv, med_v):.4f}")

                    # ── CatBoost seed loop ────────────────────────────────
                    cb_val_stack = []
                    cb_test_stack = []
                    for sd in cb_seeds:
                        cb_params = dict(CB_PARAMS)
                        cb_params["learning_rate"] = OWN_CB_LR
                        cb_params["depth"]         = OWN_CB_DEPTH
                        cb_params["iterations"]    = cb_n_iter
                        cb_params["random_seed"]   = sd
                        try:
                            cb_model = CatBoostRegressor(**cb_params)
                            cb_model.fit(Pool(Xt.values, yt, weight=w_tr),
                                         eval_set=Pool(Xv.values, yv, weight=w_va),
                                         use_best_model=True, verbose=0)
                        except Exception as _cb_e:
                            log.warning(f"  fold{fold} cb seed={sd}: GPU CB failed ({_cb_e}), retry CPU")
                            cb_params["task_type"] = "CPU"
                            cb_params.pop("devices", None)
                            cb_model = CatBoostRegressor(**cb_params)
                            cb_model.fit(Pool(Xt.values, yt, weight=w_tr),
                                         eval_set=Pool(Xv.values, yv, weight=w_va),
                                         use_best_model=True, verbose=0)
                        pred_v = cb_model.predict(Xv.values).astype(np.float32)
                        pred_t = cb_model.predict(X_test_full.values).astype(np.float32)
                        if OWN_USE_MEDIAN_SEEDS:
                            cb_val_stack.append(pred_v)
                            cb_test_stack.append(pred_t)
                        else:
                            own_oof["cb_own"][va_idx]  = pred_v
                            own_test["cb_own"]        += pred_t / N_SPLITS
                        log.info(
                            f"    fold{fold} cb seed={sd}: "
                            f"va RMSE={root_mean_squared_error(yv, pred_v):.4f}")
                        del cb_model
                        gc.collect()

                    if OWN_USE_MEDIAN_SEEDS and cb_val_stack:
                        med_v = np.median(np.stack(cb_val_stack, axis=0), axis=0).astype(np.float32)
                        med_t = np.median(np.stack(cb_test_stack, axis=0), axis=0).astype(np.float32)
                        own_oof["cb_own_med"][va_idx]  = med_v
                        own_test["cb_own_med"]        += med_t / N_SPLITS
                        log.info(
                            f"    fold{fold} cb_own_med (MEDIAN over {len(cb_val_stack)} seed): "
                            f"va RMSE={root_mean_squared_error(yv, med_v):.4f}")

                for k in own_keys:
                    log.info(f"  own OOF {k}: RMSE={root_mean_squared_error(y_kb, own_oof[k]):.4f}")
                own_ok = True
            except Exception as e:
                log.error(f"Own train phase failed: {type(e).__name__}: {e}")
                own_ok = False
        else:
            log.info("  OWN_BASE_ENABLE=False or DEBUG_SKIP_OWN_TRAIN=True → 自前 train skipped")

    except Exception as e:
        log.error(f"{pipeline_tag} train_df phase failed: {type(e).__name__}: {e}")
        log.warning("Falling back to karnakbaev 5-base NM blend (= exp005 同等)")

    # ── Step F: meta blend — path b (Phase 5 v3) or legacy n-base Ridge ────
    used_path = "fallback"
    if own_ok and y_kb is not None:
        try:
            oof_pred_path = ARTEFACT_DIR / "oof_predictions.parquet"
            oof_df_kb = pd.read_parquet(oof_pred_path)
            merge_keys = ["id"] if "id" in oof_df_kb.columns else None
            if merge_keys is None:
                raise RuntimeError("oof_predictions has no 'id' column")
            _tmp = train_df_kb[merge_keys].merge(
                oof_df_kb, on=merge_keys, how="left", suffixes=("", "_oof"))
            kb_oof_for_stack = {
                k: _tmp[f"oof_{k}"].to_numpy(np.float32)
                for k in ACTIVE_MODELS
            }
            missing_kb = [k for k in ACTIVE_MODELS
                          if np.isnan(kb_oof_for_stack[k]).any()]
            if missing_kb:
                raise RuntimeError(f"kb OOF NaN after merge: {missing_kb}")
            log.info("  using karnakbaev published OOF for kb side")
        except Exception as _kb_e:
            log.warning(f"  kb OOF load failed ({_kb_e}) → using kb_oof_q (train-predict proxy)")
            kb_oof_for_stack = kb_oof_q

        own_active_keys = list(own_oof.keys())

        if STACK_PATH_B_ENABLE:
            section(log, f"{pipeline_tag}: path b blend = kb simple-avg + own Ridge + 1D")
            try:
                kb_avg_oof = np.mean(
                    np.column_stack([kb_oof_for_stack[k] for k in ACTIVE_MODELS]),
                    axis=1,
                ).astype(np.float64)
                kb_avg_test = np.mean(
                    np.column_stack([test_preds_kb[k] for k in ACTIVE_MODELS]),
                    axis=1,
                ).astype(np.float64)

                Sx_own = np.column_stack([own_oof[k] for k in own_active_keys])
                St_own = np.column_stack([own_test[k] for k in own_active_keys])
                ridge_own = Ridge(alpha=1.0, fit_intercept=False, positive=True)
                ridge_own.fit(Sx_own, y_kb)
                own_oof_blend = ridge_own.predict(Sx_own).astype(np.float64)
                own_test_blend = ridge_own.predict(St_own).astype(np.float64)
                own_coef = ridge_own.coef_
                own_coef_norm = own_coef / max(own_coef.sum(), 1e-9)
                log.info(f"  own Ridge weights: "
                         f"{dict(zip(own_active_keys, np.round(own_coef_norm, 4)))}")

                step = STACK_PATH_B_GRID_STEP
                grid = np.arange(0.0, 1.0 + step / 2, step)
                best_w, best_r = 0.5, np.inf
                for w in grid:
                    blend = w * kb_avg_oof + (1.0 - w) * own_oof_blend
                    r = root_mean_squared_error(y_kb, blend)
                    if r < best_r:
                        best_r, best_w = float(r), float(w)
                r_kb_only  = root_mean_squared_error(y_kb, kb_avg_oof)
                r_own_only = root_mean_squared_error(y_kb, own_oof_blend)
                log.info(
                    f"  path b grid search: w_kb*={best_w:.2f} blend OOF RMSE={best_r:.4f} "
                    f"(kb_only={r_kb_only:.4f}, own_only={r_own_only:.4f})")
                if best_w in (0.0, 1.0):
                    log.warning(
                        f"  path b: w_kb* boundary={best_w} → one side dominates")

                test_delta = (
                    best_w * kb_avg_test + (1.0 - best_w) * own_test_blend
                ).astype(np.float64)
                used_path = f"path_b_w{best_w:.2f}"
                log.info(f"  → using path b blend (path tag={used_path})")
            except Exception as e:
                log.error(f"path b blend failed: {type(e).__name__}: {e}")
                log.warning("Falling back to legacy n-base Ridge")
                used_path = "fallback_path_b"

        if used_path in ("fallback", "fallback_path_b") and not STACK_PATH_B_ENABLE:
            section(log, f"{pipeline_tag}: legacy n-base Ridge meta fit + apply")
            try:
                stack_keys = ACTIVE_MODELS + own_active_keys
                Sx = np.column_stack(
                    [kb_oof_for_stack[k] for k in ACTIVE_MODELS] +
                    [own_oof[k] for k in own_active_keys]
                )
                St = np.column_stack(
                    [test_preds_kb[k] for k in ACTIVE_MODELS] +
                    [own_test[k] for k in own_active_keys]
                )
                ridge_ex = Ridge(alpha=1.0, fit_intercept=False, positive=True)
                ridge_ex.fit(Sx, y_kb)
                oof_blend = ridge_ex.predict(Sx)
                r_avg = root_mean_squared_error(y_kb, Sx.mean(axis=1))
                r_stk = root_mean_squared_error(y_kb, oof_blend)
                wts   = ridge_ex.coef_ / max(ridge_ex.coef_.sum(), 1e-9)
                log.info(f"  Simple n-avg OOF RMSE: {r_avg:.4f}")
                log.info(f"  Ridge n-stk OOF RMSE: {r_stk:.4f}")
                log.info(f"  Ridge weights        : "
                         f"{dict(zip(stack_keys, np.round(wts, 4)))}")
                own_total_w = float(sum(wts[len(ACTIVE_MODELS):]))
                log.info(f"  own base total weight: {own_total_w:.4f}")
                if own_total_w > STACK_OWN_WEIGHT_DEGRADE_LIMIT:
                    log.warning(
                        f"  own total weight {own_total_w:.3f} > limit "
                        f"{STACK_OWN_WEIGHT_DEGRADE_LIMIT} → degenerate, "
                        f"falling back to NM 5-blend"
                    )
                    raise RuntimeError("own_weight_degenerate")
                if r_stk < r_avg:
                    test_delta = ridge_ex.predict(St).astype(np.float64)
                    used_path = "ridge_legacy"
                    log.info("  → using legacy n-base Ridge re-fit blend")
                else:
                    test_delta = St.mean(axis=1).astype(np.float64)
                    used_path = "mean_legacy"
                    log.info("  → Ridge stk worse than avg, falling back to n-mean")
            except Exception as e:
                log.error(f"legacy Ridge meta failed: {type(e).__name__}: {e}")
                log.warning("Falling back to NM 5-blend")
                used_path = "fallback"

    if used_path == "fallback":
        section(log, f"{pipeline_tag} fallback: karnakbaev 5-base NM blend")
        nm_w = am.load_json(am.ENSEMBLE_WEIGHTS)
        test_delta = apply_ensemble_nm(test_preds_kb, nm_w)
        log.info(f"  NM blend test_delta range=[{test_delta.min():.3f},"
                 f"{test_delta.max():.3f}]")

    # ── Step F.5: Edge R — test-time online learning ───────────────────────
    # 出典: discussion 698002 (online 10.953 vs no-online 11.323 = -0.370 ft)
    # karnakbaev LGB 3 base のみに continued training を適用。
    # online predict と既存 test_delta (= path b blend) を w_R で重ね合わせ。
    # leak guard: visible TVT_input のみ参照、 hidden TVT に絶対触れない。
    if EDGE_R_ENABLE:
        section(log, f"{pipeline_tag}: Edge R test-time online learning")
        _edge_r_t0 = time.time()
        try:
            with timer(log, "Edge R visible dataset build"):
                visible_df = build_visible_dataset(
                    TEST_DIR,
                    max_wells=DEBUG_MAX_WELLS,
                    tail_k=EDGE_R_VISIBLE_TAIL_K,
                    min_visible_rows=EDGE_R_MIN_VISIBLE_ROWS,
                    n_jobs=NCPU,
                )

            edge_r_applied = False
            if visible_df is not None and len(visible_df) > 0:
                missing = [c for c in feature_cols_kb if c not in visible_df.columns]
                if missing:
                    log.warning(
                        f"[Edge R] visible_df missing {len(missing)} kb cols, "
                        f"e.g. {missing[:3]} — filling 0"
                    )
                    for c in missing:
                        visible_df[c] = np.float32(0.0)

                X_R = visible_df[feature_cols_kb].to_numpy(np.float32)
                y_R = visible_df["target"].to_numpy(np.float32)
                # NaN/inf guard
                finite = np.isfinite(y_R)
                if not finite.all():
                    n_bad = int((~finite).sum())
                    log.warning(f"[Edge R] dropping {n_bad} non-finite y_R rows")
                    X_R, y_R = X_R[finite], y_R[finite]
                log.info(
                    f"[Edge R] X_R shape={X_R.shape}  "
                    f"y_R range=[{y_R.min():.3f}, {y_R.max():.3f}]"
                )

                if len(y_R) >= 100:
                    X_test_full_kb = test_df[feature_cols_kb].to_numpy(np.float32)
                    online_preds = {}
                    for seed_idx, seed in enumerate(LGB_SEEDS):
                        if (time.time() - _edge_r_t0) > EDGE_R_HARD_TIMEOUT_S:
                            log.warning(
                                f"[Edge R] timeout exceeded "
                                f"({EDGE_R_HARD_TIMEOUT_S}s) — skipping remaining bases"
                            )
                            break
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
                            p_online = online_booster.predict(X_test_full_kb).astype(np.float64)
                            online_preds[key] = p_online
                            log.info(
                                f"[Edge R]   {key} online predict range="
                                f"[{p_online.min():.3f}, {p_online.max():.3f}]"
                            )
                        except Exception as exc:
                            log.warning(
                                f"[Edge R]   {key} continued training failed "
                                f"({type(exc).__name__}: {exc})"
                            )

                    if online_preds:
                        # online NM blend: online LGB を test_preds_kb の lgb スロットに差替え
                        online_full = {}
                        for k in ACTIVE_MODELS:
                            if k in online_preds:
                                online_full[k] = online_preds[k]
                            elif k in test_preds_kb:
                                online_full[k] = test_preds_kb[k]
                        try:
                            nm_w_local = am.load_json(am.ENSEMBLE_WEIGHTS)
                            test_delta_online = apply_ensemble_nm(online_full, nm_w_local)
                        except Exception as _nm_e:
                            log.warning(
                                f"[Edge R] NM weights load failed ({_nm_e}); "
                                f"falling back to simple mean"
                            )
                            stk = np.column_stack([online_full[k] for k in ACTIVE_MODELS])
                            test_delta_online = stk.mean(axis=1).astype(np.float64)
                        log.info(
                            f"[Edge R] online blend range="
                            f"[{test_delta_online.min():.3f}, {test_delta_online.max():.3f}]"
                        )
                        log.info(
                            f"[Edge R] base   blend range="
                            f"[{test_delta.min():.3f}, {test_delta.max():.3f}]"
                        )

                        w = float(EDGE_R_BLEND_W)
                        test_delta_blended = w * test_delta_online + (1.0 - w) * test_delta
                        diff = test_delta_blended - test_delta
                        log.info(
                            f"[Edge R] blended w={w} (online) + {1-w} (base)  "
                            f"diff abs mean={np.abs(diff).mean():.4f}"
                        )
                        test_delta = test_delta_blended
                        edge_r_applied = True
                        used_path = used_path + "+edge_r"
                    else:
                        log.warning("[Edge R] no online predictions produced — keeping base")
                else:
                    log.warning(
                        f"[Edge R] too few visible-as-pseudo-hidden rows "
                        f"({len(y_R)}) — disabling"
                    )

            if not edge_r_applied:
                log.warning("[Edge R] not applied; using base test_delta")
        except Exception as _er_e:
            log.error(
                f"[Edge R] pipeline failed ({type(_er_e).__name__}: {_er_e}); "
                f"falling back to base test_delta"
            )
        log.info(f"[Edge R] total elapsed: {time.time() - _edge_r_t0:.1f}s")

    # ── Step G: post-processing ────────────────────────────────────────────
    section(log, f"{pipeline_tag}: post-processing + SG smooth (path={used_path})")
    test_delta_pp = apply_postproc(test_df, test_delta, best_alpha, best_tau)
    test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)
    if y_kb is not None:
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
              f"Choose from: train | infer | infer_tabicl | infer_edge_qm | "
              f"infer_edge_d_kalman | cv | features_only | ensemble_only")


section(log, "DONE")


# In[ ]:




