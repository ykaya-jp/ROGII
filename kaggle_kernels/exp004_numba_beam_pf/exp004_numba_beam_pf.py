"""ROGII exp004 — Numba Beam + PF + LGB x3 + CB stack (Approach A).

Self-contained Kaggle kernel for ROGII Wellbore Geology Prediction.

Inspired by romantamrazov/rogii-super-solution-lb-top-3 (Apache 2.0):
  https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3
  License: Apache 2.0 (default Kaggle kernel license)

Self-reimplemented for ROGII-exp004 with:
  - Numba JIT Beam Search 7 configs (delta +/-2)
  - Particle Filter (ANCC + Z, N=500, deterministic via seeded rng)
  - Spatial imputers (FormationPlaneKNN + DenseANCCImputer) with leak-safe
    self_wid exclusion
  - AG deep-EDA finding D1 = per-well b_med cluster id (66 train clusters,
    inline hard-coded map)
  - LGB x3 diverse configs + CatBoost stack, Ridge meta blend, post-grid
    alpha/tau/w_pf
  - 8 hr hard timeout, deterministic seeds
"""

# ruff: noqa: E402

import os
import subprocess
import sys
import time

# Numba install + cache dir setup (Kaggle base image may lack numba 0.60+).
for _pkg in ["numba"]:
    if subprocess.run([sys.executable, "-m", "pip", "show", _pkg], capture_output=True).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", _pkg, "--quiet"])
os.environ.setdefault("NUMBA_CACHE_DIR", "/kaggle/working/.numba")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import gc
import multiprocessing
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from joblib import Parallel, delayed
from numba import njit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
NCPU = min(4, multiprocessing.cpu_count())
T_START = time.perf_counter()
HARD_TIMEOUT_S = 8 * 3600  # 8 hr hard budget (Kaggle limit = 9 hr)


def _elapsed() -> float:
    return time.perf_counter() - T_START


def _time_left() -> float:
    return HARD_TIMEOUT_S - _elapsed()


def _find_data():
    for p in [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]:
        if (p / "train").exists():
            return p
    for p in Path("/kaggle/input").glob("*/sample_submission.csv"):
        return p.parent
    # Local fallback for offline dry-run.
    local = Path("/home/yusuke_kaya/projects/kaggle/ROGII/data/raw")
    if (local / "train").exists():
        return local
    raise FileNotFoundError("Data not found in /kaggle/input or local fallback")


DATA = _find_data()
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"
OUT = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20
N_SPLITS = 5

# 7 beam configs (delta +/-2, diversity preservation).
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

# PF defaults (N=500).
PF_N = 500
ANCC_N = 500

# ============================================================================
# D1 cluster map (= AG deep-EDA per-well b_med cluster, 66 distinct values,
# 766 train wells; test 3 wells: 000d7d20/00e12e8b -> 37, 00bbac68 -> 52).
# Inlined to keep kernel self-contained (no external dataset upload required).
# ============================================================================
D1_CLUSTER_MAP_PLACEHOLDER = True  # populated below
D1_CLUSTER_MAP = {
    "000d7d20": 37,
    "00bbac68": 52,
    "00e12e8b": 37,
    "015fe0d2": 52,
    "01869cd4": 37,
    "01982c1d": 8,
    "028d7b28": 57,
    "02e7fe5a": 50,
    "0390d174": 6,
    "044af7d1": 41,
    "0498acab": 33,
    "052d64df": 33,
    "05948241": 7,
    "059c8f24": 1,
    "05a0ee4d": 41,
    "060ab2b8": 12,
    "06df5958": 29,
    "071d7b45": 2,
    "0849b4e2": 44,
    "084d66f9": 16,
    "08fecf00": 49,
    "09053135": 35,
    "09441b8d": 26,
    "0955cd6c": 48,
    "09ec2ca9": 15,
    "0a57a29c": 26,
    "0bbf5e67": 10,
    "0be01c24": 31,
    "0c1b160c": 20,
    "0d99f8e4": 26,
    "0dc5e64d": 15,
    "0dc835f3": 8,
    "0dd99dc5": 57,
    "0e5e560d": 15,
    "109d9d70": 48,
    "10a1281a": 26,
    "10b89021": 50,
    "10be2420": 7,
    "113011eb": 57,
    "1131525f": 29,
    "11d0f5ac": 12,
    "12203f2a": 44,
    "122bd617": 7,
    "125be846": 33,
    "1285a37b": 60,
    "1295a25b": 37,
    "12f0cd90": 31,
    "137d1e44": 60,
    "13ce113d": 26,
    "13f598d9": 57,
    "14a53cb3": 24,
    "14ab73fb": 29,
    "14ad5efb": 31,
    "14fee784": 8,
    "153552f5": 52,
    "1550491d": 31,
    "155887bb": 52,
    "1590af81": 35,
    "15e154a1": 17,
    "1619e2ca": 34,
    "16e4a047": 43,
    "173ed841": 60,
    "1770d7c6": 57,
    "177622d7": 52,
    "18d24f7b": 26,
    "19137a89": 15,
    "1944513c": 31,
    "197f8a5a": 14,
    "19871e7f": 57,
    "198a5607": 20,
    "19fb4f7b": 60,
    "1a518997": 26,
    "1a68190b": 63,
    "1a730ac2": 33,
    "1aaf1da0": 40,
    "1b08ed48": 47,
    "1b1c5372": 6,
    "1b6ba517": 46,
    "1b82b665": 6,
    "1bcb2fcf": 55,
    "1beca81d": 32,
    "1d3fbf02": 37,
    "1d78281a": 63,
    "1e30dedf": 6,
    "1e5e3573": 48,
    "1f9afacb": 43,
    "1fdbba44": 63,
    "1fee2b62": 57,
    "20096916": 41,
    "200d233d": 43,
    "201661b5": 49,
    "203721ab": 26,
    "20402d71": 14,
    "2043abc5": 33,
    "204cc64b": 60,
    "206b6193": 33,
    "2075c696": 26,
    "22c5f93f": 41,
    "230eaaa3": 33,
    "2364716c": 17,
    "23b9beb0": 35,
    "24d8997e": 25,
    "25050f63": 13,
    "25939962": 52,
    "25fd32b3": 24,
    "261785ee": 35,
    "264feceb": 26,
    "26623a50": 31,
    "26d3a96a": 60,
    "272abef3": 33,
    "27300fd5": 60,
    "276b012a": 57,
    "27c3155b": 61,
    "27ebb9b9": 46,
    "283269ac": 33,
    "28473855": 17,
    "28f4eda9": 14,
    "29b24409": 26,
    "29de15c8": 44,
    "2ab29395": 52,
    "2ac7cb6b": 8,
    "2acc78d4": 41,
    "2b034622": 0,
    "2b06ad65": 44,
    "2b547943": 52,
    "2be5c96a": 41,
    "2bff06cf": 16,
    "2c0c4a4e": 17,
    "2cee0cba": 23,
    "2d0c268a": 31,
    "2d179ded": 3,
    "2d196986": 24,
    "2d2d0c6b": 24,
    "2d35f86d": 41,
    "2d9f6cb9": 57,
    "2ddad940": 37,
    "2e29e833": 48,
    "2e43c2ea": 25,
    "2e63d9de": 55,
    "2ee0235b": 26,
    "2f19d536": 52,
    "2f8e53c3": 43,
    "2fa01aa6": 59,
    "2fd68f7b": 6,
    "2fe023be": 35,
    "30d75030": 8,
    "310c71a5": 26,
    "32fe84c9": 47,
    "33c40295": 35,
    "3407f5bf": 16,
    "3417285d": 50,
    "347242c8": 44,
    "353e5502": 18,
    "357c12f9": 5,
    "3584a6eb": 52,
    "35b30c7f": 17,
    "35b3ef6a": 5,
    "367456ce": 8,
    "368131f9": 41,
    "368fe682": 29,
    "3690a146": 26,
    "37344c2a": 16,
    "373ef48b": 26,
    "374be387": 3,
    "37d36812": 52,
    "38991fd4": 21,
    "389ae58f": 37,
    "3932faa6": 46,
    "398dce4b": 7,
    "39b51819": 16,
    "3a0af893": 48,
    "3a223519": 63,
    "3a7dd95d": 38,
    "3a824aa7": 52,
    "3a86fe8d": 33,
    "3aaa2b30": 8,
    "3ac16ad0": 14,
    "3b21ea64": 57,
    "3bbe1f5d": 26,
    "3bfee975": 52,
    "3c3dbcbc": 41,
    "3d96d897": 26,
    "3df4b19e": 31,
    "3e011332": 35,
    "3e678ea7": 55,
    "3ef6372a": 33,
    "3ef98781": 44,
    "3f603129": 15,
    "404c4384": 7,
    "4121c517": 15,
    "41fb0192": 57,
    "42669188": 37,
    "42c538a1": 29,
    "43b06c70": 41,
    "43c7f85a": 14,
    "43e16325": 16,
    "44441e54": 43,
    "445af6c2": 26,
    "4463446c": 2,
    "44a94013": 12,
    "45fb3e21": 35,
    "460f859e": 24,
    "4630ab92": 43,
    "463c38f2": 21,
    "4654e7de": 26,
    "466fc788": 43,
    "46a301e9": 20,
    "46a7190b": 55,
    "46d24a09": 63,
    "46d41ff6": 47,
    "46dfcfca": 55,
    "47222616": 52,
    "47ee27ab": 33,
    "481de5fc": 46,
    "48c1f673": 6,
    "4936af16": 8,
    "493b5b31": 61,
    "49dbc14c": 24,
    "4a035ec2": 57,
    "4a335117": 25,
    "4a8ecc0b": 41,
    "4b462989": 16,
    "4b831d20": 56,
    "4c3168cf": 44,
    "4c3df468": 14,
    "4caa7289": 26,
    "4cd1ac61": 41,
    "4cecf7b3": 43,
    "4ddde4c1": 15,
    "4e050c92": 43,
    "4ed93db6": 31,
    "4f3eb9e9": 31,
    "4f4ac5ce": 27,
    "4f4afcc6": 26,
    "4f69febf": 14,
    "4fbe9c06": 26,
    "504c8b08": 24,
    "511e1db0": 21,
    "512943b2": 31,
    "5138a660": 43,
    "516088b7": 33,
    "521a7819": 26,
    "5254b6db": 46,
    "529e88ca": 15,
    "52f1e77a": 44,
    "5305524b": 12,
    "53134831": 49,
    "5397ceb1": 35,
    "53f23031": 35,
    "543198e8": 16,
    "54753541": 5,
    "54a7e3c2": 20,
    "5542c301": 16,
    "5567d551": 33,
    "55efee7f": 21,
    "5663b2b7": 46,
    "5693dde2": 49,
    "56b00794": 16,
    "57796579": 21,
    "577eec7c": 0,
    "57f05c51": 61,
    "58850d2c": 55,
    "58fa8486": 46,
    "591cc951": 21,
    "593876b6": 31,
    "59f8b2e8": 36,
    "5a1a8fd8": 43,
    "5a5269a9": 31,
    "5a5d1982": 26,
    "5a63b5d1": 36,
    "5aa03df7": 32,
    "5aa403d0": 16,
    "5aef5c6c": 30,
    "5b7b6324": 57,
    "5bd25f59": 6,
    "5beded95": 33,
    "5cb483ac": 43,
    "5cdd8dbc": 25,
    "5d11dfb5": 53,
    "5d7198fd": 49,
    "5def1ce5": 26,
    "5eae34a8": 9,
    "5f4d2a52": 8,
    "5f6756c5": 49,
    "5f75f796": 60,
    "5fb1c15f": 16,
    "5fffa282": 33,
    "60e37807": 35,
    "61f27424": 53,
    "62569ae3": 44,
    "62a6f539": 25,
    "633774dc": 20,
    "63559d57": 49,
    "647f2a41": 41,
    "65156b96": 16,
    "653612bc": 44,
    "656a067b": 15,
    "65a5466a": 6,
    "660d9546": 44,
    "6767c111": 15,
    "67847288": 57,
    "67a8da4a": 60,
    "684a6fc1": 17,
    "698930d1": 15,
    "699ea575": 43,
    "69c1bdad": 33,
    "69e41fa6": 44,
    "6a0ff78e": 26,
    "6a8fa194": 46,
    "6ae68655": 50,
    "6d1d74e1": 14,
    "6d590e26": 31,
    "6d6d93af": 36,
    "6dbe4b60": 31,
    "6e6d122c": 52,
    "6e9ccd38": 3,
    "6ea22614": 8,
    "6ede926a": 26,
    "708caea9": 15,
    "70925e23": 60,
    "70e1788b": 3,
    "71642c7d": 63,
    "71ccf778": 46,
    "7224331b": 53,
    "722cf0d8": 14,
    "7256229b": 8,
    "726f3dc2": 33,
    "7271dd80": 17,
    "729665c0": 41,
    "729e9750": 14,
    "729f4217": 26,
    "72dd8501": 43,
    "73348914": 63,
    "739a914f": 31,
    "73b1447d": 52,
    "74b6186b": 7,
    "75c20ec7": 52,
    "75cd5f11": 26,
    "75d20a82": 46,
    "76201745": 6,
    "764072a5": 26,
    "77b0d905": 52,
    "77e4821c": 24,
    "7850c72e": 61,
    "78a4a386": 33,
    "79507629": 57,
    "796dbd4c": 8,
    "7987f2f2": 49,
    "7993a768": 50,
    "7a5660b0": 59,
    "7acda2df": 24,
    "7b2c7b3c": 35,
    "7b38844c": 43,
    "7bb17b96": 43,
    "7c607683": 26,
    "7cbbd047": 43,
    "7cd4bb31": 26,
    "7d50373e": 26,
    "7d57c75d": 15,
    "7e208414": 43,
    "7e721392": 36,
    "7ff89f8f": 3,
    "8050c789": 52,
    "807f0298": 15,
    "811f52b3": 26,
    "81fc01c6": 7,
    "8255c867": 25,
    "82aab6d2": 53,
    "82fcf37c": 14,
    "833af382": 25,
    "8382f4b6": 31,
    "84633df4": 8,
    "8478df29": 8,
    "84815d5a": 43,
    "84c3b497": 36,
    "85380836": 26,
    "8612b37f": 41,
    "861e71df": 31,
    "86454a6f": 52,
    "8648abae": 43,
    "865033f8": 25,
    "87695469": 0,
    "877ed19c": 31,
    "87aa3730": 26,
    "884ecb5f": 26,
    "8902c3f6": 5,
    "896d15b9": 42,
    "8995c945": 15,
    "89969c7a": 8,
    "89f1085d": 64,
    "89f36adf": 49,
    "89fdb2f2": 43,
    "8a3da6d1": 43,
    "8a89da23": 26,
    "8a9c8932": 29,
    "8ac2f237": 31,
    "8b12bab6": 41,
    "8b5be31b": 7,
    "8b95d6d1": 4,
    "8b9d8326": 52,
    "8bb9c1e6": 26,
    "8bda1a11": 14,
    "8bfa881d": 52,
    "8c167025": 5,
    "8c8348e8": 55,
    "8cc21f01": 37,
    "8cc800b3": 8,
    "8d5d46d7": 26,
    "8d92cb49": 35,
    "8f201368": 26,
    "90386ee4": 26,
    "91abc4c7": 12,
    "91b301ce": 16,
    "91db7070": 43,
    "925a7cd1": 52,
    "9283ae69": 16,
    "9298ad5b": 20,
    "9314ff13": 46,
    "93209a3d": 41,
    "939d9c34": 40,
    "93f5d2e6": 16,
    "940a48d9": 8,
    "940e709a": 16,
    "9426ec1e": 43,
    "94467f50": 5,
    "944f36b9": 6,
    "94d813a4": 60,
    "95b559e7": 44,
    "95c8427f": 24,
    "96936c22": 52,
    "96ae5806": 47,
    "96b7eea1": 20,
    "9719aa04": 15,
    "974802e7": 0,
    "97cd5bf9": 7,
    "98177ced": 61,
    "9896fc0b": 16,
    "98bc6f66": 14,
    "992c99e8": 31,
    "992ce078": 46,
    "99529c45": 19,
    "995ff498": 26,
    "999daf80": 25,
    "9a6b0392": 6,
    "9a8ae0d6": 43,
    "9a95e33f": 36,
    "9ab94eeb": 16,
    "9ad5410a": 41,
    "9ae248be": 63,
    "9af1ffd0": 7,
    "9b5b61c1": 20,
    "9b9e3b49": 41,
    "9bd2bba3": 57,
    "9c211c53": 0,
    "9c350a6a": 33,
    "9c8375f7": 24,
    "9d072e4c": 43,
    "9d3ec64c": 7,
    "9d4272e3": 57,
    "9d5e14d4": 12,
    "9da441ba": 31,
    "9dede370": 53,
    "9def2eb2": 35,
    "9df1b1f9": 43,
    "9dfff011": 33,
    "9e3e364d": 31,
    "9efc812e": 21,
    "9f0c7bae": 63,
    "9ffce529": 46,
    "a0383629": 26,
    "a056d01e": 37,
    "a08d9e63": 16,
    "a0b83014": 31,
    "a0e92ed6": 44,
    "a15bc102": 17,
    "a247e7cf": 14,
    "a2e8e7f6": 4,
    "a3518960": 48,
    "a3f71395": 63,
    "a4719920": 35,
    "a48640d9": 57,
    "a4cdbb0f": 14,
    "a4f989c2": 20,
    "a5aa7973": 15,
    "a5e6f7dc": 31,
    "a612d5a0": 20,
    "a622ce4d": 31,
    "a645da9a": 43,
    "a692b17e": 26,
    "a6b8ac67": 63,
    "a6ed2eff": 31,
    "a6f967fb": 59,
    "a76db406": 5,
    "a7744f5b": 26,
    "a783cc24": 5,
    "a79fd2e7": 43,
    "a7fc3e6b": 17,
    "a85bb86f": 24,
    "a85e4bc3": 0,
    "a87433c9": 39,
    "a959858c": 49,
    "a9c9b150": 20,
    "aa95015b": 31,
    "aaaf3c03": 46,
    "aaeffccb": 52,
    "ab112491": 52,
    "ab18d3fc": 52,
    "ab3ced07": 52,
    "ab6fe95d": 0,
    "ab94878e": 21,
    "abf460d2": 52,
    "ac6f01d5": 55,
    "ad2133a8": 46,
    "add9c322": 8,
    "ae069086": 21,
    "ae0784a8": 47,
    "ae35f33a": 63,
    "ae8959c3": 6,
    "aed44918": 65,
    "aee6393a": 16,
    "af7a59ce": 29,
    "aff4e561": 14,
    "b04b58a3": 5,
    "b0d42b0d": 20,
    "b0f53bf4": 51,
    "b19b0395": 26,
    "b1efb8a4": 43,
    "b32902c0": 8,
    "b32e4903": 26,
    "b3388334": 36,
    "b37fd114": 35,
    "b38e3116": 43,
    "b4d8dd4f": 25,
    "b4f37e6e": 25,
    "b56bc0aa": 25,
    "b585b4fa": 26,
    "b5aa9634": 57,
    "b7974e66": 47,
    "b7cdcecb": 14,
    "b7fea9ef": 5,
    "b85f7b06": 31,
    "b881d27e": 26,
    "b8c49c1a": 5,
    "b95e7121": 31,
    "b977be4a": 31,
    "ba48188d": 52,
    "bb2cdd83": 26,
    "bb337fa0": 33,
    "bb682ebd": 15,
    "bc4381e2": 50,
    "bcebcc5f": 17,
    "bd2c2f34": 3,
    "bd4ae5da": 41,
    "bd6f0e19": 31,
    "bd8f847e": 62,
    "be063506": 41,
    "be35c8f2": 55,
    "be83e781": 26,
    "beda8696": 14,
    "bf39bc20": 25,
    "bf70b410": 14,
    "bffe3082": 41,
    "c03b9305": 5,
    "c03d4aed": 63,
    "c03fba65": 7,
    "c07fd0cd": 58,
    "c1708b88": 32,
    "c1d046f4": 31,
    "c2717d4f": 31,
    "c2c4db09": 31,
    "c2c6fc71": 52,
    "c36625df": 41,
    "c3957531": 59,
    "c42fe306": 12,
    "c43812e2": 53,
    "c472c0b5": 6,
    "c47e0767": 52,
    "c50b42f6": 14,
    "c593337f": 47,
    "c59b6c4a": 14,
    "c66be2b8": 31,
    "c6797dd7": 43,
    "c6b782a4": 21,
    "c6c96179": 26,
    "c70c0e01": 57,
    "c84c65be": 0,
    "c8d9680c": 31,
    "c908edd0": 31,
    "c9578d27": 26,
    "c96c018a": 31,
    "c99f9fae": 49,
    "c9e6734f": 21,
    "c9e980e8": 31,
    "c9fedcf3": 47,
    "cad13e8f": 61,
    "cb34231f": 61,
    "cb879f40": 8,
    "cbb406b5": 31,
    "cbe51de1": 49,
    "cbe62450": 8,
    "cc08aa63": 25,
    "cc8a1d57": 46,
    "ccedb12b": 60,
    "cd7f1687": 35,
    "cda7eb37": 46,
    "cdc31d65": 52,
    "cdc60a53": 26,
    "ce55ba43": 33,
    "ce8399b7": 21,
    "cf50c9d1": 21,
    "d00e7eb9": 11,
    "d011f41b": 20,
    "d07aed8f": 55,
    "d085e611": 8,
    "d0a0e7b5": 8,
    "d0a43760": 21,
    "d0cb63ae": 15,
    "d114d9c2": 14,
    "d1457cc5": 3,
    "d1ecf309": 15,
    "d1ee630d": 31,
    "d217e0c5": 33,
    "d24ff243": 26,
    "d273674e": 43,
    "d2f3b1ab": 26,
    "d3df146b": 49,
    "d40f61d5": 26,
    "d41beaf2": 21,
    "d463e527": 48,
    "d4f60a28": 59,
    "d50c649a": 41,
    "d543916f": 8,
    "d5df0ff5": 44,
    "d60430e6": 22,
    "d63196fe": 46,
    "d6c320d9": 26,
    "d6dd7330": 43,
    "d7ba4f9d": 57,
    "d84b621b": 26,
    "d90aa14c": 11,
    "d924e971": 43,
    "d9ae59b2": 15,
    "d9d6d94d": 60,
    "d9eca87b": 14,
    "da2d4d7c": 52,
    "da3bfe4e": 14,
    "da3cf798": 5,
    "da940160": 31,
    "dafc88c3": 26,
    "dbee6c86": 57,
    "dc5fbe29": 45,
    "dc7da28f": 12,
    "dc7f9757": 15,
    "dd168de0": 16,
    "dd64382d": 7,
    "dd6bc5b2": 20,
    "dd7d638e": 25,
    "ddc790ff": 31,
    "df71d1ff": 14,
    "df73b8f3": 57,
    "df8290be": 26,
    "e0079ed1": 17,
    "e03b45fd": 63,
    "e0f36d98": 33,
    "e149204f": 14,
    "e14d641e": 26,
    "e161bd0c": 52,
    "e201fd6d": 55,
    "e25f1537": 5,
    "e2fc7745": 44,
    "e32da4ec": 44,
    "e3b34f5b": 17,
    "e45a2a62": 26,
    "e46f4ef4": 55,
    "e4726608": 44,
    "e5864244": 59,
    "e5c92e59": 49,
    "e5ff9fd2": 37,
    "e6c55748": 26,
    "e730c32b": 31,
    "e737965a": 12,
    "e748dd88": 43,
    "e7818f7a": 15,
    "e7c36e81": 24,
    "e7d0c1de": 31,
    "e85074c5": 16,
    "e97032ad": 24,
    "e9949a23": 46,
    "ea0b99b7": 57,
    "ea3a0e38": 17,
    "ea41324e": 31,
    "eabc711b": 41,
    "eac10e86": 14,
    "eafb863e": 24,
    "eb06a4e7": 31,
    "eb640a35": 8,
    "eb934281": 47,
    "eba6605e": 28,
    "ebe69318": 26,
    "ec0d597b": 12,
    "ec13af42": 52,
    "ecdab904": 50,
    "ecdaf714": 16,
    "ed6436d6": 52,
    "ed6e6e54": 43,
    "edf885a3": 52,
    "ee0300f7": 55,
    "ee288beb": 15,
    "ee4aecac": 37,
    "eeda36f5": 63,
    "eeed308d": 26,
    "ef04658e": 5,
    "ef8e3ed0": 47,
    "efde6ac3": 8,
    "efe96181": 5,
    "f0188a48": 35,
    "f021b650": 50,
    "f03f10fe": 26,
    "f053a1b9": 52,
    "f074d277": 43,
    "f07d7037": 47,
    "f08774c3": 46,
    "f140c2fa": 46,
    "f16e25e3": 8,
    "f2d4c8c9": 14,
    "f321a31c": 57,
    "f40cd091": 40,
    "f439a621": 16,
    "f49fdea3": 50,
    "f4d12d23": 25,
    "f56d3ace": 31,
    "f5859199": 54,
    "f5d8aab6": 15,
    "f631da6e": 20,
    "f6824bb0": 26,
    "f6bc699b": 17,
    "f6d009f4": 15,
    "f72b268f": 15,
    "f8662c22": 37,
    "f88ddb26": 50,
    "f8afa78a": 14,
    "f8f59188": 31,
    "f9fc81aa": 26,
    "fa16d114": 14,
    "fa31da94": 49,
    "fa667be2": 57,
    "fae0c593": 8,
    "fb03ae90": 36,
    "fb0904bd": 17,
    "fb3848a1": 12,
    "fb73b13a": 15,
    "fba7683c": 41,
    "fbd68d27": 26,
    "fc03f21f": 40,
    "fc0d20b2": 5,
    "fcfcc902": 35,
    "fd3b4faa": 31,
    "fd710aea": 52,
    "fd8f77fa": 31,
    "fde20ecf": 44,
    "fdfd57da": 31,
    "febb4411": 59,
    "fef8af96": 16,
    "ff0aea78": 37,
    "ff8bb73a": 21,
    "ffefef30": 63,
}


# ============================================================================
# Numba JIT Beam kernel (inspired by romantamrazov ll. 108-144, Apache 2.0)
# ============================================================================
@njit(cache=True)
def _beam_kernel(sgr, tw_gr, start_idx, beam_size, move_cost, emit_scale):
    n = len(sgr)
    n_tw = len(tw_gr)
    cap = beam_size * 5
    beam_idx = np.zeros(beam_size, dtype=np.int64)
    beam_idx[0] = start_idx
    beam_cost = np.full(beam_size, 1e30, dtype=np.float64)
    beam_cost[0] = 0.0
    n_active = np.int64(1)
    hist_idx = np.zeros((n, beam_size), dtype=np.int64)
    hist_parent = np.zeros((n, beam_size), dtype=np.int64)
    cand_idx = np.zeros(cap, dtype=np.int64)
    cand_cost = np.full(cap, 1e30, dtype=np.float64)
    cand_parent = np.zeros(cap, dtype=np.int64)
    for step in range(n):
        gv = sgr[step]
        n_cand = np.int64(0)
        for bi in range(n_active):
            cur_idx = beam_idx[bi]
            cur_cost = beam_cost[bi]
            for d in range(-2, 3):
                next_idx = cur_idx + d
                if next_idx < 0 or next_idx >= n_tw:
                    continue
                ad = d if d >= 0 else -d
                tot = (
                    cur_cost
                    + (gv - tw_gr[next_idx]) * (gv - tw_gr[next_idx]) / emit_scale
                    + move_cost * ad
                )
                found = np.int64(-1)
                for ci in range(n_cand):
                    if cand_idx[ci] == next_idx:
                        found = ci
                        break
                if found >= 0:
                    if tot < cand_cost[found]:
                        cand_cost[found] = tot
                        cand_parent[found] = bi
                else:
                    if n_cand < cap:
                        cand_idx[n_cand] = next_idx
                        cand_cost[n_cand] = tot
                        cand_parent[n_cand] = bi
                        n_cand += 1
        kept = min(beam_size, n_cand)
        for i in range(kept):
            min_i = i
            for j in range(i + 1, n_cand):
                if cand_cost[j] < cand_cost[min_i]:
                    min_i = j
            if min_i != i:
                tmp_i = cand_idx[i]
                cand_idx[i] = cand_idx[min_i]
                cand_idx[min_i] = tmp_i
                tmp_c = cand_cost[i]
                cand_cost[i] = cand_cost[min_i]
                cand_cost[min_i] = tmp_c
                tmp_p = cand_parent[i]
                cand_parent[i] = cand_parent[min_i]
                cand_parent[min_i] = tmp_p
        hist_idx[step, :kept] = cand_idx[:kept]
        hist_parent[step, :kept] = cand_parent[:kept]
        beam_idx[:kept] = cand_idx[:kept]
        beam_cost[:kept] = cand_cost[:kept]
        n_active = kept
    best = np.int64(0)
    for b in range(1, n_active):
        if beam_cost[b] < beam_cost[best]:
            best = b
    path = np.zeros(n, dtype=np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hist_idx[s, b]
        b = hist_parent[s, b]
    return path


def _nn(a, v):
    i = int(np.searchsorted(a, v, "left"))
    if i >= len(a):
        return len(a) - 1
    if i > 0 and abs(a[i - 1] - v) <= abs(a[i] - v):
        return i - 1
    return i


def _smooth(v, fb, r):
    s = pd.Series(v, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20.0, es=144.0, r=2):
    si = _nn(tw_tvt, float(start_tvt))
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_kernel(sgr, tw_gr.astype(np.float64), np.int64(si), np.int64(bs), float(mc), float(es))].astype(
        np.float32
    )


print("Warming up Numba JIT...")
_beam_kernel(np.random.randn(30), np.random.randn(50), np.int64(25), np.int64(8), 15.0, 100.0)
print("Numba beam JIT ok")


# ============================================================================
# Particle Filters (inspired by romantamrazov ll. 223-313, Apache 2.0)
# ============================================================================
PF_GR_SIG_MIN, PF_GR_SIG_MAX, PF_GR_SIG_DEF = 10.0, 60.0, 30.0


def _gr_sigma(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return PF_GR_SIG_DEF
    return float(
        np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)), PF_GR_SIG_MIN, PF_GR_SIG_MAX)
    )


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    gs = _gr_sigma(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    pos = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1]) + rng.normal(0.0, 0.3, N)
    rate = ir + rng.normal(0.0, 0.01, N)
    w = np.ones(N) / N
    md_v = ev["MD"].values
    z_v = ev["Z"].values
    gr_v = ev["GR"].values
    pm = float(kn["MD"].iloc[-1])
    pts = np.empty(len(ev))
    std_o = np.empty(len(ev))
    for i in range(len(ev)):
        dm2 = max(md_v[i] - pm, 1.0)
        rate = 0.998 * rate + rng.normal(0.0, 0.002, N)
        pos = pos + rate * dm2 + rng.normal(0.0, 0.005, N)
        tvt_e = np.clip(pos - z_v[i], tmin - 50.0, tmax + 50.0)
        pos = tvt_e + z_v[i]
        if not np.isnan(gr_v[i]):
            eg = np.interp(tvt_e, tw_tvt, tw_gr)
            lk = np.exp(-0.5 * ((gr_v[i] - eg) / gs) ** 2)
            lk = np.maximum(lk, 1e-300)
            w *= lk
            ws = w.sum()
            w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
        ne = 1.0 / np.sum(w * w)
        if ne < 0.5 * N:
            ix = np.searchsorted(np.cumsum(w), (np.arange(N) + rng.uniform()) / N)
            ix = np.clip(ix, 0, N - 1)
            pos = pos[ix]
            rate = rate[ix]
            w[:] = 1.0 / N
            pos += rng.normal(0.0, 0.1, N)
            rate += rng.normal(0.0, 0.001, N)
        tv = float(np.average(pos - z_v[i], weights=w))
        pts[i] = tv
        std_o[i] = float(np.sqrt(np.average((pos - z_v[i] - tv) ** 2, weights=w)))
        pm = md_v[i]
    return pts.astype(np.float32), std_o.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N, rng_seed=SEED + 1):
    rng = np.random.default_rng(rng_seed)
    tw_s = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().values
    tf_p = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    tf_s = interp1d(tw_tvt, tw_s, bounds_error=False, fill_value=(tw_s[0], tw_s[-1]))
    tmin, tmax = tw_tvt.min(), tw_tvt.max()
    gs = _gr_sigma(hw, tw_tvt, tw_gr)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = c[0], c[1], max(np.std(vt - (c[0] * vz + c[1])), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    tail2 = kna.tail(20)
    dvt2 = np.diff(tail2["TVT_input"].values)
    dmd2 = np.diff(tail2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0
    gr_sm = hw["GR"].rolling(5, center=True, min_periods=1).mean()
    pos = float(kna["TVT_input"].iloc[-1]) + rng.normal(0.0, 0.5, N)
    vel = iv + rng.normal(0.0, 0.02, N)
    w = np.ones(N) / N
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
        vel = 0.993 * vel + rng.normal(0.0, 0.005, N)
        pos = pos + vel * dm + rng.normal(0.0, 0.01, N)
        pos = np.clip(pos, tmin - 50.0, tmax + 50.0)
        if not np.isnan(gr_v[i]):
            ep = tf_p(pos)
            lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)
            try:
                gsm = gr_sm.iloc[hw.index.get_loc(idx)]
            except KeyError:
                gsm = np.nan
            if not np.isnan(gsm):
                ls2 = np.exp(-0.5 * ((gsm - tf_s(pos)) / (gs * 1.5)) ** 2)
                lk = 0.7 * lp + 0.3 * ls2
            else:
                lk = lp
            lk = np.maximum(lk, 1e-300)
            w *= lk
            ws = w.sum()
            w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
        lz = np.exp(-0.5 * ((vel - ve) / max(zsig * 2.0, 0.005)) ** 2)
        lz = np.maximum(lz, 1e-300)
        w *= lz
        ws = w.sum()
        w = (w / ws) if ws > 0 else np.full(N, 1.0 / N)
        ne = 1.0 / np.sum(w * w)
        if ne < 0.5 * N:
            ix = np.searchsorted(np.cumsum(w), (np.arange(N) + rng.uniform()) / N)
            ix = np.clip(ix, 0, N - 1)
            pos = pos[ix]
            vel = vel[ix]
            w[:] = 1.0 / N
            pos += rng.normal(0.0, 0.2, N)
            vel += rng.normal(0.0, 0.003, N)
        pts[i] = np.average(pos, weights=w)
        std_o[i] = np.sqrt(np.average((pos - pts[i]) ** 2, weights=w))
        pm = md_v[i]
        pz = z_v[i]
    return pts.astype(np.float32), std_o.astype(np.float32)


# ============================================================================
# Feature helpers
# ============================================================================
def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def wls_b_well(ktvt, kz, form_col, decay=0.02):
    n = len(ktvt)
    if n < 3:
        return float(np.median(ktvt + kz - form_col))
    w = np.exp(decay * np.arange(n))
    w /= w.sum()
    return float(np.dot(w, ktvt + kz - form_col))


def multi_scale_sc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
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


def gr_detrend_resid(gr_arr, md_arr):
    m = np.isfinite(gr_arr) & np.isfinite(md_arr)
    if m.sum() < 5:
        return gr_arr.copy()
    slope = robust_slope(md_arr[m], gr_arr[m])
    return (gr_arr - slope * md_arr).astype(np.float32)


# ============================================================================
# Spatial Imputers
# ============================================================================
class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
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
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack(
            [(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1
        )
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except np.linalg.LinAlgError:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
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
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=3000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (
            ap.astype(np.float32),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            np.where(vk, dk, np.inf).min(1).astype(np.float32),
        )


# ============================================================================
# Per-well feature builder
# ============================================================================
ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
UNKNOWN_CLUSTER = -1


def build_well(hw_path, tw_path, is_train, fi, di, use_self_exclusion):
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

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a
    std_use = std_a
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

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

    bpaths = {}
    for bs, mc, es, r, tag in BEAM_CONFIGS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res = multi_scale_sc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]
    sc15, sc15s = sc_res[1]
    sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc15

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)

    kmd = kn["MD"].to_numpy(np.float32)
    kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt)
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)
    pfx_gr_slope = robust_slope(kmd, kgr)

    swid = wid if (is_train and use_self_exclusion) else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = fi.impute(xy_ev, self_wid=swid)
    form_kn, _ = fi.impute(xy_kn, self_wid=swid)

    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
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

    all_sigs = [pf_use, *(p for p in bpaths.values()), sc8, sc15, sc25, tvt_fs["ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values)
    rolls = {}
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

    md_since = hmd_s - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    d1_cluster = np.full(nh, D1_CLUSTER_MAP.get(wid, UNKNOWN_CLUSTER), dtype=np.int32)

    feats = {
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
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
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
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
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


def build_dataset(paths, is_train, fi, di, label, use_self_exclusion=True):
    args = [
        (str(p), str(p.parent / f"{p.stem.replace('__horizontal_well', '')}__typewell.csv"), is_train)
        for p in paths
        if (p.parent / f"{p.stem.replace('__horizontal_well', '')}__typewell.csv").exists()
    ]
    print(f"  {label}: {len(args)} wells | {NCPU} threads", flush=True)
    res = Parallel(n_jobs=NCPU, prefer="threads", verbose=3)(
        delayed(build_well)(hp, tp, it, fi, di, use_self_exclusion) for hp, tp, it in args
    )
    parts = [r for r in res if r is not None]
    print(f"  {label}: OK={len(parts)} skipped={len(args) - len(parts)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ============================================================================
# Train + predict
# ============================================================================
# Best params (will be filled in from HPO local run, AC-14).
# (HPO not yet run; using baseline params.)
LGB_HPO_BEST = None
_HPO_OBJECTIVE = "regression"
_HPO_HUBER_ALPHA = None
CB_HPO_BEST = None
LGB_BASE = dict(
    boosting_type="gbdt",
    num_leaves=255,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_lambda=3.0,
    reg_alpha=0.05,
    min_split_gain=0.01,
    objective="regression",
    verbose=-1,
    n_jobs=-1,
    max_bin=255,
)
LGB_CONFIGS = [
    dict(learning_rate=0.025, n_estimators=8000, seed=42),
    dict(learning_rate=0.020, n_estimators=8000, seed=7),
    dict(learning_rate=0.030, n_estimators=8000, seed=123),
]
CB_PARAMS = dict(
    iterations=8000,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=2.0,
    min_data_in_leaf=15,
    subsample=0.75,
    border_count=254,
    loss_function="RMSE",
    random_seed=SEED,
    od_type="Iter",
    od_wait=300,
    verbose=0,
)
CATEGORICAL_FEATURES = ["d1_cluster_id"]


def _train_main():
    hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
    print(f"Building imputers ({len(train_wids)} wells)...", flush=True)
    fi = FormationPlaneKNN(train_wids, TRAIN_DIR)
    di = DenseANCCImputer(train_wids, TRAIN_DIR)
    print(f"  FPK={len(fi.df)} centroids | Dense={len(di.ancc):,} pts | elapsed={_elapsed():.0f}s", flush=True)
    if _time_left() < 600:
        print("TIMEOUT before train build, fallback to mean prediction.", flush=True)
        return _fallback_submission()
    print("Building train...")
    t0 = time.time()
    train_df = build_dataset(hw_paths, True, fi, di, "train")
    print(f"  train: {train_df.shape}  ({time.time() - t0:.0f}s)", flush=True)
    if _time_left() < 600:
        print("TIMEOUT after train build, fallback.", flush=True)
        return _fallback_submission()
    test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    print("Building test...")
    t0 = time.time()
    test_df = build_dataset(test_paths, False, fi, di, "test", use_self_exclusion=False)
    print(f"  test:  {test_df.shape}  ({time.time() - t0:.0f}s)", flush=True)

    SKIP = {"well", "id", "target"}
    feature_cols = [c for c in train_df.columns if c not in SKIP]
    print(f"#features: {len(feature_cols)}", flush=True)

    X = train_df[feature_cols]
    y = train_df["target"]
    g = train_df["well"]
    Xt = test_df[feature_cols]
    gc.collect()

    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(X, y, g))
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    def run_lgb(cfg_idx):
        cfg = LGB_CONFIGS[cfg_idx]
        p = dict(LGB_BASE, **cfg)
        n_est = p.pop("n_estimators")
        oof = np.zeros(len(train_df), np.float32)
        tp = np.zeros(len(test_df), np.float32)
        for fold, (tr, va) in enumerate(splits):
            dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=cat_cols)
            dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=cat_cols)
            m = lgb.train(
                p, dtr, valid_sets=[dva], num_boost_round=n_est,
                callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(800)],
            )
            oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration).astype(np.float32)
            tp += m.predict(Xt, num_iteration=m.best_iteration).astype(np.float32) / N_SPLITS
            print(f"  LGB{cfg_idx} f{fold}: rmse={root_mean_squared_error(y.iloc[va], oof[va]):.4f} iter={m.best_iteration}", flush=True)
        r = root_mean_squared_error(y, oof)
        print(f"  LGB{cfg_idx} OOF={r:.4f}", flush=True)
        return oof, tp, r

    def run_cb():
        oof = np.zeros(len(train_df), np.float32)
        tp = np.zeros(len(test_df), np.float32)
        cat_idx = [feature_cols.index(c) for c in cat_cols] if cat_cols else None
        for fold, (tr, va) in enumerate(splits):
            m = CatBoostRegressor(**CB_PARAMS)
            m.fit(
                Pool(X.iloc[tr].values, label=y.iloc[tr].values, cat_features=cat_idx),
                eval_set=Pool(X.iloc[va].values, label=y.iloc[va].values, cat_features=cat_idx),
                use_best_model=True,
            )
            oof[va] = m.predict(X.iloc[va].values).astype(np.float32)
            tp += m.predict(Xt.values).astype(np.float32) / N_SPLITS
            print(f"  CB f{fold}: rmse={root_mean_squared_error(y.iloc[va], oof[va]):.4f}", flush=True)
        r = root_mean_squared_error(y, oof)
        print(f"  CB OOF={r:.4f}", flush=True)
        return oof, tp, r

    results = {}
    for i in range(3):
        if _time_left() < 600:
            print(f"TIMEOUT at LGB{i}, breaking", flush=True)
            break
        oof, tp, r = run_lgb(i)
        results[f"lgb{i}"] = {"oof": oof, "test": tp, "rmse": r}
    if _time_left() >= 600:
        oof, tp, r = run_cb()
        results["cb"] = {"oof": oof, "test": tp, "rmse": r}

    Sx = np.column_stack([v["oof"] for v in results.values()])
    St = np.column_stack([v["test"] for v in results.values()])
    ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
    ridge.fit(Sx, y.values)
    oof_s = ridge.predict(Sx)
    test_s = ridge.predict(St)
    r_avg = root_mean_squared_error(y, Sx.mean(1))
    r_stk = root_mean_squared_error(y, oof_s)
    print(f"\nSimple avg: {r_avg:.4f}  |  Ridge stack: {r_stk:.4f}", flush=True)
    final_oof = oof_s if r_stk < r_avg else Sx.mean(1)
    final_test = test_s if r_stk < r_avg else St.mean(1)

    base = train_df["last_known_tvt"].values
    ytrue = y.values + base
    pf_oof = (train_df["pf_ancc"].values - base)
    best_cfg, best_r = (None, None, None), np.inf
    for alpha in np.arange(0.65, 1.01, 0.05):
        for tau in [None, 25.0, 50.0, 100.0, 200.0]:
            for w_pf in [0.0, 0.05, 0.10]:
                d = final_oof * (1 - w_pf) + pf_oof * w_pf
                if tau:
                    d *= (1.0 - np.exp(-np.maximum(train_df["md_since"].values, 0.0) / tau))
                d *= alpha
                r = root_mean_squared_error(ytrue, base + d)
                if r < best_r:
                    best_r, best_cfg = r, (alpha, tau, w_pf, None)
    print(f"Best PP: alpha={best_cfg[0]:.2f} tau={best_cfg[1]} w_pf={best_cfg[2]:.2f}  abs TVT RMSE={best_r:.4f}", flush=True)
    ALPHA, TAU, W_PF = best_cfg[0], best_cfg[1], best_cfg[2]

    def apply_pp(df, delta, pf_delta, alpha, tau, w_pf):
        d = delta * (1 - w_pf) + pf_delta * w_pf
        if tau:
            d *= (1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau))
        return d * alpha

    def sg_smooth(df, col, sg_w=17, sg_p=3):
        df = df.copy()
        for _well, gp in df.groupby("well", sort=False):
            v = gp[col].values
            n = len(v)
            wl = min(sg_w, n)
            if wl % 2 == 0:
                wl -= 1
            if wl >= sg_p + 2:
                v = savgol_filter(v, wl, sg_p)
            df.loc[gp.index, col] = v
        return df

    test_df2 = test_df.copy()
    pf_test = (test_df2["pf_ancc"].values - test_df2["last_known_tvt"].values)
    test_df2["pred"] = (test_df2["last_known_tvt"].values + apply_pp(test_df2, final_test, pf_test, ALPHA, TAU, W_PF))
    test_df2 = sg_smooth(test_df2, "pred")

    sample = pd.read_csv(SAMPLE)
    sub = (sample[["id"]].merge(test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}), on="id", how="left"))
    fb = float(train_df["last_known_tvt"].mean() + train_df["target"].mean())
    sub["tvt"] = sub["tvt"].fillna(fb)
    sub[["id", "tvt"]].to_csv(OUT, index=False)
    print(f"\nDone {OUT}  {len(sub)} rows  elapsed={_elapsed():.0f}s", flush=True)
    return 0


def _fallback_submission():
    print("Fallback to mean-TVT submission.", flush=True)
    sample = pd.read_csv(SAMPLE)
    train_files = list(TRAIN_DIR.glob("*__horizontal_well.csv"))[:20]
    means = []
    for p in train_files:
        try:
            df = pd.read_csv(p, usecols=["TVT"]).dropna()
            if len(df):
                means.append(df["TVT"].mean())
        except Exception:
            continue
    fb = float(np.mean(means)) if means else 11500.0
    sample["tvt"] = fb
    sample[["id", "tvt"]].to_csv(OUT, index=False)
    print(f"Wrote fallback {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    print(f"GPUs: (skipped check)  CPUs={NCPU}  HARD_TIMEOUT={HARD_TIMEOUT_S}s", flush=True)
    try:
        _train_main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        _fallback_submission()
