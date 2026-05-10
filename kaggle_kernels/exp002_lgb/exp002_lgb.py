"""ROGII exp002 — Kaggle script submission.

Replicates local exp002:
    target = TVT - last_known_TVT (residual)
    tvt_formula = -Z + ANCC_imputed + b_well  (6 formations × FormationPlaneKNN)
    LightGBM 5-fold GroupKFold by well, num_leaves 63, lr 0.1

Local CV TVT hidden RMSE = 13.82 ft.

Runtime: <2 hr expected on Kaggle CPU (8-16 cores). Internet disabled.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# -------------------- env --------------------

# Auto-detect data root: Kaggle uses /kaggle/input/<comp>/, but path may vary by run mode.
import os
import sys

def _find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for p in candidates:
        if (p / "train").exists() and (p / "test").exists():
            return p
    # last resort: scan /kaggle/input/* for any directory containing train/test
    if Path("/kaggle/input").exists():
        for sub in Path("/kaggle/input").iterdir():
            if (sub / "train").exists() and (sub / "test").exists():
                return sub
    return Path("data/raw")  # fallback for local runs


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
NUM_BOOST_ROUND = 2000
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
                # Try with formation cols first; some test-like wells may not have them
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
                f"FormationPlaneKNN: no training wells with formation columns found in {train_dir}. "
                f"This usually means the CSV path is wrong. DATA_DIR={DATA_DIR}"
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
        min_dist = np.where(vk, dk, np.inf).min(1).astype(np.float32)
        return pred, min_dist


# -------------------- features --------------------


def _planefit(visible_x, visible_y, visible_z, visible_t):
    if len(visible_t) < 4:
        return 0.0, 0.0, 0.0, float(visible_t.mean()) if len(visible_t) else 0.0
    A = np.column_stack([visible_x, visible_y, visible_z, np.ones(len(visible_t))])
    coef, *_ = np.linalg.lstsq(A, visible_t, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])


def _rolling_mean_std(arr: np.ndarray, window: int):
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=1, center=True).mean().to_numpy()
    sd = s.rolling(window, min_periods=1, center=True).std().fillna(0.0).to_numpy()
    return m, sd


def _per_well_features(well_df: pd.DataFrame, formation_imputed):
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
            tvt_F_all = (-z + f_imp + b_F_all).astype(np.float32)
            tvt_F_50 = (-z + f_imp + b_F_50).astype(np.float32)
            out[f"tvtF_{fname}"] = tvt_F_all
            out[f"tvtF50_{fname}"] = tvt_F_50
            out[f"bw_{fname}"] = np.full(n, b_F_all, dtype=np.float32)
            out[f"bw50_{fname}"] = np.full(n, b_F_50, dtype=np.float32)
            out[f"tvtF_{fname}_d"] = (tvt_F_all - last_tvt).astype(np.float32)
            out[f"f_imp_{fname}"] = f_imp

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
        ]
    return base


FEATURE_COLS = feature_columns()


# -------------------- IO --------------------


def list_wells(split_dir: Path) -> list[str]:
    return sorted({p.stem.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv")})


def load_horizontal(split_dir: Path, well: str) -> pd.DataFrame:
    df = pd.read_csv(split_dir / f"{well}__horizontal_well.csv")
    df.insert(0, "well", well)
    df["row_idx"] = np.arange(len(df), dtype=np.int32)
    return df


def load_split(split_dir: Path, imputer: FormationPlaneKNN, exclude_self: bool):
    wells = list_wells(split_dir)
    all_dfs: list[pd.DataFrame] = []
    print(f"   {len(wells)} wells in {split_dir.name}", flush=True)
    t0 = time.perf_counter()
    for i, wid in enumerate(wells):
        sub = load_horizontal(split_dir, wid)
        xy = sub[["X", "Y"]].to_numpy(np.float64)
        self_wid = wid if exclude_self else None
        formation_imputed, _ = imputer.impute(xy, self_wid=self_wid)
        feats = _per_well_features(sub, formation_imputed=formation_imputed)
        # Build per-well DataFrame
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
    return pd.concat(all_dfs, ignore_index=True)


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
    print("ROGII exp002 — residual + tvt_formula + 5-fold LightGBM", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] build FormationPlaneKNN imputer", flush=True)
    t0 = time.perf_counter()
    imputer = FormationPlaneKNN(TRAIN_DIR, k=10)
    print(f"   {len(imputer.df)} centroids in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n[2/4] feature engineering — train", flush=True)
    train = load_split(TRAIN_DIR, imputer, exclude_self=True)
    print(f"   train shape: {train.shape}", flush=True)

    print("\n[2/4] feature engineering — test", flush=True)
    test = load_split(TEST_DIR, imputer, exclude_self=False)
    print(f"   test shape: {test.shape}", flush=True)

    last_known_train = train["last_known_TVT"].to_numpy(np.float32)
    y_residual = (train["TVT"].to_numpy(np.float32) - last_known_train).astype(np.float32)
    train_hidden = train["TVT_input"].isna().to_numpy()
    well_ids = train["well"].to_numpy()
    last_known_test = test["last_known_TVT"].to_numpy(np.float32)
    test_hidden = test["TVT_input"].isna().to_numpy()

    X_train = train[FEATURE_COLS].astype(np.float32)
    X_test = test[FEATURE_COLS].astype(np.float32)

    print(
        f"\n   features: {len(FEATURE_COLS)}  |  residual std={y_residual.std():.2f}  |  "
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
