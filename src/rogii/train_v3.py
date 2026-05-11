"""Train + CV runner for ROGII exp004 (Approach A self-reimplementation).

Entry point invoked by `.criteria/kaggle-rogii-exp004-numba-beam-pf.yaml` AC-3:

    .venv/bin/python -m rogii.train_v3 --exp exp004 --cv-only --quiet

Prints final summary line:

    CV TVT hidden RMSE = X.XXXX

Inspired by romantamrazov/rogii-super-solution-lb-top-3 (Apache 2.0).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from rogii.beam import warmup as beam_warmup
from rogii.features_d1 import build_cluster_map
from rogii.features_v4 import build_well
from rogii.imputers import DenseANCCImputer, FormationPlaneKNN

DATA_ROOT_DEFAULT = "/home/yusuke_kaya/projects/kaggle/ROGII/data/raw"
GROUPS_PATH_DEFAULT = "/home/yusuke_kaya/projects/kaggle/ROGII/data/processed/typewell_groups.parquet"
CLUSTER_PARQUET = Path(
    "/home/yusuke_kaya/projects/kaggle/ROGII/outputs/eda/deepest_eda/per-well-outlier.parquet"
)

LGB_BASE: dict[str, Any] = dict(
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
LGB_CONFIGS: list[dict[str, Any]] = [
    dict(learning_rate=0.025, n_estimators=5000, seed=42),
    dict(learning_rate=0.020, n_estimators=5000, seed=7),
    dict(learning_rate=0.030, n_estimators=5000, seed=123),
]
CB_PARAMS: dict[str, Any] = dict(
    iterations=5000,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=2.0,
    min_data_in_leaf=15,
    subsample=0.75,
    border_count=254,
    loss_function="RMSE",
    random_seed=42,
    od_type="Iter",
    od_wait=300,
    verbose=0,
)
N_SPLITS = 5
CATEGORICAL_FEATURES = ["d1_cluster_id"]


def _resolve_paths() -> tuple[Path, Path, Path]:
    data_root = Path(os.environ.get("ROGII_DATA_ROOT", DATA_ROOT_DEFAULT))
    return data_root / "train", data_root / "test", Path(GROUPS_PATH_DEFAULT)


def build_features(
    train_dir: Path,
    test_dir: Path,
    cluster_map: dict[str, int],
    n_jobs: int = 4,
    quiet: bool = False,
    n_wells_limit: int | None = None,
    use_self_exclusion: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not quiet:
        print(f"[build] train_dir={train_dir} test_dir={test_dir}", flush=True)
    t0 = time.perf_counter()
    fi = FormationPlaneKNN(train_dir)
    di = DenseANCCImputer(train_dir)
    if not quiet:
        print(f"[build] imputers built in {time.perf_counter() - t0:.0f}s", flush=True)

    train_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    test_paths = sorted(test_dir.glob("*__horizontal_well.csv"))
    if n_wells_limit is not None:
        train_paths = train_paths[:n_wells_limit]

    def _args(paths: list[Path]) -> list[tuple[Path, Path]]:
        out = []
        for p in paths:
            tw = p.parent / f"{p.stem.replace('__horizontal_well', '')}__typewell.csv"
            if tw.exists():
                out.append((p, tw))
        return out

    train_args = _args(train_paths)
    test_args = _args(test_paths)
    if not quiet:
        print(f"[build] train wells={len(train_args)}  test wells={len(test_args)}", flush=True)

    t0 = time.perf_counter()
    train_res = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(build_well)(hp, tp, True, fi, di, cluster_map, use_self_exclusion) for hp, tp in train_args
    )
    train_parts = [r for r in train_res if r is not None]
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    if not quiet:
        print(
            f"[build] train_df shape={train_df.shape} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    t0 = time.perf_counter()
    test_res = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(build_well)(hp, tp, False, fi, di, cluster_map, False) for hp, tp in test_args
    )
    test_parts = [r for r in test_res if r is not None]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    if not quiet:
        print(
            f"[build] test_df shape={test_df.shape} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    return train_df, test_df


def make_folds(train_df: pd.DataFrame, groups_path: Path, n_splits: int = N_SPLITS) -> list:
    """GroupKFold over `group_id` from typewell_groups.parquet."""
    g = pd.read_parquet(groups_path)[["well_id", "group_id"]].drop_duplicates(subset="well_id")
    g_map = dict(zip(g["well_id"].astype(str), g["group_id"].astype(str), strict=True))
    train_groups = train_df["well"].map(g_map).fillna(train_df["well"]).to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(train_df, train_df["target"], train_groups))


def run_lgb_fold(
    cfg_idx: int,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    feature_cols: list[str],
    objective: str = "regression",
    huber_delta: float = 1.0,
    quiet: bool = False,
) -> tuple[np.ndarray, float]:
    cfg = LGB_CONFIGS[cfg_idx]
    p = dict(LGB_BASE, **cfg)
    p["objective"] = objective
    if objective in ("huber", "regression_l1"):
        p.pop("min_split_gain", None)
    if objective == "huber":
        p["alpha"] = huber_delta
    n_est = p.pop("n_estimators")
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    oof = np.zeros(len(X), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=cat_cols)
        m = lgb.train(
            p,
            dtr,
            valid_sets=[dva],
            num_boost_round=n_est,
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration).astype(np.float32)
        if not quiet:
            print(
                f"[lgb{cfg_idx}/{objective}] f{fold}: rmse={root_mean_squared_error(y.iloc[va], oof[va]):.4f}"
                f" iter={m.best_iteration}",
                flush=True,
            )
    r = root_mean_squared_error(y, oof)
    if not quiet:
        print(f"[lgb{cfg_idx}/{objective}] OOF={r:.4f}", flush=True)
    return oof, float(r)


def run_cb_fold(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    feature_cols: list[str],
    quiet: bool = False,
) -> tuple[np.ndarray, float]:
    oof = np.zeros(len(X), dtype=np.float32)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols] if cat_cols else None
    for fold, (tr, va) in enumerate(splits):
        m = CatBoostRegressor(**CB_PARAMS)
        m.fit(
            Pool(X.iloc[tr].values, label=y.iloc[tr].values, cat_features=cat_idx),
            eval_set=Pool(X.iloc[va].values, label=y.iloc[va].values, cat_features=cat_idx),
            use_best_model=True,
        )
        oof[va] = m.predict(X.iloc[va].values).astype(np.float32)
        if not quiet:
            print(f"[cb] f{fold}: rmse={root_mean_squared_error(y.iloc[va], oof[va]):.4f}", flush=True)
    r = root_mean_squared_error(y, oof)
    if not quiet:
        print(f"[cb] OOF={r:.4f}", flush=True)
    return oof, float(r)


def post_grid_search(
    final_oof: np.ndarray,
    pf_oof: np.ndarray,
    md_since: np.ndarray,
    ytrue_abs: np.ndarray,
    base: np.ndarray,
    quiet: bool = False,
) -> tuple[float, dict]:
    best_r = float("inf")
    best_cfg: dict[str, Any] = {}
    for alpha in np.arange(0.65, 1.01, 0.05):
        for tau in [None, 25.0, 50.0, 100.0, 200.0]:
            for w_pf in [0.0, 0.05, 0.10]:
                d = final_oof * (1 - w_pf) + pf_oof * w_pf
                if tau:
                    d = d * (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))
                d = d * alpha
                r = root_mean_squared_error(ytrue_abs, base + d)
                if r < best_r:
                    best_r = float(r)
                    best_cfg = dict(alpha=float(alpha), tau=tau, w_pf=float(w_pf))
    if not quiet:
        print(
            f"[postgrid] best alpha={best_cfg['alpha']:.2f} tau={best_cfg['tau']} "
            f"w_pf={best_cfg['w_pf']:.2f} abs RMSE={best_r:.4f}",
            flush=True,
        )
    return best_r, best_cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=False, default="exp004")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--objective", default="regression", choices=["regression", "huber", "regression_l1"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--use-d1", action="store_true", default=True)
    parser.add_argument("--no-d1", dest="use_d1", action="store_false")
    parser.add_argument("--cache", type=str, default="outputs/exp004/features.parquet",
                        help="Path to cached features parquet (skip rebuild if exists)")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--n-wells-limit", type=int, default=None)
    args = parser.parse_args()

    quiet = args.quiet
    if not quiet:
        print(f"[main] exp={args.exp} cv-only={args.cv_only} objective={args.objective}", flush=True)
    beam_warmup()

    cluster_map = build_cluster_map(CLUSTER_PARQUET) if args.use_d1 else {}
    if not quiet:
        print(f"[main] cluster_map size={len(cluster_map)} (D1 active={args.use_d1})", flush=True)

    train_dir, test_dir, groups_path = _resolve_paths()
    cache_train = Path(args.cache)
    cache_test = cache_train.with_name(cache_train.stem + "_test.parquet")
    cache_train.parent.mkdir(parents=True, exist_ok=True)

    if cache_train.exists() and cache_test.exists() and not args.rebuild and args.n_wells_limit is None:
        if not quiet:
            print(f"[main] loading cached features from {cache_train}", flush=True)
        train_df = pd.read_parquet(cache_train)
        test_df = pd.read_parquet(cache_test)
    else:
        train_df, test_df = build_features(
            train_dir,
            test_dir,
            cluster_map,
            n_jobs=args.n_jobs,
            quiet=quiet,
            n_wells_limit=args.n_wells_limit,
            use_self_exclusion=True,
        )
        if args.n_wells_limit is None:
            train_df.to_parquet(cache_train)
            test_df.to_parquet(cache_test)

    if train_df.empty:
        print("ERROR: no train features built", flush=True)
        return 1

    skip = {"well", "id", "target"}
    feature_cols = [c for c in train_df.columns if c not in skip]
    if not args.use_d1 and "d1_cluster_id" in feature_cols:
        feature_cols.remove("d1_cluster_id")
    if not quiet:
        print(f"[main] #features={len(feature_cols)}", flush=True)

    X = train_df[feature_cols]
    y = train_df["target"]
    splits = make_folds(train_df, groups_path)
    if not quiet:
        print(f"[main] folds: {len(splits)}", flush=True)

    oof_list: list[np.ndarray] = []
    rmses: dict[str, float] = {}
    for i in range(3):
        oof, r = run_lgb_fold(
            i, X, y, splits, feature_cols, objective=args.objective, huber_delta=args.huber_delta, quiet=quiet
        )
        oof_list.append(oof)
        rmses[f"lgb{i}"] = r
    cb_oof, cb_r = run_cb_fold(X, y, splits, feature_cols, quiet=quiet)
    oof_list.append(cb_oof)
    rmses["cb"] = cb_r

    Sx = np.column_stack(oof_list)
    ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
    ridge.fit(Sx, y.values)
    oof_s = ridge.predict(Sx).astype(np.float32)
    r_avg = root_mean_squared_error(y, Sx.mean(1))
    r_stk = root_mean_squared_error(y, oof_s)
    if not quiet:
        wts = ridge.coef_ / max(ridge.coef_.sum(), 1e-9)
        print(f"[stack] simple_avg={r_avg:.4f}  ridge={r_stk:.4f}  weights={wts.round(4)}", flush=True)
    final_oof = oof_s if r_stk < r_avg else Sx.mean(1)

    base = train_df["last_known_tvt"].values
    ytrue = y.values + base
    md_since = train_df["md_since"].values
    pf_oof = (train_df["pf_ancc"].values - base).astype(np.float32)
    best_r, best_cfg = post_grid_search(final_oof, pf_oof, md_since, ytrue, base, quiet=quiet)

    # Final CV report
    print(f"CV TVT hidden RMSE = {best_r:.4f}", flush=True)

    # Persist OOF + diagnostics
    outdir = Path(f"outputs/{args.exp}")
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "oof_final.npy", final_oof)
    np.save(outdir / "oof_pf.npy", pf_oof)
    with open(outdir / "rmses.json", "w") as f:
        json.dump({**rmses, "final_residual": min(r_avg, r_stk), "final_abs": best_r, "post": best_cfg}, f, indent=2)

    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
