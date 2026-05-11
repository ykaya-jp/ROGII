"""Optuna HPO for ROGII exp004 — LightGBM + CatBoost.

Per AC-14 (.criteria/kaggle-rogii-exp004-numba-beam-pf.yaml):
  - LGB search space: learning_rate [0.01-0.1], num_leaves [15-255],
    min_data_in_leaf [5-100], lambda_l1/l2 [0-10],
    feature_fraction [0.5-1.0], bagging_fraction [0.5-1.0]
  - CB search space:  learning_rate [0.01-0.1], depth [4-10], l2_leaf_reg [1-10]
  - 100 trial, TPE sampler, objective = CV TVT hidden RMSE (post-grid)
  - Output: best params hard-coded in kernel + notes.md

Invoke as:
    PYTHONPATH=src .venv/bin/python -m rogii.hpo --target lgb --trials 100 \
        --features outputs/exp004/features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from rogii.train_v3 import (
    CATEGORICAL_FEATURES,
    GROUPS_PATH_DEFAULT,
    N_SPLITS,
    post_grid_search,
)

SEED = 42
N_BOOST_HPO = 2500
EARLY_STOP_HPO = 150


def _load(cache_train: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(cache_train)
    skip = {"well", "id", "target"}
    feature_cols = [c for c in df.columns if c not in skip]
    X = df[feature_cols]
    y = df["target"]
    base = df["last_known_tvt"].values
    md_since = df["md_since"].values
    pf_oof_seed = (df["pf_ancc"].values - base).astype(np.float32)
    return X, y, base, md_since, pf_oof_seed


def _make_splits(df_train: pd.DataFrame) -> list:
    g = pd.read_parquet(GROUPS_PATH_DEFAULT)[["well_id", "group_id"]].drop_duplicates(subset="well_id")
    g_map = dict(zip(g["well_id"].astype(str), g["group_id"].astype(str), strict=True))
    train_groups = df_train["well"].map(g_map).fillna(df_train["well"]).to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)
    return list(cv.split(df_train, df_train["target"], train_groups))


def objective_lgb(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    feature_cols: list[str],
    base: np.ndarray,
    md_since: np.ndarray,
    pf_oof: np.ndarray,
) -> float:
    params: dict[str, Any] = dict(
        boosting_type="gbdt",
        objective="regression",
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
        lambda_l1=trial.suggest_float("lambda_l1", 0.0, 10.0),
        lambda_l2=trial.suggest_float("lambda_l2", 0.0, 10.0),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        verbose=-1,
        n_jobs=-1,
        seed=SEED,
    )
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    oof = np.zeros(len(X), dtype=np.float32)
    for tr, va in splits:
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=cat_cols)
        m = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            num_boost_round=N_BOOST_HPO,
            callbacks=[lgb.early_stopping(EARLY_STOP_HPO, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration).astype(np.float32)
    ytrue = y.values + base
    best_r, _ = post_grid_search(oof, pf_oof, md_since, ytrue, base, quiet=True)
    return best_r


def objective_cb(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    feature_cols: list[str],
    base: np.ndarray,
    md_since: np.ndarray,
    pf_oof: np.ndarray,
) -> float:
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols] if cat_cols else None
    params: dict[str, Any] = dict(
        iterations=N_BOOST_HPO,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        loss_function="RMSE",
        random_seed=SEED,
        od_type="Iter",
        od_wait=EARLY_STOP_HPO,
        verbose=0,
    )
    oof = np.zeros(len(X), dtype=np.float32)
    for tr, va in splits:
        m = CatBoostRegressor(**params)
        m.fit(
            Pool(X.iloc[tr].values, label=y.iloc[tr].values, cat_features=cat_idx),
            eval_set=Pool(X.iloc[va].values, label=y.iloc[va].values, cat_features=cat_idx),
            use_best_model=True,
        )
        oof[va] = m.predict(X.iloc[va].values).astype(np.float32)
    ytrue = y.values + base
    best_r, _ = post_grid_search(oof, pf_oof, md_since, ytrue, base, quiet=True)
    return best_r


def run_hpo(target: str, trials: int, features_path: Path, out_path: Path) -> dict[str, Any]:
    df_train = pd.read_parquet(features_path)
    skip = {"well", "id", "target"}
    feature_cols = [c for c in df_train.columns if c not in skip]
    X = df_train[feature_cols]
    y = df_train["target"]
    base = df_train["last_known_tvt"].values
    md_since = df_train["md_since"].values
    pf_oof = (df_train["pf_ancc"].values - base).astype(np.float32)
    splits = _make_splits(df_train)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.perf_counter()
    if target == "lgb":
        study.optimize(
            lambda t: objective_lgb(t, X, y, splits, feature_cols, base, md_since, pf_oof),
            n_trials=trials,
            show_progress_bar=False,
        )
    elif target == "cb":
        study.optimize(
            lambda t: objective_cb(t, X, y, splits, feature_cols, base, md_since, pf_oof),
            n_trials=trials,
            show_progress_bar=False,
        )
    else:
        raise ValueError(f"unknown target {target}")
    elapsed = time.perf_counter() - t0
    print(f"[hpo:{target}] best value={study.best_value:.4f}, trials={trials}, time={elapsed:.0f}s")
    print(f"[hpo:{target}] best params: {study.best_params}")
    result = {
        "target": target,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": trials,
        "elapsed_sec": elapsed,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["lgb", "cb"], required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(f"outputs/exp004/hpo_{args.target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    run_hpo(args.target, args.trials, Path(args.features), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
