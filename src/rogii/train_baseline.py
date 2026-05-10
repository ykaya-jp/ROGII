"""ROGII exp001 baseline — LightGBM 5-fold GroupKFold by well, eval RMSE on hidden rows.

Reads cached features from data/processed/{train,test}_features.parquet.
Run `python -m rogii.build_features` first if cache missing.

Usage:
    PYTHONUNBUFFERED=1 uv run python -m rogii.train_baseline

Outputs:
    outputs/oof/exp001_lgb.parquet
    outputs/preds/exp001_lgb_test.parquet
    submissions/exp001_lgb.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cv import make_well_folds, rmse, rmse_hidden
from .features import FEATURE_COLS

EXP = "exp001"
PROCESSED = Path("data/processed")
OUT_OOF = Path("outputs/oof")
OUT_PRED = Path("outputs/preds")
OUT_SUB = Path("submissions")

# Lightweight params for fast first submission. Tuning happens in exp002 (Optuna).
# Refs:
#   - num_leaves 63 / lr 0.1 / 2000 rounds is the standard "fast first baseline"
#     (NVIDIA Kaggle Grandmaster Playbook 2024).
#   - early_stopping 100 is enough at lr 0.1; tighten in tuning.
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 256,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.0,
    "verbosity": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "num_threads": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100
LOG_PERIOD = 100  # log val every N rounds for visibility


def main() -> None:
    OUT_OOF.mkdir(parents=True, exist_ok=True)
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_SUB.mkdir(parents=True, exist_ok=True)

    print("==> loading cached features", flush=True)
    t0 = time.perf_counter()
    train = pd.read_parquet(PROCESSED / "train_features.parquet")
    test = pd.read_parquet(PROCESSED / "test_features.parquet")
    print(
        f"   train: {len(train):,} rows / {train['well'].nunique()} wells "
        f"|  test: {len(test):,} rows / {test['well'].nunique()} wells "
        f"|  ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    X_train = train[FEATURE_COLS].astype(np.float32)
    y_train = train["TVT"].to_numpy(np.float32)
    train_hidden = train["TVT_input"].isna().to_numpy()
    well_ids = train["well"].to_numpy()

    X_test = test[FEATURE_COLS].astype(np.float32)
    test_hidden = test["TVT_input"].isna().to_numpy()

    print(
        f"   features: {len(FEATURE_COLS)}  |  "
        f"train hidden rows: {int(train_hidden.sum()):,} / {len(X_train):,} "
        f"({train_hidden.mean():.1%})",
        flush=True,
    )

    print("==> 5-fold GroupKFold by well", flush=True)
    folds = make_well_folds(well_ids, n_splits=5, seed=42)

    oof = np.zeros(len(X_train), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    fold_iters: list[int] = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        t_fold = time.perf_counter()
        print(f"\n[fold {k + 1}/5]  train rows={len(tr_idx):,}  val rows={len(va_idx):,}", flush=True)
        ds_tr = lgb.Dataset(X_train.iloc[tr_idx], y_train[tr_idx], free_raw_data=False)
        ds_va = lgb.Dataset(X_train.iloc[va_idx], y_train[va_idx], reference=ds_tr, free_raw_data=False)
        booster = lgb.train(
            LGB_PARAMS,
            ds_tr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_va],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(LOG_PERIOD),
            ],
        )
        oof[va_idx] = booster.predict(X_train.iloc[va_idx], num_iteration=booster.best_iteration)
        test_preds += booster.predict(X_test, num_iteration=booster.best_iteration) / len(folds)

        fold_iters.append(booster.best_iteration)
        rmse_all = rmse(y_train[va_idx], oof[va_idx])
        rmse_h = rmse_hidden(y_train[va_idx], oof[va_idx], train_hidden[va_idx])
        print(
            f"   best_iter={booster.best_iteration:5d}  "
            f"val RMSE all={rmse_all:.4f}  hidden={rmse_h:.4f}  "
            f"({time.perf_counter() - t_fold:.1f}s)",
            flush=True,
        )

    cv_all = rmse(y_train, oof)
    cv_hidden = rmse_hidden(y_train, oof, train_hidden)
    print(
        f"\n==> OOF CV all rows    = {cv_all:.4f}\n"
        f"==> OOF CV hidden only = {cv_hidden:.4f}    <-- LB-aligned\n"
        f"==> mean best_iter     = {int(np.mean(fold_iters))}",
        flush=True,
    )

    pd.DataFrame(
        {
            "well": train["well"],
            "row_idx": train["row_idx"].astype(np.int32),
            "tvt_pred": oof,
            "TVT": y_train,
            "is_hidden": train_hidden,
        }
    ).to_parquet(OUT_OOF / f"{EXP}_lgb.parquet", index=False)

    pd.DataFrame(
        {
            "well": test["well"],
            "row_idx": test["row_idx"].astype(np.int32),
            "tvt_pred": test_preds,
            "is_hidden": test_hidden,
        }
    ).to_parquet(OUT_PRED / f"{EXP}_lgb_test.parquet", index=False)

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
            "tvt": test_preds[test_hidden],
        }
    )
    out_csv = OUT_SUB / f"{EXP}_lgb.csv"
    sub.to_csv(out_csv, index=False)
    print(
        f"\n==> wrote submission {out_csv} ({len(sub):,} rows, CV hidden = {cv_hidden:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
