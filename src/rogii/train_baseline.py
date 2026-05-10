"""ROGII exp002 baseline — residual target + tvt_formula + LightGBM GroupKFold(5).

Public-notebook insight: target = TVT - last_known_TVT (residual). Absolute TVT
prediction (= what exp001 attempted) underperforms by ~3x on RMSE.

Reads cached features from data/processed/{train,test}_features.parquet
(produced by `python -m rogii.build_features`).

Usage:
    PYTHONUNBUFFERED=1 uv run python -m rogii.train_baseline

Outputs:
    outputs/oof/exp002_lgb.parquet
    outputs/preds/exp002_lgb_test.parquet
    submissions/exp002_lgb.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cv import make_well_folds, rmse, rmse_hidden
from .features import FEATURE_COLS

EXP = "exp002"
PROCESSED = Path("data/processed")
OUT_OOF = Path("outputs/oof")
OUT_PRED = Path("outputs/preds")
OUT_SUB = Path("submissions")
OUT_LOG = Path("outputs/logs")

# Lightweight params for fast first submission (Kaggler standard "quick baseline").
# Refs: NVIDIA Grandmaster Playbook 2024 / kagglib.baselines.
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
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "num_threads": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100
LOG_PERIOD = 50


def main() -> None:
    OUT_OOF.mkdir(parents=True, exist_ok=True)
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    OUT_LOG.mkdir(parents=True, exist_ok=True)

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

    # Residual target: TVT - last_known_TVT
    last_known_train = train["last_known_TVT"].to_numpy(np.float32)
    y_residual = (train["TVT"].to_numpy(np.float32) - last_known_train).astype(np.float32)
    train_hidden = train["TVT_input"].isna().to_numpy()
    well_ids = train["well"].to_numpy()

    last_known_test = test["last_known_TVT"].to_numpy(np.float32)
    test_hidden = test["TVT_input"].isna().to_numpy()

    X_train = train[FEATURE_COLS].astype(np.float32)
    X_test = test[FEATURE_COLS].astype(np.float32)

    print(
        f"   features: {len(FEATURE_COLS)}  |  "
        f"train residual range: [{y_residual.min():.2f}, {y_residual.max():.2f}] "
        f"std={y_residual.std():.2f}",
        flush=True,
    )
    print(
        f"   train hidden rows: {int(train_hidden.sum()):,} / {len(X_train):,} "
        f"({train_hidden.mean():.1%})  |  test hidden: {int(test_hidden.sum()):,}",
        flush=True,
    )

    print("\n==> 5-fold GroupKFold by well", flush=True)
    folds = make_well_folds(well_ids, n_splits=5, seed=42)

    oof_residual = np.zeros(len(X_train), dtype=np.float32)
    test_residual_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_iters: list[int] = []
    fold_rmses: list[float] = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        t_fold = time.perf_counter()
        print(f"\n[fold {k + 1}/5]  train rows={len(tr_idx):,}  val rows={len(va_idx):,}", flush=True)
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
                lgb.log_evaluation(LOG_PERIOD),
            ],
        )
        oof_residual[va_idx] = booster.predict(
            X_train.iloc[va_idx], num_iteration=booster.best_iteration
        )
        test_residual_pred += booster.predict(X_test, num_iteration=booster.best_iteration) / len(
            folds
        )

        fold_iters.append(booster.best_iteration)
        # Convert residual back to absolute TVT for hidden-row RMSE (= LB-aligned)
        oof_tvt_va = last_known_train[va_idx] + oof_residual[va_idx]
        true_tvt_va = train["TVT"].iloc[va_idx].to_numpy(np.float32)
        rmse_resid = rmse(y_residual[va_idx], oof_residual[va_idx])
        rmse_tvt_all = rmse(true_tvt_va, oof_tvt_va)
        rmse_tvt_h = rmse_hidden(true_tvt_va, oof_tvt_va, train_hidden[va_idx])
        fold_rmses.append(rmse_tvt_h)
        print(
            f"   best_iter={booster.best_iteration:5d}  "
            f"residual RMSE={rmse_resid:.4f}  "
            f"TVT RMSE all={rmse_tvt_all:.4f}  hidden={rmse_tvt_h:.4f}  "
            f"({time.perf_counter() - t_fold:.1f}s)",
            flush=True,
        )

    oof_tvt = last_known_train + oof_residual
    cv_residual = rmse(y_residual, oof_residual)
    cv_tvt_all = rmse(train["TVT"].to_numpy(np.float32), oof_tvt)
    cv_tvt_hidden = rmse_hidden(
        train["TVT"].to_numpy(np.float32), oof_tvt, train_hidden
    )
    print(
        f"\n==> CV residual RMSE       = {cv_residual:.4f}\n"
        f"==> CV TVT RMSE all rows   = {cv_tvt_all:.4f}\n"
        f"==> CV TVT RMSE hidden     = {cv_tvt_hidden:.4f}    <-- LB-aligned\n"
        f"==> mean best_iter         = {int(np.mean(fold_iters))}",
        flush=True,
    )

    pd.DataFrame(
        {
            "well": train["well"],
            "row_idx": train["row_idx"].astype(np.int32),
            "tvt_pred": oof_tvt,
            "residual_pred": oof_residual,
            "TVT": train["TVT"].to_numpy(np.float32),
            "is_hidden": train_hidden,
        }
    ).to_parquet(OUT_OOF / f"{EXP}_lgb.parquet", index=False)

    test_tvt = last_known_test + test_residual_pred
    pd.DataFrame(
        {
            "well": test["well"],
            "row_idx": test["row_idx"].astype(np.int32),
            "tvt_pred": test_tvt,
            "residual_pred": test_residual_pred,
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
            "tvt": test_tvt[test_hidden],
        }
    )
    out_csv = OUT_SUB / f"{EXP}_lgb.csv"
    sub.to_csv(out_csv, index=False)
    print(
        f"\n==> wrote submission {out_csv} ({len(sub):,} rows, "
        f"CV TVT hidden = {cv_tvt_hidden:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
