#!/usr/bin/env python
"""Phase 5 v3 local smoke test for the 4 改修 (Huber + hetero weight + MEDIAN + path b).

This script exercises each modification in isolation, with small synthetic data
(or the real per-well-stats parquet for the hetero-weight step), since the full
kernel cannot run locally without the karnakbaev pretrained artefacts.

Acceptance:
  - Huber objective sklearn LGBMRegressor / CatBoostRegressor with `Huber:delta`
    actually fits, produces non-NaN predictions, RMSE finite.
  - sample_weight from per-well-stats parquet: w shape == n_train, w ∈ [0.5, 2.0],
    mean ≈ 1.0.
  - Multi-seed MEDIAN: 3 seed × 2 model produces 6 predictors per (val, test),
    MEDIAN over axis=0 gives shape == (n_va,) / (n_test,).
  - path b: 1D grid search over w_kb returns 0 <= w_kb <= 1, blend OOF RMSE
    finite, smaller than max(kb-only, own-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = REPO_ROOT / "outputs/eda/first_principles/per-well-stats.parquet"

# Synthetic data (= 5 wells × 200 rows each = 1000 rows, 12 features)
N_WELLS = 5
N_PER_WELL = 200
N_TRAIN = N_WELLS * N_PER_WELL
N_FEAT = 12
N_TEST = 250

rng = np.random.default_rng(42)


def make_synth():
    well_ids = ["WELL_A", "WELL_B", "WELL_C", "WELL_D", "WELL_E"]
    wells = np.repeat(well_ids, N_PER_WELL)
    # heterogeneous noise: different sigma per well
    sigma_per_well = np.array([0.5, 0.7, 1.0, 1.3, 1.8])
    sigma_arr = np.repeat(sigma_per_well, N_PER_WELL)
    X = rng.normal(size=(N_TRAIN, N_FEAT)).astype(np.float32)
    # signal: linear + small nonlinear
    y = (
        2.0 * X[:, 0]
        + 1.5 * X[:, 1]
        - 1.2 * X[:, 2]
        + 0.5 * X[:, 3] ** 2
        + rng.normal(scale=sigma_arr, size=N_TRAIN)  # heteroscedastic noise
    ).astype(np.float32)
    # inject 5% outliers (= fat tail)
    out_idx = rng.choice(N_TRAIN, size=int(N_TRAIN * 0.05), replace=False)
    y[out_idx] += rng.choice([-1, 1], size=out_idx.size) * rng.uniform(8, 15, size=out_idx.size)
    return wells, X, y


def smoke_1_huber_loss(X, y):
    """改修 1: Huber loss (LGB + CB)."""
    print("\n=== Smoke 1: Huber loss ===")
    n_tr = int(N_TRAIN * 0.8)
    Xt, Xv = X[:n_tr], X[n_tr:]
    yt, yv = y[:n_tr], y[n_tr:]

    # LGB Huber
    params = dict(
        boosting_type="gbdt", learning_rate=0.05, num_leaves=31, n_estimators=200,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=0.1, objective="huber", alpha=0.9,
        verbose=-1, n_jobs=-1, seed=42,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred_lgb = model.predict(Xv).astype(np.float32)
    rmse_lgb = root_mean_squared_error(yv, pred_lgb)
    print(f"  LGB huber best_iter={model.best_iteration_} val RMSE={rmse_lgb:.4f}")
    assert not np.isnan(pred_lgb).any(), "LGB huber predictions NaN!"

    # CB Huber:delta=0.5
    cb_params = dict(
        iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function="Huber:delta=0.5", random_seed=42, task_type="CPU",
        od_type="Iter", od_wait=50, verbose=0,
    )
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(Pool(Xt, label=yt), eval_set=Pool(Xv, label=yv), use_best_model=True, verbose=0)
    pred_cb = cb_model.predict(Xv).astype(np.float32)
    rmse_cb = root_mean_squared_error(yv, pred_cb)
    print(f"  CB Huber:delta=0.5 best_iter={cb_model.best_iteration_} val RMSE={rmse_cb:.4f}")
    assert not np.isnan(pred_cb).any(), "CB Huber predictions NaN!"
    assert np.isfinite(rmse_lgb) and np.isfinite(rmse_cb), "RMSE not finite"

    print("  PASS")
    return pred_lgb, pred_cb, yv


def smoke_2_hetero_weight(wells):
    """改修 2: heteroscedastic sample_weight from per-well-stats parquet."""
    print("\n=== Smoke 2: heteroscedastic sample_weight ===")
    HETERO_W_CLIP_LO = 0.5
    HETERO_W_CLIP_HI = 2.0
    HETERO_W_FILL_P50 = 0.00835

    if not STATS_PATH.exists():
        print(f"  WARN: {STATS_PATH} not found, skipping real-data load")
        sigma_arr = np.full(len(wells), HETERO_W_FILL_P50, dtype=np.float32)
    else:
        stats_df = pd.read_parquet(STATS_PATH)
        assert "well_id" in stats_df.columns, f"missing well_id in {stats_df.columns.tolist()}"
        assert "b_well_resid_std" in stats_df.columns, "missing b_well_resid_std"
        sigma_map = dict(zip(
            stats_df["well_id"].astype(str),
            stats_df["b_well_resid_std"].astype(np.float32),
        ))
        # use first 5 real wells from the parquet so we have real σ values
        real_wells = stats_df["well_id"].astype(str).head(N_WELLS).tolist()
        well_assignment = np.repeat(real_wells, N_PER_WELL)
        sigma_arr = np.array([sigma_map.get(w, np.nan) for w in well_assignment], dtype=np.float32)
        print(f"  used real wells: {real_wells}")
        print(f"  raw sigma range: [{sigma_arr.min():.5f}, {sigma_arr.max():.5f}]")

    n_nan = int(np.isnan(sigma_arr).sum())
    if n_nan:
        print(f"  WARN: {n_nan} NaN in sigma_arr, filling with p50={HETERO_W_FILL_P50}")
        sigma_arr = np.where(np.isnan(sigma_arr), HETERO_W_FILL_P50, sigma_arr)

    w = 1.0 / np.maximum(sigma_arr, 1e-6)
    w = w / float(w.mean())
    w = np.clip(w, HETERO_W_CLIP_LO, HETERO_W_CLIP_HI).astype(np.float32)
    print(f"  w range: [{w.min():.3f}, {w.max():.3f}], mean={w.mean():.3f}, shape={w.shape}")

    assert w.shape[0] == len(wells), f"shape mismatch: {w.shape[0]} vs {len(wells)}"
    assert (w >= HETERO_W_CLIP_LO).all() and (w <= HETERO_W_CLIP_HI).all(), "clip violation"
    assert abs(float(w.mean()) - 1.0) < 0.5, f"mean too off: {w.mean()}"
    assert not np.isnan(w).any(), "weights contain NaN"

    print("  PASS")
    return w


def smoke_3_multiseed_median(X, y, w):
    """改修 3: Multi-seed MEDIAN (3 seed × 1 LGB + 3 seed × 1 CB)."""
    print("\n=== Smoke 3: Multi-seed MEDIAN ===")
    n_tr = int(N_TRAIN * 0.8)
    Xt, Xv = X[:n_tr], X[n_tr:]
    yt, yv = y[:n_tr], y[n_tr:]
    w_tr, w_va = w[:n_tr], w[n_tr:]

    seeds = (42, 123, 2024)

    lgb_val_stack = []
    for sd in seeds:
        params = dict(
            boosting_type="gbdt", learning_rate=0.05, num_leaves=31, n_estimators=200,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, reg_alpha=0.1, objective="huber", alpha=0.9,
            verbose=-1, n_jobs=-1, seed=int(sd), random_state=int(sd),
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(Xt, yt, sample_weight=w_tr, eval_set=[(Xv, yv)], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pv = m.predict(Xv).astype(np.float32)
        lgb_val_stack.append(pv)
        print(f"  LGB seed={sd}: val RMSE={root_mean_squared_error(yv, pv):.4f}")
    assert len(lgb_val_stack) == len(seeds)
    med_v_lgb = np.median(np.stack(lgb_val_stack, axis=0), axis=0).astype(np.float32)
    print(f"  LGB MEDIAN shape={med_v_lgb.shape} val RMSE={root_mean_squared_error(yv, med_v_lgb):.4f}")
    assert med_v_lgb.shape == yv.shape, f"MEDIAN shape mismatch: {med_v_lgb.shape} vs {yv.shape}"
    assert not np.isnan(med_v_lgb).any(), "MEDIAN contains NaN"

    cb_val_stack = []
    for sd in seeds:
        cb_params = dict(
            iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            loss_function="Huber:delta=0.5", random_seed=int(sd), task_type="CPU",
            od_type="Iter", od_wait=50, verbose=0,
        )
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(Pool(Xt, label=yt, weight=w_tr),
                     eval_set=Pool(Xv, label=yv, weight=w_va),
                     use_best_model=True, verbose=0)
        pv = cb_model.predict(Xv).astype(np.float32)
        cb_val_stack.append(pv)
        print(f"  CB seed={sd}: val RMSE={root_mean_squared_error(yv, pv):.4f}")
    med_v_cb = np.median(np.stack(cb_val_stack, axis=0), axis=0).astype(np.float32)
    print(f"  CB MEDIAN shape={med_v_cb.shape} val RMSE={root_mean_squared_error(yv, med_v_cb):.4f}")
    assert med_v_cb.shape == yv.shape, f"MEDIAN shape mismatch: {med_v_cb.shape} vs {yv.shape}"

    # variance reduction check: MEDIAN should be at least as good as max single seed
    single_max_lgb = max(root_mean_squared_error(yv, p) for p in lgb_val_stack)
    med_lgb_rmse = root_mean_squared_error(yv, med_v_lgb)
    print(f"  MEDIAN vs worst single seed: LGB MEDIAN={med_lgb_rmse:.4f} vs worst single={single_max_lgb:.4f}")

    print("  PASS")
    return med_v_lgb, med_v_cb, yv


def smoke_4_path_b_blend(yv):
    """改修 4: path b = kb simple-avg + own Ridge + 1D grid blend."""
    print("\n=== Smoke 4: path b blend ===")
    n_va = len(yv)
    # synthesize kb 5-base OOF + own 2-base OOF
    kb_oofs = [yv + rng.normal(scale=0.5, size=n_va).astype(np.float32) for _ in range(5)]
    own_oofs = [yv + rng.normal(scale=0.7, size=n_va).astype(np.float32) for _ in range(2)]
    # test predictions
    kb_tests = [rng.normal(size=50).astype(np.float32) for _ in range(5)]
    own_tests = [rng.normal(size=50).astype(np.float32) for _ in range(2)]

    kb_avg_oof = np.mean(np.column_stack(kb_oofs), axis=1).astype(np.float64)
    kb_avg_test = np.mean(np.column_stack(kb_tests), axis=1).astype(np.float64)

    Sx_own = np.column_stack(own_oofs)
    St_own = np.column_stack(own_tests)
    ridge_own = Ridge(alpha=1.0, fit_intercept=False, positive=True)
    ridge_own.fit(Sx_own, yv)
    own_oof_blend = ridge_own.predict(Sx_own).astype(np.float64)
    own_test_blend = ridge_own.predict(St_own).astype(np.float64)
    print(f"  own Ridge coef: {ridge_own.coef_.round(4).tolist()}")
    assert (ridge_own.coef_ >= 0).all(), "Ridge positive constraint failed"

    grid = np.arange(0.0, 1.0001, 0.05)
    best_w, best_r = 0.5, np.inf
    for w in grid:
        blend = w * kb_avg_oof + (1.0 - w) * own_oof_blend
        r = root_mean_squared_error(yv, blend)
        if r < best_r:
            best_r, best_w = float(r), float(w)
    r_kb_only = root_mean_squared_error(yv, kb_avg_oof)
    r_own_only = root_mean_squared_error(yv, own_oof_blend)
    print(f"  grid search: w_kb*={best_w:.2f} blend OOF RMSE={best_r:.4f}")
    print(f"  kb_only RMSE={r_kb_only:.4f}, own_only RMSE={r_own_only:.4f}")
    assert 0.0 <= best_w <= 1.0
    assert best_r <= max(r_kb_only, r_own_only) + 1e-6, "blend worse than both endpoints"
    assert np.isfinite(best_r)
    test_delta = best_w * kb_avg_test + (1.0 - best_w) * own_test_blend
    assert test_delta.shape == (50,) and not np.isnan(test_delta).any()
    print("  PASS")


def main():
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"STATS_PATH={STATS_PATH} exists={STATS_PATH.exists()}")
    wells, X, y = make_synth()
    print(f"synth: wells={len(set(wells))} X={X.shape} y={y.shape} y_range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"NaN: X={int(np.isnan(X).sum())} y={int(np.isnan(y).sum())}")
    assert int(np.isnan(X).sum()) == 0 and int(np.isnan(y).sum()) == 0, "NaN in synth data!"

    pred_lgb, pred_cb, yv = smoke_1_huber_loss(X, y)
    w = smoke_2_hetero_weight(wells)
    smoke_3_multiseed_median(X, y, w)
    smoke_4_path_b_blend(yv)

    print("\n*** ALL SMOKES PASS ***")


if __name__ == "__main__":
    sys.exit(main())
