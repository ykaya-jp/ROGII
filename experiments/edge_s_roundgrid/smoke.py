"""Edge S smoke test.

Goal: Edge S の round-to-grid logic を 3 kernel に inject した後で、
build_submission() の round 部分のみを抽出してローカル検証する。

approach:
  1. test wells から 3 wells 分の id を抽出 (= 軽量化)
  2. dummy delta_pred を generate (= 真の dTVT の構造を模倣)
  3. round 前後の diff / unique count を測定
  4. failure modes (= F1/F2/F3) のサインを観察
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger("edge_s_smoke")

DATA_DIR = Path("data/raw")


def build_submission_with_edge_s(
    test_df:     pd.DataFrame,
    delta_pred:  np.ndarray,
    sample_sub:  pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """3 kernel に inject した logic と完全に同一の build_submission。"""
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
    EDGE_S_ROUND_DECIMALS = 2  # 0.01 ft grid
    if EDGE_S_ROUND_DECIMALS is not None:
        final_tvt_pre = sub["tvt"].astype(float).values.copy()
        sub["tvt"] = np.round(sub["tvt"].astype(float).values, EDGE_S_ROUND_DECIMALS)
        diff = sub["tvt"].values - final_tvt_pre
        log.info(f"[Edge S] applied round-to-{EDGE_S_ROUND_DECIMALS}-decimals")
        log.info(f"[Edge S]   diff abs mean: {np.abs(diff).mean():.6f}")
        log.info(f"[Edge S]   diff abs max:  {np.abs(diff).max():.6f}")
        log.info(f"[Edge S]   unique tvt count: {len(np.unique(sub['tvt']))}")
        log.info(f"[Edge S]   tvt sample (5): {sub['tvt'].head(5).tolist()}")

        # Return diagnostics for assertion in main
        return sub, {
            "diff_abs_mean": float(np.abs(diff).mean()),
            "diff_abs_max":  float(np.abs(diff).max()),
            "unique_count_pre":  len(np.unique(final_tvt_pre)),
            "unique_count_post": len(np.unique(sub["tvt"])),
        }

    sub[["id", "tvt"]].to_csv(output_path, index=False)
    return sub, {}


def main() -> None:
    # ── 1. Load sample submission + select 3 wells ───────────────────────
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample_sub["well"] = sample_sub["id"].str.split("_").str[0]
    log.info(f"Sample sub: {len(sample_sub):,} rows, {sample_sub['well'].nunique():,} wells")

    # 3 wells のみ (= smoke 軽量化)
    sample_wells = sample_sub["well"].unique()[:3]
    log.info(f"Smoke wells: {list(sample_wells)}")
    sub3 = sample_sub[sample_sub["well"].isin(sample_wells)].drop(columns="well").reset_index(drop=True)
    log.info(f"  rows: {len(sub3):,}")

    # ── 2. Build dummy test_df (= ids + last_known_tvt) ──────────────────
    rng = np.random.default_rng(2026)
    test_df = pd.DataFrame({
        "id":            sub3["id"].values,
        "last_known_tvt": rng.uniform(5000.0, 7000.0, size=len(sub3)),
    })
    # delta_pred = continuous prediction (= 非 grid)、典型的な ML 出力
    delta_pred = rng.normal(0.0, 30.0, size=len(test_df)).astype(np.float32)
    log.info(f"Dummy delta_pred: min={delta_pred.min():.4f} max={delta_pred.max():.4f} std={delta_pred.std():.4f}")

    # ── 3. Run build_submission_with_edge_s ──────────────────────────────
    out_path = Path("experiments/edge_s_roundgrid/smoke_submission.csv")
    sub, diag = build_submission_with_edge_s(test_df, delta_pred, sub3, out_path)

    # ── 4. Assertions (= failure mode checks) ────────────────────────────
    log.info("─" * 60)
    log.info("Edge S smoke diagnostics:")
    for k, v in diag.items():
        log.info(f"  {k}: {v}")

    # F1: diff abs max は理論上 0.005 ft 以下のはず (= round の最大誤差)
    assert diag["diff_abs_max"] <= 0.005 + 1e-9, f"F1 FAIL: diff_abs_max = {diag['diff_abs_max']}"
    log.info("[F1 check] PASS: diff abs max <= 0.005 ft")

    # F2: dtype が float64 で round 後の sub に NaN/inf 無し
    assert sub["tvt"].dtype == np.float64, f"F2 FAIL: dtype = {sub['tvt'].dtype}"
    assert not sub["tvt"].isna().any(), "F2 FAIL: NaN present"
    assert np.isfinite(sub["tvt"].values).all(), "F2 FAIL: inf present"
    log.info("[F2 check] PASS: dtype=float64, no NaN, no inf")

    # F3: round 後の unique count が round 前より減少
    assert diag["unique_count_post"] <= diag["unique_count_pre"], "F3 FAIL: round did not reduce uniqueness"
    log.info(f"[F3 check] PASS: unique count {diag['unique_count_pre']} → {diag['unique_count_post']}")

    # ── 5. Verify round actually snaps to 0.01 grid ──────────────────────
    # 全 row の tvt * 100 が int に snap しているか確認
    tvt_scaled = sub["tvt"].values * 100.0
    snap_error = np.abs(tvt_scaled - np.round(tvt_scaled))
    log.info(f"Snap to 0.01 grid: max residual = {snap_error.max():.8f} (= < 1e-9 のはず)")
    assert snap_error.max() < 1e-6, f"Snap FAIL: max residual = {snap_error.max()}"
    log.info("[Grid snap check] PASS: all values on 0.01 grid")

    log.info("─" * 60)
    log.info("Edge S smoke ALL PASS ✓")
    log.info(f"Diagnostics: {diag}")


if __name__ == "__main__":
    main()
