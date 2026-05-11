"""Deepest EDA Phase 6: failure wells signature analysis.

pseudo-test-errors.parquet (Phase 5) と既存 EDA outputs を merge して、
'last_rmse 大 / formula_rmse 大' な wells の structural signature を抽出する。

実行: uv run python notebooks/_deepest_eda_failure_correlation.py
出力: outputs/eda/deepest_eda/failure-correlations.csv + signature.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
OUT_DIR = REPO / "outputs/eda/deepest_eda"


def main():
    # 1. load all EDA tables
    df_e = pd.read_parquet(OUT_DIR / "pseudo-test-errors.parquet")
    df_h = pd.read_parquet(OUT_DIR / "per-well-high-order.parquet")
    df_o = pd.read_parquet(OUT_DIR / "per-well-outlier.parquet")
    df_b = pd.read_parquet(OUT_DIR / "b-cluster-xy.parquet")
    df_t = pd.read_parquet(OUT_DIR / "tail-stats.parquet")
    df_fp = pd.read_parquet(REPO / "outputs/eda/first_principles/per-well-stats.parquet")

    # 2. merge
    df_merge = df_e[["well_id", "last_rmse", "lin_rmse", "formula_rmse", "n_hidden", "n_visible",
                       "last_max_err", "last_abs_at_500", "last_abs_at_2000", "last_abs_at_4000"]].copy()
    df_merge = df_merge.merge(df_h.drop(columns=["split"], errors="ignore"), on="well_id", how="left")
    df_merge = df_merge.merge(df_o[["well_id", "b_med", "b_std", "b_mad", "b_quartile_diff",
                                       "ncc_self_max", "ncc_self_p90"]], on="well_id", how="left")
    df_merge = df_merge.merge(df_b[["well_id", "x_med", "y_med"]], on="well_id", how="left")
    df_merge = df_merge.merge(df_t[["well_id", "tail_50_dtvt_mean", "tail_50_dtvt_std",
                                       "tail_100_slope_md", "tail_100_slope_resid_std", "tail_100_gr_std",
                                       "tail_50_dir_sign"]], on="well_id", how="left")
    df_merge = df_merge.merge(df_fp, on="well_id", how="left", suffixes=("", "_fp"))

    print(f"merged shape: {df_merge.shape}")

    # 3. Spearman correlation with last_rmse (= simple model error)
    targets = ["last_rmse", "lin_rmse", "formula_rmse"]
    skip = set(["well_id"] + targets + ["split", "n_rows"])

    cor_rows = []
    for col in df_merge.columns:
        if col in skip:
            continue
        if not np.issubdtype(df_merge[col].dtype, np.number):
            continue
        s = df_merge[[col] + targets].dropna()
        if len(s) < 50:
            continue
        for t in targets:
            try:
                # spearman
                rs = s[col].rank().corr(s[t].rank())
                cor_rows.append({"feature": col, "target": t, "spearman": rs, "n": len(s)})
            except Exception:
                pass
    df_cor = pd.DataFrame(cor_rows)
    df_cor.to_csv(OUT_DIR / "failure-correlations.csv", index=False)

    for t in targets:
        sub = df_cor[df_cor["target"] == t].sort_values("spearman", key=abs, ascending=False).head(20)
        print(f"\n=== top-20 features predicting {t} (Spearman, abs) ===")
        print(sub.to_string(index=False))

    # 4. failure cluster analysis: last_rmse > p90 wells の b cluster 分布
    p90 = df_merge["last_rmse"].quantile(0.9)
    fail = df_merge[df_merge["last_rmse"] > p90].copy()
    print(f"\nfailure wells (last_rmse > p90={p90:.2f}): {len(fail)}")
    if "b_med" in fail.columns:
        fail["b_round"] = fail["b_med"].round(2)
        bc = fail["b_round"].value_counts().head(10)
        print("=== failure wells の b_round (top 10) ===")
        print(bc)

    # 5. CRITICAL: tail_50_dtvt_mean が大 (大 dtvt = TVT が末端で急変) と last_rmse の関係
    print("\n=== last_rmse by tail_50_dtvt_mean quintile ===")
    df_merge["dtvt_quintile"] = pd.qcut(df_merge["tail_50_dtvt_mean"].abs(), 5, labels=range(5))
    print(df_merge.groupby("dtvt_quintile")["last_rmse"].describe()[["mean", "50%", "max"]])

    # 6. failure wells の trajectory signature
    print("\n=== top-30 failure wells: signatures ===")
    sigs = ["well_id", "last_rmse", "n_hidden", "n_visible", "b_med", "tail_50_dtvt_mean",
             "tail_50_dtvt_std", "visible_ratio", "ar1_phi", "ar1_eps_std", "dtvt_std", "dogleg_p95"]
    sigs = [c for c in sigs if c in df_merge.columns]
    fail_top = df_merge.sort_values("last_rmse", ascending=False).head(30)
    print(fail_top[sigs].to_string())


if __name__ == "__main__":
    main()
