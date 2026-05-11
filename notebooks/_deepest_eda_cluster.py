"""Deepest EDA Phase 3: b_well cluster + XY centroid との関係.

b_med が 66 unique values しかない衝撃発見の深掘り:
- 各 cluster は同一 (X, Y) 領域に属するか? (= geological zone)
- cluster は formation 構造と対応するか? (= 同 cluster で 6 formation 値が一致するか)
- test 3 wells はどの cluster の最寄りか?

実行: uv run python notebooks/_deepest_eda_cluster.py
出力: outputs/eda/deepest_eda/b-cluster-xy.parquet + summary-cluster.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
TEST_DIR = REPO / "data/raw/test"
OUT_DIR = REPO / "outputs/eda/deepest_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def list_wells(d: Path):
    return sorted(p.stem.replace("__horizontal_well", "") for p in d.glob("*__horizontal_well.csv"))


def per_well_xy_b(well_id: str, src: Path, split: str) -> dict:
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    is_vis = df["TVT_input"].notna().values

    out = dict(
        well_id=well_id,
        split=split,
        x_med=float(df["X"].median()),
        y_med=float(df["Y"].median()),
        z_med=float(df["Z"].median()),
        x_first=float(df["X"].iloc[0]),
        y_first=float(df["Y"].iloc[0]),
        z_first=float(df["Z"].iloc[0]),
    )

    # b_med, b_med per formation
    if "TVT" in df.columns and is_vis.sum() >= 30:
        tvt = df["TVT"].values
        z = df["Z"].values
        b_per_form = {}
        for f in FORMATIONS:
            if f in df.columns:
                b_f = tvt + z - df[f].values
                b_v = b_f[is_vis]
                b_per_form[f] = float(np.nanmedian(b_v))
            else:
                b_per_form[f] = np.nan
        for f, v in b_per_form.items():
            out[f"b_{f}"] = v
        # b_med using ANCC
        out["b_ANCC_med"] = b_per_form.get("ANCC", np.nan)
    else:
        for f in FORMATIONS:
            out[f"b_{f}"] = np.nan
        out["b_ANCC_med"] = np.nan

    return out


def main():
    train_wells = list_wells(TRAIN_DIR)
    test_wells = list_wells(TEST_DIR)
    print(f"train wells: {len(train_wells)}, test wells: {len(test_wells)}")

    rows = []
    for i, w in enumerate(train_wells):
        if i % 100 == 0:
            print(f"  train [{i:4d}/{len(train_wells)}] {w}")
        rows.append(per_well_xy_b(w, TRAIN_DIR, "train"))
    for w in test_wells:
        rows.append(per_well_xy_b(w, TEST_DIR, "test"))

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "b-cluster-xy.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}")

    # cluster analysis: b_ANCC_med の cluster 1 つあたりの XY centroid 分散
    df_t = df[df["split"] == "train"].dropna(subset=["b_ANCC_med"]).copy()
    df_t["b_round"] = df_t["b_ANCC_med"].round(2)
    print("\n=== cluster size + XY spread ===")
    cluster_info = []
    for b_val, g in df_t.groupby("b_round"):
        if len(g) < 3:
            continue
        xy = g[["x_med", "y_med"]].values
        # XY range / std per cluster
        x_std = xy[:, 0].std()
        y_std = xy[:, 1].std()
        x_range = xy[:, 0].max() - xy[:, 0].min()
        y_range = xy[:, 1].max() - xy[:, 1].min()
        # XY centroid distance to global mean
        x_mean = xy[:, 0].mean()
        y_mean = xy[:, 1].mean()
        # formation の一致度 (= same cluster で 6 formation 値の per-cluster std)
        form_stds = []
        for f in FORMATIONS:
            col = f"b_{f}"
            if col in g.columns:
                s = g[col].dropna()
                if len(s) >= 3:
                    form_stds.append(s.std())
        avg_form_std = np.nanmean(form_stds) if form_stds else np.nan
        cluster_info.append({
            "b_round": b_val,
            "n_wells": int(len(g)),
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": x_std,
            "y_std": y_std,
            "x_range": x_range,
            "y_range": y_range,
            "xy_diag_range": float(np.sqrt(x_range**2 + y_range**2)),
            "avg_form_std": avg_form_std,
        })
    df_cluster = pd.DataFrame(cluster_info).sort_values("n_wells", ascending=False)
    print(df_cluster.head(20).to_string())
    print(f"\ncluster count (>=3 wells): {len(df_cluster)}")
    print(f"median xy_diag_range across clusters: {df_cluster['xy_diag_range'].median():.0f} ft")
    print(f"median avg_form_std: {df_cluster['avg_form_std'].median():.4f} ft (= 同 cluster 内 6 formation b 値の std)")

    df_cluster.to_parquet(OUT_DIR / "cluster-xy-summary.parquet", index=False)
    print(f"saved -> {OUT_DIR / 'cluster-xy-summary.parquet'}")

    # test wells: 最寄り train wells (XY 距離) の b_ANCC_med
    print("\n=== test 3 wells: 最寄り 5 train wells (XY 距離) ===")
    df_test = df[df["split"] == "test"].copy()
    df_train_b = df[df["split"] == "train"].dropna(subset=["b_ANCC_med"])[["well_id", "x_med", "y_med", "b_ANCC_med"]].copy()

    test_summary = []
    for _, t in df_test.iterrows():
        dist = np.sqrt((df_train_b["x_med"] - t["x_med"]) ** 2 + (df_train_b["y_med"] - t["y_med"]) ** 2)
        nearest = df_train_b.iloc[dist.values.argsort()[:5]].copy()
        nearest["dist_to_test"] = dist.values[dist.values.argsort()[:5]]
        print(f"\ntest {t['well_id']} (X={t['x_med']:.0f}, Y={t['y_med']:.0f}):")
        print(nearest[["well_id", "x_med", "y_med", "b_ANCC_med", "dist_to_test"]].to_string(index=False))
        test_summary.append({
            "test_well": t["well_id"],
            "test_x": t["x_med"],
            "test_y": t["y_med"],
            "nearest_b_med": float(nearest["b_ANCC_med"].iloc[0]),
            "nearest_dist": float(nearest["dist_to_test"].iloc[0]),
            "top5_b_med_mean": float(nearest["b_ANCC_med"].mean()),
            "top5_b_med_std": float(nearest["b_ANCC_med"].std()),
        })

    summary = dict(
        n_clusters_ge3=int(len(df_cluster)),
        median_xy_diag_range=float(df_cluster["xy_diag_range"].median()),
        median_avg_form_std=float(df_cluster["avg_form_std"].median()),
        test_wells=test_summary,
    )
    (OUT_DIR / "summary-cluster.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
