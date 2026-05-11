"""Deepest EDA Phase 7: typewell GR detailed + 6 formation top depth uniqueness.

1. typewell GR signature の per-well structure (= geology label-conditioned mean, std)
2. 6 formation top depth の値域 + per-well variability (= 各 formation の per-well 値が cluster 化されるか)
3. typewell の TVT-GR pair の DTW-like overlap score with horizontal visible

実行: uv run python notebooks/_deepest_eda_typewell_formation.py
出力: outputs/eda/deepest_eda/typewell-stats.parquet, formation-uniqueness.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
OUT_DIR = REPO / "outputs/eda/deepest_eda"

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def list_wells(d: Path):
    return sorted(p.stem.replace("__horizontal_well", "") for p in d.glob("*__horizontal_well.csv"))


def typewell_stats(well_id: str) -> dict:
    tw_path = TRAIN_DIR / f"{well_id}__typewell.csv"
    if not tw_path.exists():
        return None
    tw = pd.read_csv(tw_path)
    out = dict(well_id=well_id, tw_n_rows=len(tw),
                tw_tvt_min=float(tw["TVT"].min()), tw_tvt_max=float(tw["TVT"].max()),
                tw_tvt_range=float(tw["TVT"].max() - tw["TVT"].min()),
                tw_gr_n_valid=int(tw["GR"].notna().sum()),
                tw_gr_mean=float(np.nanmean(tw["GR"])), tw_gr_std=float(np.nanstd(tw["GR"])),
                tw_gr_p5=float(np.nanpercentile(tw["GR"], 5)),
                tw_gr_p95=float(np.nanpercentile(tw["GR"], 95)))
    # geology label
    if "Geology" in tw.columns:
        geo = tw["Geology"].dropna()
        out["tw_geo_n_labeled"] = len(geo)
        out["tw_geo_n_unique"] = geo.nunique()
        # per-geology GR stats
        gr_by_geo = tw.groupby("Geology")["GR"].agg(["mean", "std", "count"])
        out["tw_geo_n_classes"] = len(gr_by_geo)
        # 多数 class の name
        if len(gr_by_geo) > 0:
            top_class = gr_by_geo["count"].idxmax()
            out["tw_geo_top_class"] = top_class
            out["tw_geo_top_count"] = int(gr_by_geo.loc[top_class, "count"])
        # 全 label の GR mean spread
        mean_spread = gr_by_geo["mean"].std()
        out["tw_geo_class_gr_mean_spread"] = float(mean_spread) if pd.notna(mean_spread) else np.nan
    else:
        out["tw_geo_n_labeled"] = 0
        out["tw_geo_n_unique"] = 0
        out["tw_geo_n_classes"] = 0
        out["tw_geo_top_class"] = None
        out["tw_geo_top_count"] = 0
        out["tw_geo_class_gr_mean_spread"] = np.nan

    # TVT step (= resolution)
    if len(tw) > 1:
        steps = np.diff(tw["TVT"].values)
        steps_pos = steps[steps > 1e-6]
        if len(steps_pos) > 0:
            out["tw_tvt_step_p50"] = float(np.nanmedian(steps_pos))
        else:
            out["tw_tvt_step_p50"] = np.nan
    else:
        out["tw_tvt_step_p50"] = np.nan

    return out


def formation_top_depth(well_id: str, src: Path) -> dict:
    """per-well の 6 formation top depth (= ground-truth) の per-well median."""
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    out = {"well_id": well_id}
    if "ANCC" not in df.columns:
        for f in FORMATIONS:
            out[f"top_{f}_med"] = np.nan
            out[f"top_{f}_std"] = np.nan
        return out
    is_vis = df["TVT_input"].notna().values
    for f in FORMATIONS:
        if f in df.columns:
            v = df.loc[is_vis, f].values if is_vis.any() else df[f].values
            out[f"top_{f}_med"] = float(np.nanmedian(v))
            out[f"top_{f}_std"] = float(np.nanstd(v))
        else:
            out[f"top_{f}_med"] = np.nan
            out[f"top_{f}_std"] = np.nan
    return out


def main():
    wells = list_wells(TRAIN_DIR)
    print(f"train wells: {len(wells)}")
    rows_tw = []
    rows_f = []
    for i, w in enumerate(wells):
        if i % 100 == 0:
            print(f"  [{i:4d}/{len(wells)}] {w}")
        r = typewell_stats(w)
        if r:
            rows_tw.append(r)
        rows_f.append(formation_top_depth(w, TRAIN_DIR))

    df_tw = pd.DataFrame(rows_tw)
    df_tw.to_parquet(OUT_DIR / "typewell-stats.parquet", index=False)
    print(f"saved typewell -> {OUT_DIR / 'typewell-stats.parquet'}")

    df_f = pd.DataFrame(rows_f)
    df_f.to_parquet(OUT_DIR / "formation-uniqueness.parquet", index=False)
    print(f"saved formation -> {OUT_DIR / 'formation-uniqueness.parquet'}")

    print("\n=== typewell stats summary ===")
    summ_tw = {}
    for col in df_tw.columns:
        if col in ("well_id", "tw_geo_top_class"):
            continue
        s = df_tw[col].dropna()
        if not np.issubdtype(s.dtype, np.number):
            continue
        summ_tw[col] = dict(n=int(len(s)), p10=float(s.quantile(0.1)),
                              p50=float(s.quantile(0.5)), p90=float(s.quantile(0.9)),
                              min=float(s.min()), max=float(s.max()))
        print(f"  {col}: p10={summ_tw[col]['p10']:.3f}, p50={summ_tw[col]['p50']:.3f}, p90={summ_tw[col]['p90']:.3f}, min={summ_tw[col]['min']:.3f}, max={summ_tw[col]['max']:.3f}")

    print("\n=== 6 formation top per-well median の unique count ===")
    for f in FORMATIONS:
        col = f"top_{f}_med"
        if col in df_f.columns:
            s = df_f[col].dropna()
            n_uniq_2 = s.round(2).nunique()
            n_uniq_4 = s.round(4).nunique()
            print(f"  {f}: n={len(s)}, unique(round 0.01)={n_uniq_2}, unique(round 0.0001)={n_uniq_4}, range=[{s.min():.2f}, {s.max():.2f}]")

    print("\n=== geology label top class distribution ===")
    if "tw_geo_top_class" in df_tw.columns:
        tc = df_tw["tw_geo_top_class"].value_counts().head(15)
        print(tc)
    print("\n=== tw_geo_n_classes distribution ===")
    if "tw_geo_n_classes" in df_tw.columns:
        nc = df_tw["tw_geo_n_classes"].value_counts().sort_index()
        print(nc)

    (OUT_DIR / "summary-typewell.json").write_text(json.dumps(summ_tw, indent=2))


if __name__ == "__main__":
    main()
