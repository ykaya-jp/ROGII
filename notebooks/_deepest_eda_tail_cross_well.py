"""Deepest EDA Phase 4: tail-K visible 統計 + 末端 dTVT distribution + cross-well GR similarity.

LB top kernel が使う「visible 末端を hidden の prior にする」アプローチを extends:
1. visible 末端 tail-K (= 50/100/200 行) の dTVT 分布 (= hidden 直後の continuation prediction quality)
2. 末端 dTVT mean / sign / std で wells を分類 (= 上昇/下降 trajectory)
3. cross-well GR similarity: 各 well の GR profile (visible 部) を 50-dim PCA → KMeans cluster
4. typewell-horizontal GR amplitude scale (= scale_best) と b_med cluster の関係

実行: uv run python notebooks/_deepest_eda_tail_cross_well.py
出力: outputs/eda/deepest_eda/tail-stats.parquet + gr-pca.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
TEST_DIR = REPO / "data/raw/test"
OUT_DIR = REPO / "outputs/eda/deepest_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_wells(d: Path):
    return sorted(p.stem.replace("__horizontal_well", "") for p in d.glob("*__horizontal_well.csv"))


def tail_stats(well_id: str, src: Path, split: str) -> dict:
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    is_vis = df["TVT_input"].notna().values
    if not is_vis.any():
        return dict(well_id=well_id, split=split)
    boundary = int(np.where(is_vis)[0].max())
    out = dict(well_id=well_id, split=split, boundary_idx=boundary, n_rows=len(df), n_visible=int(is_vis.sum()))

    # visible TVT_input series
    tvt_v = df.loc[is_vis, "TVT_input"].values

    for K in (20, 50, 100, 200):
        if len(tvt_v) >= K + 1:
            tail = tvt_v[-K:]
            d = np.diff(tail)
            out[f"tail_{K}_dtvt_mean"] = float(np.mean(d))
            out[f"tail_{K}_dtvt_std"] = float(np.std(d))
            out[f"tail_{K}_dtvt_p10"] = float(np.percentile(d, 10))
            out[f"tail_{K}_dtvt_p90"] = float(np.percentile(d, 90))
            # 末端 K の TVT 範囲 (= 末端 trajectory)
            out[f"tail_{K}_tvt_range"] = float(tail.max() - tail.min())
            # 末端の方向性 (= last - first sign)
            out[f"tail_{K}_dir_sign"] = float(np.sign(tail[-1] - tail[0]))
        else:
            for s in ["dtvt_mean", "dtvt_std", "dtvt_p10", "dtvt_p90", "tvt_range", "dir_sign"]:
                out[f"tail_{K}_{s}"] = np.nan

    # 末端 100 ft の slope (linear fit y=ax+b on TVT_input vs MD)
    md_v = df.loc[is_vis, "MD"].values
    if len(tvt_v) >= 100:
        md_tail = md_v[-100:]
        tvt_tail = tvt_v[-100:]
        try:
            a, b = np.polyfit(md_tail - md_tail[0], tvt_tail, 1)
            out["tail_100_slope_md"] = float(a)
            # residual = TVT - (a*MD + b) std
            resid = tvt_tail - (a * (md_tail - md_tail[0]) + b)
            out["tail_100_slope_resid_std"] = float(np.std(resid))
        except Exception:
            out["tail_100_slope_md"] = np.nan
            out["tail_100_slope_resid_std"] = np.nan
    else:
        out["tail_100_slope_md"] = np.nan
        out["tail_100_slope_resid_std"] = np.nan

    # GR signal of last 100 visible rows
    gr_v = df.loc[is_vis, "GR"].values
    if len(gr_v) >= 100:
        out["tail_100_gr_mean"] = float(np.nanmean(gr_v[-100:]))
        out["tail_100_gr_std"] = float(np.nanstd(gr_v[-100:]))
    else:
        out["tail_100_gr_mean"] = np.nan
        out["tail_100_gr_std"] = np.nan

    return out


def per_well_gr_profile_512(well_id: str, src: Path, split: str) -> np.ndarray:
    """各 well の visible GR profile を 512 サンプルに resample.

    cross-well GR similarity / PCA 入力用.
    """
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    is_vis = df["TVT_input"].notna().values
    gr = df.loc[is_vis, "GR"].values
    gr = gr[~np.isnan(gr)]
    if len(gr) < 50:
        return np.full(512, np.nan)
    # robust standardize (= per-well median, MAD)
    med = np.nanmedian(gr)
    mad = np.nanmedian(np.abs(gr - med)) + 1e-6
    gr_norm = (gr - med) / (1.4826 * mad)
    # resample to 512 (linear interp on idx)
    src_idx = np.linspace(0, len(gr) - 1, 512)
    tgt = np.interp(src_idx, np.arange(len(gr)), gr_norm)
    return tgt


def main():
    train_wells = list_wells(TRAIN_DIR)
    test_wells = list_wells(TEST_DIR)
    print(f"train wells: {len(train_wells)}, test wells: {len(test_wells)}")

    # tail stats
    tail_rows = []
    gr_matrix = []
    well_ids = []
    splits = []
    for i, w in enumerate(train_wells):
        if i % 100 == 0:
            print(f"  train tail [{i:4d}/{len(train_wells)}] {w}")
        tail_rows.append(tail_stats(w, TRAIN_DIR, "train"))
        prof = per_well_gr_profile_512(w, TRAIN_DIR, "train")
        gr_matrix.append(prof)
        well_ids.append(w)
        splits.append("train")
    for w in test_wells:
        tail_rows.append(tail_stats(w, TEST_DIR, "test"))
        prof = per_well_gr_profile_512(w, TEST_DIR, "test")
        gr_matrix.append(prof)
        well_ids.append(w)
        splits.append("test")

    df_tail = pd.DataFrame(tail_rows)
    df_tail.to_parquet(OUT_DIR / "tail-stats.parquet", index=False)
    print(f"saved tail -> {OUT_DIR / 'tail-stats.parquet'}")

    # GR PCA + KMeans cluster
    gr_arr = np.array(gr_matrix)
    valid = ~np.any(np.isnan(gr_arr), axis=1)
    print(f"valid GR profiles: {valid.sum()} / {len(gr_arr)}")
    if valid.sum() >= 50:
        pca = PCA(n_components=10, random_state=0)
        gr_pca = pca.fit_transform(gr_arr[valid])
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"sum top-10 variance: {pca.explained_variance_ratio_.sum():.4f}")

        # KMeans on PCA top 10
        for n_clusters in (10, 20, 40, 67):
            km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = km.fit_predict(gr_pca)
            # cluster size distribution
            uniq, cnt = np.unique(labels, return_counts=True)
            print(f"  KMeans n={n_clusters}: cluster sizes p10={np.percentile(cnt, 10):.1f}, p50={np.median(cnt):.1f}, p90={np.percentile(cnt, 90):.1f}, max={cnt.max()}")

        # 採用: 40 clusters
        km = KMeans(n_clusters=40, random_state=0, n_init=10)
        labels_40 = km.fit_predict(gr_pca)

        # full result
        df_pca = pd.DataFrame(gr_pca, columns=[f"pc{i+1}" for i in range(10)])
        df_pca["well_id"] = np.array(well_ids)[valid]
        df_pca["split"] = np.array(splits)[valid]
        df_pca["gr_cluster_40"] = labels_40
        df_pca.to_parquet(OUT_DIR / "gr-pca.parquet", index=False)
        print(f"saved gr-pca -> {OUT_DIR / 'gr-pca.parquet'}")

        # test wells の cluster 所属確認
        print("\n=== test wells GR PCA cluster ===")
        print(df_pca[df_pca["split"] == "test"][["well_id", "pc1", "pc2", "pc3", "gr_cluster_40"]].to_string())

        # b cluster と GR cluster の関係
        df_bc = pd.read_parquet(OUT_DIR / "b-cluster-xy.parquet")
        df_bc["b_round"] = df_bc["b_ANCC_med"].round(2)
        df_merge = df_pca.merge(df_bc[["well_id", "b_round"]], on="well_id", how="left")
        df_merge = df_merge.dropna(subset=["b_round"])
        # GR cluster と b cluster の相関 (= 同 GR cluster 内で b cluster 同じか?)
        from collections import Counter
        gr_to_b_purity = []
        for gc, g in df_merge.groupby("gr_cluster_40"):
            b_counts = g["b_round"].value_counts()
            purity = b_counts.iloc[0] / b_counts.sum() if len(b_counts) > 0 else 0
            gr_to_b_purity.append({"gr_cluster": gc, "size": len(g), "b_modes": len(b_counts),
                                    "top_b_round": float(b_counts.index[0]), "purity": purity})
        df_purity = pd.DataFrame(gr_to_b_purity).sort_values("size", ascending=False)
        print("\n=== GR cluster -> b cluster purity (= 同 GR cluster 内で同 b cluster な比率) ===")
        print(df_purity.head(15).to_string())
        print(f"median purity: {df_purity['purity'].median():.3f} (= 0.5 = ランダム/0.9+ = strong link)")
    else:
        print("not enough valid profiles for PCA")


if __name__ == "__main__":
    main()
