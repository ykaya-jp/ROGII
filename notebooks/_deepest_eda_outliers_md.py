"""Deepest EDA Phase 2: outlier well 検出 + MD-dependent statistics + 物理 formula residual deep.

1. 物理 formula residual outlier (= per-well b_well_resid_std の p99 超え)
2. MD-dependent dTVT distribution (= visible 末端からの MD offset 別 dTVT 分布)
3. typewell-horizontal pair の per-well alignment quality (= NCC max score)
4. visible 比率 × hidden_len の interaction (= 難 well の sigmaiture)
5. test 3 well の position in train distribution (= "test wells はどこにいるか")

実行: uv run python notebooks/_deepest_eda_outliers_md.py
出力: outputs/eda/deepest_eda/per-well-outlier.parquet
       outputs/eda/deepest_eda/md-dependent-dtvt.parquet
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


def per_well_outlier(well_id: str, src: Path, split: str) -> dict:
    """物理 formula `TVT = -Z + ANCC + b_well` の per-well residual 詳細."""
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    n = len(df)
    is_vis = df["TVT_input"].notna().values
    n_vis = int(is_vis.sum())
    out = dict(well_id=well_id, split=split, n_rows=n, n_visible=n_vis,
               visible_ratio=n_vis / n if n else 0.0)

    if "TVT" not in df.columns or "ANCC" not in df.columns or n_vis < 30:
        for k in ["b_med", "b_std", "b_mad", "b_iqr", "b_resid_p99", "b_resid_max",
                   "ncc_self_max", "ncc_self_p90", "ncc_self_median",
                   "vis_ratio_quartile_b_std", "first_quartile_b_med",
                   "last_quartile_b_med", "b_quartile_diff"]:
            out[k] = np.nan
        return out

    # b_well per row = TVT + Z - ANCC (visible only)
    tvt = df["TVT"].values
    z = df["Z"].values
    ancc = df["ANCC"].values
    b = tvt + z - ancc
    b_v = b[is_vis]
    out["b_med"] = float(np.nanmedian(b_v))
    out["b_std"] = float(np.nanstd(b_v))
    out["b_mad"] = float(np.nanmedian(np.abs(b_v - np.nanmedian(b_v))))
    out["b_iqr"] = float(np.nanpercentile(b_v, 75) - np.nanpercentile(b_v, 25))
    resid = b_v - np.nanmedian(b_v)
    out["b_resid_p99"] = float(np.nanpercentile(np.abs(resid), 99))
    out["b_resid_max"] = float(np.nanmax(np.abs(resid)))

    # visible 内で 4 quartile に分けて per-quartile b_well の drift を測る
    q_idx = np.linspace(0, n_vis - 1, 5).astype(int)
    b_q_med = [float(np.nanmedian(b_v[q_idx[i]:q_idx[i+1]+1])) for i in range(4)]
    out["first_quartile_b_med"] = b_q_med[0]
    out["last_quartile_b_med"] = b_q_med[3]
    out["b_quartile_diff"] = b_q_med[3] - b_q_med[0]
    out["vis_ratio_quartile_b_std"] = float(np.std(b_q_med))

    # 自分自身の GR signal を NCC (= self-correlation)
    gr = df["GR"].values
    gr_v = gr[is_vis]
    gr_v = gr_v[~np.isnan(gr_v)]
    if len(gr_v) >= 100:
        # 30 ft template を visible 中央から 5 個取り、別位置との NCC max を per-template 計測
        win = 31  # ft
        centers = np.linspace(win, len(gr_v) - win - 1, 5).astype(int)
        max_corrs = []
        for c in centers:
            tpl = gr_v[c - win // 2: c + win // 2 + 1]
            tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-9)
            corrs = []
            for c2 in range(win, len(gr_v) - win, 5):
                if abs(c2 - c) < win:
                    continue
                wnd = gr_v[c2 - win // 2: c2 + win // 2 + 1]
                wnd = (wnd - wnd.mean()) / (wnd.std() + 1e-9)
                if len(wnd) == len(tpl):
                    corrs.append(np.mean(tpl * wnd))
            if corrs:
                max_corrs.append(np.nanmax(corrs))
        if max_corrs:
            out["ncc_self_max"] = float(np.nanmax(max_corrs))
            out["ncc_self_median"] = float(np.nanmedian(max_corrs))
            out["ncc_self_p90"] = float(np.nanpercentile(max_corrs, 90))
        else:
            out["ncc_self_max"] = np.nan
            out["ncc_self_median"] = np.nan
            out["ncc_self_p90"] = np.nan
    else:
        out["ncc_self_max"] = np.nan
        out["ncc_self_median"] = np.nan
        out["ncc_self_p90"] = np.nan

    return out


def md_dependent_dtvt_dist(well_id: str, src: Path, split: str) -> list:
    """visible 末端から MD offset 別の dTVT distribution.

    各 well で visible 末端 idx を境界として、hidden 部 (train) または
    visible 末端から +X ft 進んだ点の dTVT を bin-50ft ごとに集計.
    """
    df = pd.read_csv(src / f"{well_id}__horizontal_well.csv")
    if "TVT" not in df.columns:
        return []
    is_vis = df["TVT_input"].notna().values
    if not is_vis.any() or is_vis.all():
        return []
    boundary_idx = int(np.where(is_vis)[0].max())
    boundary_md = df["MD"].iloc[boundary_idx]
    md = df["MD"].values
    tvt = df["TVT"].values
    is_hidden = ~is_vis
    dtvt = np.diff(tvt)
    md_off = md[1:] - boundary_md  # 末端 MD からの offset
    is_hidden_pair = is_hidden[1:]
    rows = []
    bin_size = 100
    for bin_start in range(0, int(np.nanmax(md_off)) + bin_size, bin_size):
        mask = (md_off >= bin_start) & (md_off < bin_start + bin_size) & is_hidden_pair
        if mask.sum() < 3:
            continue
        dtvt_bin = dtvt[mask]
        rows.append({
            "well_id": well_id,
            "split": split,
            "md_off_start": bin_start,
            "n": int(mask.sum()),
            "mean": float(np.nanmean(dtvt_bin)),
            "std": float(np.nanstd(dtvt_bin)),
            "p5": float(np.nanpercentile(dtvt_bin, 5)),
            "p50": float(np.nanmedian(dtvt_bin)),
            "p95": float(np.nanpercentile(dtvt_bin, 95)),
        })
    return rows


def main():
    train_wells = list_wells(TRAIN_DIR)
    test_wells = list_wells(TEST_DIR)
    print(f"train wells: {len(train_wells)}, test wells: {len(test_wells)}")

    outliers = []
    md_dep = []
    for i, w in enumerate(train_wells):
        if i % 100 == 0:
            print(f"  train [{i:4d}/{len(train_wells)}] {w}")
        try:
            outliers.append(per_well_outlier(w, TRAIN_DIR, "train"))
            md_dep.extend(md_dependent_dtvt_dist(w, TRAIN_DIR, "train"))
        except Exception as e:
            print(f"  ERROR train {w}: {e}")
    for w in test_wells:
        print(f"  test {w}")
        try:
            outliers.append(per_well_outlier(w, TEST_DIR, "test"))
        except Exception as e:
            print(f"  ERROR test {w}: {e}")

    df_out = pd.DataFrame(outliers)
    out_path = OUT_DIR / "per-well-outlier.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"saved {len(df_out)} rows -> {out_path}")

    df_md = pd.DataFrame(md_dep)
    out_path2 = OUT_DIR / "md-dependent-dtvt.parquet"
    df_md.to_parquet(out_path2, index=False)
    print(f"saved {len(df_md)} bin rows -> {out_path2}")

    # outlier summary
    summary = {}
    for col in df_out.columns:
        if col in ("well_id", "split"):
            continue
        s = df_out[col].dropna()
        if len(s) == 0 or not np.issubdtype(s.dtype, np.number):
            continue
        summary[col] = dict(
            n=int(len(s)),
            mean=float(s.mean()),
            std=float(s.std()),
            p1=float(s.quantile(0.01)),
            p10=float(s.quantile(0.10)),
            p50=float(s.quantile(0.50)),
            p90=float(s.quantile(0.90)),
            p99=float(s.quantile(0.99)),
            min=float(s.min()),
            max=float(s.max()),
        )

    # md-dependent summary: bin 別の global aggregate
    md_summary = {}
    if len(df_md):
        for bin_start, g in df_md.groupby("md_off_start"):
            md_summary[f"bin_{bin_start}"] = dict(
                n_wells=int(g["well_id"].nunique()),
                n_pts=int(g["n"].sum()),
                std_p10=float(g["std"].quantile(0.10)),
                std_p50=float(g["std"].quantile(0.50)),
                std_p90=float(g["std"].quantile(0.90)),
                mean_p10=float(g["mean"].quantile(0.10)),
                mean_p50=float(g["mean"].quantile(0.50)),
                mean_p90=float(g["mean"].quantile(0.90)),
            )

    full_summary = {"outliers": summary, "md_dependent": md_summary}
    (OUT_DIR / "summary-outlier-md.json").write_text(json.dumps(full_summary, indent=2))
    print(f"summary -> {OUT_DIR / 'summary-outlier-md.json'}")

    # outlier list 抽出: b_resid_p99 上位 20
    if "b_resid_p99" in df_out.columns:
        top_resid = df_out.dropna(subset=["b_resid_p99"]).sort_values("b_resid_p99", ascending=False).head(20)
        print("\n=== top-20 b_resid_p99 outlier wells ===")
        print(top_resid[["well_id", "split", "b_resid_p99", "b_med", "b_std", "n_rows", "visible_ratio"]].to_string())

    # b_quartile_diff outliers: b_well が visible 内で drift する well
    if "b_quartile_diff" in df_out.columns:
        df_out_clean = df_out.dropna(subset=["b_quartile_diff"])
        top_drift = df_out_clean.iloc[df_out_clean["b_quartile_diff"].abs().sort_values(ascending=False).index[:20]]
        print("\n=== top-20 abs(b_quartile_diff) drift wells ===")
        print(top_drift[["well_id", "split", "b_quartile_diff", "first_quartile_b_med", "last_quartile_b_med",
                           "b_std", "visible_ratio"]].to_string())


if __name__ == "__main__":
    main()
