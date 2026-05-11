"""Deepest EDA Phase 1: 高次 per-well statistics + formation boundary 挙動.

公開 EDA (= pilkwang super stack, cdeotte EDA) + 既存 first_principles を超える deep dive。
773 train wells + 3 test wells を全数走査し、以下を計測:

1. GR の高次統計 (kurtosis, skewness, lag-1/lag-10 autocorr)
2. TVT 微分の高次統計 (d2TVT/dMD^2 = curvature)
3. trajectory 統計 (dogleg / azimuth / dip / inclination)
4. formation top discontinuity (6 formation boundary における TVT-Z 関係 jump)
5. visible-hidden boundary 近傍 statistics (boundary ±50 ft の dTVT)
6. typewell-horizontal alignment (lag, scale)

実行: uv run python notebooks/_deepest_eda_high_order.py
出力: outputs/eda/deepest_eda/per-well-high-order.parquet
"""
from __future__ import annotations

import json
import os
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


def safe_kurt(x: np.ndarray) -> float:
    """Fisher kurtosis (= excess kurtosis), 値が定数のときは 0 を返す."""
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)


def safe_skew(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**3))


def autocorr(x: np.ndarray, lag: int) -> float:
    x = x[~np.isnan(x)]
    if len(x) <= lag + 1:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def trajectory_features(md: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """trajectory derivatives: inclination, azimuth, dogleg, dip changes."""
    if len(md) < 3:
        return {k: np.nan for k in ["incl_mean", "incl_std", "azi_mean", "azi_std",
                                     "dogleg_mean", "dogleg_p95", "z_dip_mean", "z_dip_std"]}
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    dmd = np.diff(md)
    dxy = np.sqrt(dx**2 + dy**2)
    # inclination = angle from vertical, deg. dxy / |dr| ratio.
    dr = np.sqrt(dxy**2 + dz**2)
    incl = np.degrees(np.arctan2(dxy, -dz + 1e-9))  # vertical=0, horizontal=90
    # azimuth in horizontal plane (deg from north clockwise)
    azi = np.degrees(np.arctan2(dx, dy + 1e-9)) % 360.0
    # dogleg severity = angle between successive direction vectors per 100 ft
    v = np.column_stack([dx, dy, dz])
    norms = np.linalg.norm(v, axis=1) + 1e-9
    v_norm = v / norms[:, None]
    if len(v_norm) >= 2:
        cos_th = np.sum(v_norm[:-1] * v_norm[1:], axis=1).clip(-1, 1)
        dl = np.degrees(np.arccos(cos_th))
        dl_per_100 = dl / np.maximum(dmd[1:], 1.0) * 100.0
    else:
        dl_per_100 = np.array([np.nan])
    # vertical dip rate = dz / dmd
    z_dip = dz / np.maximum(dmd, 1e-6)
    return dict(
        incl_mean=float(np.nanmean(incl)),
        incl_std=float(np.nanstd(incl)),
        azi_mean=float(np.nanmean(azi)),
        azi_std=float(np.nanstd(azi)),
        dogleg_mean=float(np.nanmean(dl_per_100)),
        dogleg_p95=float(np.nanpercentile(dl_per_100, 95)),
        z_dip_mean=float(np.nanmean(z_dip)),
        z_dip_std=float(np.nanstd(z_dip)),
    )


def formation_boundary_features(df: pd.DataFrame) -> dict:
    """formation top depth = ANCC..BUDA を所有する well で、formation 境界で TVT - (-Z + formation) のジャンプを計測.

    各 formation の depth は train でしか観測されない。test では NaN なので
    train only で計算。値域: formation 境界 ±5 ft window で per-well slope の jump 量。
    """
    out = {}
    if "ANCC" not in df.columns:
        # test well: skip
        for f in FORMATIONS:
            out[f"b_jump_at_{f}"] = np.nan
        out["b_jump_max_abs"] = np.nan
        return out
    # b_well = TVT + Z - formation (per row, per formation)
    b_mat = np.full((len(df), len(FORMATIONS)), np.nan)
    for i, f in enumerate(FORMATIONS):
        if f in df.columns:
            b_mat[:, i] = df["TVT"].values + df["Z"].values - df[f].values
    # per formation: b_well の per-well std (= 各 formation で b_well がどれだけ一定か)
    # b_jump = 隣接 formation での b 値の差 abs (= per-row median 6 formation 間 std)
    f_std = np.nanstd(b_mat, axis=1)
    for i, f in enumerate(FORMATIONS):
        b_f = b_mat[:, i]
        out[f"b_jump_at_{f}"] = float(np.nanstd(b_f)) if np.any(~np.isnan(b_f)) else np.nan
    out["b_jump_max_abs"] = float(np.nanmax(f_std)) if np.any(~np.isnan(f_std)) else np.nan
    # cross-formation b agreement: 同 row で 6 formation 計算した b の std
    out["b_cross_formation_std_p50"] = float(np.nanmedian(f_std)) if np.any(~np.isnan(f_std)) else np.nan
    return out


def visible_boundary_features(df: pd.DataFrame) -> dict:
    """visible-hidden 境界 ±50 ft window での挙動.

    boundary = visible 末端の MD。±50 ft 内で dTVT / dGR の rate change 計測。
    visible は train only (test では visible のみ観測、hidden 部 TVT は NaN なので train でしか計算可)。
    """
    out = {}
    if "TVT" not in df.columns:
        for k in ["dtvt_pre_boundary_std", "dtvt_post_boundary_std", "dtvt_post_pre_ratio",
                   "gr_pre_boundary_std", "gr_post_boundary_std"]:
            out[k] = np.nan
        return out
    is_vis = df["TVT_input"].notna().values
    if not is_vis.any() or is_vis.all():
        for k in ["dtvt_pre_boundary_std", "dtvt_post_boundary_std", "dtvt_post_pre_ratio",
                   "gr_pre_boundary_std", "gr_post_boundary_std"]:
            out[k] = np.nan
        return out
    boundary_idx = int(np.where(is_vis)[0].max())
    md = df["MD"].values
    tvt = df["TVT"].values
    gr = df["GR"].values
    # ±50 ft window
    boundary_md = md[boundary_idx]
    pre_mask = (md >= boundary_md - 50) & (md <= boundary_md) & is_vis
    post_mask = (md > boundary_md) & (md <= boundary_md + 50) & ~is_vis
    pre_dtvt = np.diff(tvt[pre_mask])
    post_dtvt = np.diff(tvt[post_mask])
    pre_gr = gr[pre_mask]
    post_gr = gr[post_mask]
    out["dtvt_pre_boundary_std"] = float(np.nanstd(pre_dtvt)) if len(pre_dtvt) >= 2 else np.nan
    out["dtvt_post_boundary_std"] = float(np.nanstd(post_dtvt)) if len(post_dtvt) >= 2 else np.nan
    if out["dtvt_pre_boundary_std"] and out["dtvt_pre_boundary_std"] > 1e-9:
        out["dtvt_post_pre_ratio"] = out["dtvt_post_boundary_std"] / out["dtvt_pre_boundary_std"]
    else:
        out["dtvt_post_pre_ratio"] = np.nan
    out["gr_pre_boundary_std"] = float(np.nanstd(pre_gr)) if len(pre_gr) >= 2 else np.nan
    out["gr_post_boundary_std"] = float(np.nanstd(post_gr)) if len(post_gr) >= 2 else np.nan
    return out


def per_well_high_order(well_id: str, src: Path) -> dict:
    h_path = src / f"{well_id}__horizontal_well.csv"
    df = pd.read_csv(h_path)
    n = len(df)
    md = df["MD"].values
    x = df["X"].values
    y = df["Y"].values
    z = df["Z"].values
    gr = df["GR"].values

    is_vis = df["TVT_input"].notna().values
    n_vis = int(is_vis.sum())

    # GR 高次統計
    gr_vis = gr[is_vis] if n_vis > 0 else gr
    gr_kurt = safe_kurt(gr_vis)
    gr_skew = safe_skew(gr_vis)
    gr_ac1 = autocorr(gr_vis, 1)
    gr_ac10 = autocorr(gr_vis, 10)
    gr_ac50 = autocorr(gr_vis, 50)

    # TVT 高次 (train のみ全体使える)
    tvt_kurt = np.nan
    tvt_skew = np.nan
    dtvt_ac1 = np.nan
    d2tvt_std = np.nan
    d2tvt_p95 = np.nan
    if "TVT" in df.columns:
        tvt = df["TVT"].values
        # visible 部 (= TVT_input 非NaN) で計測
        tvt_v = tvt[is_vis]
        dtvt = np.diff(tvt_v)
        d2tvt = np.diff(dtvt)
        tvt_kurt = safe_kurt(dtvt)
        tvt_skew = safe_skew(dtvt)
        dtvt_ac1 = autocorr(dtvt, 1)
        d2tvt_std = float(np.nanstd(d2tvt)) if len(d2tvt) >= 2 else np.nan
        d2tvt_p95 = float(np.nanpercentile(np.abs(d2tvt), 95)) if len(d2tvt) >= 2 else np.nan

    # trajectory 特徴
    traj = trajectory_features(md, x, y, z)

    # formation boundary
    fbf = formation_boundary_features(df)

    # visible boundary
    vbf = visible_boundary_features(df)

    # typewell alignment (簡易): typewell GR と horizontal GR の per-well median lag
    tw_path = src / f"{well_id}__typewell.csv"
    tw_lag_best = np.nan
    tw_scale_best = np.nan
    if tw_path.exists():
        tw = pd.read_csv(tw_path)
        # typewell TVT, GR で TVT_input range と overlap する区間を抽出
        if "TVT_input" in df.columns and is_vis.any():
            tvt_vis = df.loc[is_vis, "TVT_input"].values
            gr_vis_arr = df.loc[is_vis, "GR"].values
            t_min, t_max = float(np.nanmin(tvt_vis)), float(np.nanmax(tvt_vis))
            tw_sub = tw[(tw["TVT"] >= t_min - 50) & (tw["TVT"] <= t_max + 50)]
            if len(tw_sub) >= 50 and len(gr_vis_arr) >= 50:
                # interp tw GR onto horizontal TVT_input grid
                gr_tw_interp = np.interp(tvt_vis, tw_sub["TVT"].values, tw_sub["GR"].values)
                gr_h = gr_vis_arr.copy()
                gr_tw_interp = gr_tw_interp[~np.isnan(gr_h)]
                gr_h = gr_h[~np.isnan(gr_h)]
                if len(gr_h) >= 30 and gr_h.std() > 1e-6 and gr_tw_interp.std() > 1e-6:
                    # 線形 OLS scale (GR_h ≈ a * GR_tw + b)
                    A = np.column_stack([gr_tw_interp, np.ones_like(gr_tw_interp)])
                    try:
                        coef, *_ = np.linalg.lstsq(A, gr_h, rcond=None)
                        tw_scale_best = float(coef[0])
                    except Exception:
                        pass

    out = dict(
        well_id=well_id,
        gr_kurt=gr_kurt,
        gr_skew=gr_skew,
        gr_ac1=gr_ac1,
        gr_ac10=gr_ac10,
        gr_ac50=gr_ac50,
        dtvt_kurt=tvt_kurt,
        dtvt_skew=tvt_skew,
        dtvt_ac1=dtvt_ac1,
        d2tvt_std=d2tvt_std,
        d2tvt_p95=d2tvt_p95,
        tw_scale_best=tw_scale_best,
        tw_lag_best=tw_lag_best,
    )
    out.update(traj)
    out.update(fbf)
    out.update(vbf)
    return out


def main():
    train_wells = list_wells(TRAIN_DIR)
    test_wells = list_wells(TEST_DIR)
    print(f"train wells: {len(train_wells)}, test wells: {len(test_wells)}")

    rows = []
    for i, w in enumerate(train_wells):
        if i % 50 == 0:
            print(f"  train [{i:4d}/{len(train_wells)}] {w}")
        try:
            rows.append({"split": "train", **per_well_high_order(w, TRAIN_DIR)})
        except Exception as e:
            print(f"  ERROR train {w}: {e}")
    for w in test_wells:
        print(f"  test {w}")
        try:
            rows.append({"split": "test", **per_well_high_order(w, TEST_DIR)})
        except Exception as e:
            print(f"  ERROR test {w}: {e}")

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "per-well-high-order.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"saved {len(df_out)} rows -> {out_path}")

    # summary
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
            p10=float(s.quantile(0.10)),
            p50=float(s.quantile(0.50)),
            p90=float(s.quantile(0.90)),
            min=float(s.min()),
            max=float(s.max()),
        )
    (OUT_DIR / "summary-high-order.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {OUT_DIR / 'summary-high-order.json'}")


if __name__ == "__main__":
    main()
