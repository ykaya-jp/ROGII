"""First-Principles 数値計測 script.

notebooks/01_first_principles_eda.ipynb の計算ロジックをそのまま script 化したもの。
nbformat 経由で notebook を生成するときに参照する。

実行: uv run python notebooks/_first_principles_measure.py
出力: outputs/eda/first_principles/*.parquet  + summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
TEST_DIR = REPO / "data/raw/test"
OUT_DIR = REPO / "outputs/eda/first_principles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def list_train_wells():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    return [f.stem.replace("__horizontal_well", "") for f in files]


def load_well(wid: str, split: str = "train"):
    base = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv")
    return hw, tw


def visible_mask(hw: pd.DataFrame) -> np.ndarray:
    """visible = TVT_input not NaN."""
    return ~hw["TVT_input"].isna().to_numpy()


# ---- Track E-2 計測 ----------------------------------------------------------


def measure_gr_noise_std(hw: pd.DataFrame, vmask: np.ndarray, win: int = 11) -> float:
    """visible 区間で GR の rolling-mean 残差 std を測る (= sensor + lithology HF noise).

    horizontal GR は NaN を含むため nanstd を使う.
    """
    if vmask.sum() < win * 2:
        return np.nan
    gr = hw.loc[vmask, "GR"].to_numpy()
    s = pd.Series(gr).rolling(win, center=True, min_periods=win // 2).mean()
    resid = gr - s.to_numpy()
    return float(np.nanstd(resid))


def measure_dtvt_dmd(hw: pd.DataFrame, vmask: np.ndarray) -> dict:
    """visible 区間 dTVT/dMD 統計 (= bit が 1ft 進むときに TVT がどれだけ変動するか)."""
    if vmask.sum() < 5:
        return {"dtvt_std": np.nan, "dtvt_p95": np.nan}
    tvt = hw.loc[vmask, "TVT"].to_numpy()
    md = hw.loc[vmask, "MD"].to_numpy()
    if len(tvt) < 2:
        return {"dtvt_std": np.nan, "dtvt_p95": np.nan}
    d = np.diff(tvt) / np.maximum(np.diff(md), 1e-6)
    return {
        "dtvt_std": float(np.std(d)),
        "dtvt_p95": float(np.percentile(np.abs(d), 95)),
        "dtvt_mean_abs": float(np.mean(np.abs(d))),
    }


def measure_b_well_resid_std(hw: pd.DataFrame, vmask: np.ndarray) -> dict:
    """visible 区間で TVT + Z - ANCC を計算し、その std (= b_well 推定誤差) を測る.

    train only (test では ANCC NaN なので skip)。
    """
    if "ANCC" not in hw.columns:
        return {"b_well_resid_std": np.nan, "b_well_mean": np.nan}
    sub = hw.loc[vmask]
    b = sub["TVT"].to_numpy() + sub["Z"].to_numpy() - sub["ANCC"].to_numpy()
    if len(b) < 5 or np.any(~np.isfinite(b)):
        return {"b_well_resid_std": np.nan, "b_well_mean": np.nan}
    return {
        "b_well_resid_std": float(np.std(b)),
        "b_well_mean": float(np.mean(b)),
        "b_well_drift_first_to_last": float(np.mean(b[-50:]) - np.mean(b[:50]))
        if len(b) >= 100
        else np.nan,
    }


def measure_typewell_horizontal_formation_diff(
    hw: pd.DataFrame, tw: pd.DataFrame, vmask: np.ndarray
) -> dict:
    """typewell の TVT で見た GR pattern と horizontal の visible TVT で見た GR pattern の差分.

    具体的には、visible TVT 範囲で typewell GR を補間 (TVT 軸) し、horizontal GR との差分の std を取る.
    """
    if vmask.sum() < 5 or len(tw) < 5:
        return {"tw_gr_resid_std": np.nan, "tw_gr_resid_mean_abs": np.nan}
    sub = hw.loc[vmask]
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy()
    tw_gr = tw_sorted["GR"].to_numpy()
    finite = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[finite]
    tw_gr = tw_gr[finite]
    if len(tw_tvt) < 5:
        return {"tw_gr_resid_std": np.nan, "tw_gr_resid_mean_abs": np.nan}
    h_tvt = sub["TVT"].to_numpy()
    h_gr = sub["GR"].to_numpy()
    in_range = (h_tvt >= tw_tvt.min()) & (h_tvt <= tw_tvt.max())
    if in_range.sum() < 5:
        return {"tw_gr_resid_std": np.nan, "tw_gr_resid_mean_abs": np.nan}
    interp_gr = np.interp(h_tvt[in_range], tw_tvt, tw_gr)
    diff = h_gr[in_range] - interp_gr
    return {
        "tw_gr_resid_std": float(np.nanstd(diff)),
        "tw_gr_resid_mean_abs": float(np.nanmean(np.abs(diff))),
        "tw_gr_resid_mean": float(np.nanmean(diff)),
    }


def measure_hidden_length(hw: pd.DataFrame, vmask: np.ndarray) -> int:
    return int((~vmask).sum())


# ---- AR(1) noise structure ----------------------------------------------


def measure_ar1_residual(hw: pd.DataFrame, vmask: np.ndarray) -> dict:
    """visible 区間 TVT を de-meaned して AR(1) 当てはめ.

    TVT(s) - mu = phi * (TVT(s-1) - mu) + eps(s)
    eps 残差 std と phi を返す.
    """
    if vmask.sum() < 50:
        return {"ar1_phi": np.nan, "ar1_eps_std": np.nan}
    tvt = hw.loc[vmask, "TVT"].to_numpy()
    if not np.all(np.isfinite(tvt)) or len(tvt) < 50:
        return {"ar1_phi": np.nan, "ar1_eps_std": np.nan}
    # 1 階差分の AR(1) (= TVT 自体は trend を持つので diff の AR(1))
    d = np.diff(tvt)
    if len(d) < 10:
        return {"ar1_phi": np.nan, "ar1_eps_std": np.nan}
    x = d[:-1]
    y = d[1:]
    if np.std(x) < 1e-12:
        return {"ar1_phi": 0.0, "ar1_eps_std": float(np.std(d))}
    phi = float(np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-12))
    eps = y - phi * x
    return {"ar1_phi": phi, "ar1_eps_std": float(np.std(eps))}


# ---- main loop ---------------------------------------------------------------


def main():
    wids = list_train_wells()
    rows = []
    for i, wid in enumerate(wids):
        try:
            hw, tw = load_well(wid, "train")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {wid}: {e}")
            continue
        vmask = visible_mask(hw)
        rec: dict = {"well_id": wid, "n_rows": len(hw), "n_visible": int(vmask.sum())}
        rec["hidden_len"] = measure_hidden_length(hw, vmask)
        rec["visible_ratio"] = float(vmask.mean())
        rec["gr_noise_std"] = measure_gr_noise_std(hw, vmask)
        rec.update(measure_dtvt_dmd(hw, vmask))
        rec.update(measure_b_well_resid_std(hw, vmask))
        rec.update(measure_typewell_horizontal_formation_diff(hw, tw, vmask))
        rec.update(measure_ar1_residual(hw, vmask))
        # GR scale per well (NaN を含むので nan-aware)
        gr_v = hw.loc[vmask, "GR"].to_numpy()
        rec["gr_mean"] = float(np.nanmean(gr_v)) if len(gr_v) else np.nan
        rec["gr_std"] = float(np.nanstd(gr_v)) if len(gr_v) else np.nan
        rec["gr_nan_frac"] = float(np.isnan(gr_v).mean()) if len(gr_v) else np.nan
        rec["tvt_range"] = float(hw["TVT"].max() - hw["TVT"].min())
        rec["z_range"] = float(hw["Z"].max() - hw["Z"].min())
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  done {i + 1}/{len(wids)}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "per-well-stats.parquet", index=False)

    # aggregate summary
    summary = {
        "n_wells_measured": int(len(df)),
        "GR_noise_std_p10": float(df["gr_noise_std"].quantile(0.10)),
        "GR_noise_std_p50": float(df["gr_noise_std"].quantile(0.50)),
        "GR_noise_std_p90": float(df["gr_noise_std"].quantile(0.90)),
        "dTVT_std_p50": float(df["dtvt_std"].quantile(0.50)),
        "dTVT_p95_p50": float(df["dtvt_p95"].quantile(0.50)),
        "b_well_resid_std_p10": float(df["b_well_resid_std"].quantile(0.10)),
        "b_well_resid_std_p50": float(df["b_well_resid_std"].quantile(0.50)),
        "b_well_resid_std_p90": float(df["b_well_resid_std"].quantile(0.90)),
        "tw_gr_resid_std_p10": float(df["tw_gr_resid_std"].quantile(0.10)),
        "tw_gr_resid_std_p50": float(df["tw_gr_resid_std"].quantile(0.50)),
        "tw_gr_resid_std_p90": float(df["tw_gr_resid_std"].quantile(0.90)),
        "ar1_phi_p50": float(df["ar1_phi"].quantile(0.50)),
        "ar1_eps_std_p50": float(df["ar1_eps_std"].quantile(0.50)),
        "hidden_len_p10": float(df["hidden_len"].quantile(0.10)),
        "hidden_len_p50": float(df["hidden_len"].quantile(0.50)),
        "hidden_len_p90": float(df["hidden_len"].quantile(0.90)),
        "hidden_len_max": float(df["hidden_len"].max()),
        "visible_ratio_p50": float(df["visible_ratio"].quantile(0.50)),
        "n_visible_p50": float(df["n_visible"].quantile(0.50)),
        "n_visible_p10": float(df["n_visible"].quantile(0.10)),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return df, summary


if __name__ == "__main__":
    main()
