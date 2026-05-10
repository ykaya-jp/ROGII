"""First-Principles 補助計測 (Track E-2 完成用).

1. typewell-horizontal の formation thickness diff (= TVT 軸上で同一 GR pattern が現れる位置のズレ)
2. b_well の per-well drift (visible 区間で md_since=0 と md_since=max での差)
3. visible 末尾 N=200 ft の dTVT/dMD 安定性 (= last_known anchor の信頼度)
4. typewell 単独で予測したときの理論的 RMSE 下限 (= typewell GR pattern matching の理論性能)

実行: uv run python notebooks/_first_principles_extras.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
OUT_DIR = REPO / "outputs/eda/first_principles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_wells():
    return sorted(
        [f.stem.replace("__horizontal_well", "") for f in TRAIN_DIR.glob("*__horizontal_well.csv")]
    )


def load(wid: str):
    hw = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv")
    return hw, tw


def visible_mask(hw: pd.DataFrame) -> np.ndarray:
    return ~hw["TVT_input"].isna().to_numpy()


# ---- 1. typewell の "TVT 範囲全体での formation thickness 差" ---------------
# 仮説: typewell GR pattern が horizontal の visible TVT 範囲で見られる pattern と
# 同一 TVT で完全一致する場合 RMSE = 0。実際には地理的に離れているので
# 同じ depth でも GR が違う (地層厚 / 岩相変化)。
# このズレを「visible で TVT が既知のとき、horizontal GR を typewell GR(TVT) に
# どれだけ一致させられるか (best lag を探した上で)」で測る。
# best lag = 上下に shift した tw_gr で h_gr との MSE が最小になる shift。
# その lag が "実効的な formation thickness diff" の代理。


def measure_best_lag_diff(hw: pd.DataFrame, tw: pd.DataFrame, vmask: np.ndarray, max_lag_ft: float = 50.0) -> dict:
    """visible 区間 で best lag を grid search して、その lag と residual std を返す."""
    if vmask.sum() < 100:
        return {"best_lag_ft": np.nan, "min_resid_std": np.nan}
    sub = hw.loc[vmask]
    h_tvt = sub["TVT"].to_numpy()
    h_gr = sub["GR"].to_numpy()
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy()
    tw_gr = tw_sorted["GR"].to_numpy()
    finite_h = np.isfinite(h_gr)
    finite_t = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
    if finite_h.sum() < 100 or finite_t.sum() < 50:
        return {"best_lag_ft": np.nan, "min_resid_std": np.nan}
    h_tvt = h_tvt[finite_h]
    h_gr = h_gr[finite_h]
    tw_tvt = tw_tvt[finite_t]
    tw_gr = tw_gr[finite_t]
    lags = np.arange(-max_lag_ft, max_lag_ft + 0.5, 0.5)
    best = (np.inf, np.nan)
    for L in lags:
        shifted = h_tvt + L
        mask = (shifted >= tw_tvt.min()) & (shifted <= tw_tvt.max())
        if mask.sum() < 50:
            continue
        interp = np.interp(shifted[mask], tw_tvt, tw_gr)
        resid = h_gr[mask] - interp
        # affine cal で z-score 残差にする
        if np.std(interp) < 1e-6 or np.std(h_gr[mask]) < 1e-6:
            continue
        a = np.cov(interp, h_gr[mask], bias=True)[0, 1] / (np.var(interp) + 1e-9)
        b = np.mean(h_gr[mask]) - a * np.mean(interp)
        cal_resid = h_gr[mask] - (a * interp + b)
        s = float(np.nanstd(cal_resid))
        if s < best[0]:
            best = (s, float(L))
    return {"best_lag_ft": best[1], "min_resid_std": best[0]}


# ---- 2. b_well の visible 区間内 drift -------------------------------------


def measure_b_well_drift(hw: pd.DataFrame, vmask: np.ndarray) -> dict:
    if "ANCC" not in hw.columns or vmask.sum() < 200:
        return {"b_drift_first_last": np.nan, "b_drift_per_ft": np.nan}
    sub = hw.loc[vmask].reset_index(drop=True)
    b = sub["TVT"].to_numpy() + sub["Z"].to_numpy() - sub["ANCC"].to_numpy()
    md = sub["MD"].to_numpy()
    if not np.all(np.isfinite(b)):
        return {"b_drift_first_last": np.nan, "b_drift_per_ft": np.nan}
    n = len(b)
    head_mean = float(np.mean(b[: n // 5]))
    tail_mean = float(np.mean(b[-n // 5 :]))
    drift = tail_mean - head_mean
    md_span = md[-1] - md[0]
    return {
        "b_drift_first_last": drift,
        "b_drift_per_ft": float(drift / max(md_span, 1.0)),
    }


# ---- 3. visible 末尾の dTVT/dMD 統計 (last_known anchor 用) -----------------


def measure_visible_tail_dtvt(hw: pd.DataFrame, vmask: np.ndarray, tail_ft: int = 200) -> dict:
    if vmask.sum() < tail_ft + 5:
        return {"tail_dtvt_mean": np.nan, "tail_dtvt_std": np.nan}
    sub = hw.loc[vmask].iloc[-tail_ft:]
    md = sub["MD"].to_numpy()
    tvt = sub["TVT"].to_numpy()
    if not np.all(np.isfinite(tvt)):
        return {"tail_dtvt_mean": np.nan, "tail_dtvt_std": np.nan}
    d = np.diff(tvt) / np.maximum(np.diff(md), 1e-6)
    return {
        "tail_dtvt_mean": float(np.mean(d)),
        "tail_dtvt_std": float(np.std(d)),
    }


# ---- 4. extrapolation length に対する residual growth ---------------------
# visible 末尾 anchor から hidden を full 区間で延長 (= last_known broadcast) したとき
# md_since の関数として residual std が grow rate.
# これは "naive last-value" baseline の理論性能。


def extrap_resid_curve(hw: pd.DataFrame, vmask: np.ndarray) -> pd.DataFrame:
    """各 well で hidden 区間内の md_since vs (TVT_true - last_known) を返す.

    返り値: DataFrame[md_since, residual] (1 well 分)。後で aggregate.
    """
    if vmask.sum() < 50 or (~vmask).sum() < 50:
        return pd.DataFrame(columns=["md_since", "residual"])
    last_known_idx = np.where(vmask)[0][-1]
    last_known_tvt = hw["TVT"].iloc[last_known_idx]
    last_known_md = hw["MD"].iloc[last_known_idx]
    hidden = hw.iloc[last_known_idx + 1 :].copy()
    if len(hidden) == 0:
        return pd.DataFrame(columns=["md_since", "residual"])
    md_since = hidden["MD"].to_numpy() - last_known_md
    resid = hidden["TVT"].to_numpy() - last_known_tvt
    return pd.DataFrame({"md_since": md_since, "residual": resid})


def main():
    wids = list_wells()
    rows = []
    extrap_chunks = []
    for i, wid in enumerate(wids):
        try:
            hw, tw = load(wid)
        except Exception:
            continue
        vmask = visible_mask(hw)
        rec = {"well_id": wid}
        rec.update(measure_best_lag_diff(hw, tw, vmask))
        rec.update(measure_b_well_drift(hw, vmask))
        rec.update(measure_visible_tail_dtvt(hw, vmask))
        rows.append(rec)
        ec = extrap_resid_curve(hw, vmask)
        if len(ec) > 0:
            ec["well_id"] = wid
            extrap_chunks.append(ec)
        if (i + 1) % 100 == 0:
            print(f"  done {i+1}/{len(wids)}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "per-well-extras.parquet", index=False)
    extrap = pd.concat(extrap_chunks, ignore_index=True)
    # md_since を 50ft bin に集計
    extrap["bin"] = (extrap["md_since"] // 50).astype(int)
    agg = (
        extrap.groupby("bin")["residual"]
        .agg(["count", "mean", "std", lambda x: float(np.percentile(np.abs(x), 95))])
        .reset_index()
    )
    agg.columns = ["bin50ft", "n", "mean", "std", "p95_abs"]
    agg["md_since_center"] = agg["bin50ft"] * 50 + 25
    agg = agg[agg["n"] >= 50].reset_index(drop=True)
    agg.to_parquet(OUT_DIR / "extrapolation-residual-curve.parquet", index=False)

    # summary
    summary = {
        "best_lag_ft_p50_abs": float(np.nanmedian(np.abs(df["best_lag_ft"]))),
        "min_resid_std_p10": float(df["min_resid_std"].quantile(0.10)),
        "min_resid_std_p50": float(df["min_resid_std"].quantile(0.50)),
        "min_resid_std_p90": float(df["min_resid_std"].quantile(0.90)),
        "b_drift_first_last_p50_abs": float(np.nanmedian(np.abs(df["b_drift_first_last"]))),
        "b_drift_per_ft_p50_abs": float(np.nanmedian(np.abs(df["b_drift_per_ft"]))),
        "tail_dtvt_std_p50": float(df["tail_dtvt_std"].quantile(0.50)),
        "extrap_resid_std_at_md_50": float(
            agg.loc[agg["md_since_center"].between(0, 100), "std"].mean()
        )
        if len(agg) > 0
        else np.nan,
        "extrap_resid_std_at_md_500": float(
            agg.loc[agg["md_since_center"].between(450, 550), "std"].mean()
        )
        if len(agg) > 0
        else np.nan,
        "extrap_resid_std_at_md_2000": float(
            agg.loc[agg["md_since_center"].between(1950, 2050), "std"].mean()
        )
        if len(agg) > 0
        else np.nan,
        "extrap_resid_std_at_md_4000": float(
            agg.loc[agg["md_since_center"].between(3950, 4050), "std"].mean()
        )
        if len(agg) > 0
        else np.nan,
        "n_wells_extras": int(len(df)),
        "n_wells_extrap_curve": int(extrap["well_id"].nunique()) if len(extrap) > 0 else 0,
    }
    # 合算した naive last-value broadcast の RMSE
    if len(extrap) > 0:
        rmse_naive = float(np.sqrt(np.nanmean(extrap["residual"] ** 2)))
        summary["naive_last_value_rmse_overall"] = rmse_naive
    with open(OUT_DIR / "summary-extras.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
