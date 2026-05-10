"""理論的 RMSE 下限の数値見積もり.

Track E-2 の核心。

仮説 1 (GR-only oracle): hidden 区間で bit GR が観測できる. typewell GR(TVT) が完全に
  正しい reference なら、bit GR を typewell GR(TVT) に最尤 matching したときの TVT
  推定誤差 = (GR sensor noise) / (|d GR / d TVT|).
  → これを per-well で計測。

仮説 2 (formula oracle): formula `TVT = -Z + ANCC + b_well`. test では ANCC が
  Plane-fit imputer で imputed, b_well は visible 末尾から推定. plane-fit RMSE を
  既存 docs (konbu17 は 17 ft / R kernel は 11.4 ft 報告) から引用 → これだけで
  formula RMSE.

仮説 3 (Self-similarity oracle): hidden GR pattern を visible 区間にある同じ pattern と
  match する. visible 内 best lag matching の min resid (上で計測) が baseline.

実行: uv run python notebooks/_first_principles_lower_bound.py
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


# ---- 仮説 1: GR oracle (Cramér-Rao 風) -----------------------------------------


def measure_gr_cramer_rao(hw: pd.DataFrame, tw: pd.DataFrame, vmask: np.ndarray) -> dict:
    """visible 区間内で TVT 軸上の typewell GR 微分 |dGR/dTVT| を計算し、
    GR sensor noise std / |dGR/dTVT| = 1 点 GR 観測から TVT 推定の Cramér-Rao 下限.

    1 点だけだと L0 = noise_std / slope。N 個の連続観測を blend すると 1/sqrt(N) スケールで縮む。
    visible 区間で実測 sensor noise std と typewell GR slope を取って、両者の比を返す.
    """
    if vmask.sum() < 50:
        return {"gr_cr_lb_1pt": np.nan, "gr_slope_p50": np.nan, "gr_noise_std": np.nan}
    sub = hw.loc[vmask]
    gr_v = sub["GR"].to_numpy()
    finite = np.isfinite(gr_v)
    if finite.sum() < 50:
        return {"gr_cr_lb_1pt": np.nan, "gr_slope_p50": np.nan, "gr_noise_std": np.nan}
    # GR sensor noise (rolling 11 mean からの残差 std)
    s = pd.Series(gr_v[finite]).rolling(11, center=True, min_periods=5).mean().to_numpy()
    noise_std = float(np.nanstd(gr_v[finite] - s))
    # typewell GR(TVT) slope
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy()
    tw_gr = tw_sorted["GR"].to_numpy()
    finite_t = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if finite_t.sum() < 20:
        return {"gr_cr_lb_1pt": np.nan, "gr_slope_p50": np.nan, "gr_noise_std": noise_std}
    tw_tvt = tw_tvt[finite_t]
    tw_gr = tw_gr[finite_t]
    # smooth で噪音除去 してから微分
    tw_gr_smooth = pd.Series(tw_gr).rolling(11, center=True, min_periods=5).mean().to_numpy()
    valid_d = ~np.isnan(tw_gr_smooth)
    if valid_d.sum() < 20:
        return {"gr_cr_lb_1pt": np.nan, "gr_slope_p50": np.nan, "gr_noise_std": noise_std}
    dgr_dtvt = np.gradient(tw_gr_smooth[valid_d], tw_tvt[valid_d])
    abs_slope = np.abs(dgr_dtvt)
    # 0 に近い slope は除外 (= flat region は GR では位置わからない)
    valid_slope = abs_slope[abs_slope > 1e-3]
    if len(valid_slope) < 5:
        return {"gr_cr_lb_1pt": np.nan, "gr_slope_p50": np.nan, "gr_noise_std": noise_std}
    slope_p50 = float(np.median(valid_slope))
    # CR lower bound for 1-point GR observation
    cr_lb_1pt = noise_std / slope_p50  # ft
    return {
        "gr_cr_lb_1pt": cr_lb_1pt,
        "gr_slope_p50": slope_p50,
        "gr_slope_p10": float(np.percentile(abs_slope, 10)),
        "gr_slope_p90": float(np.percentile(abs_slope, 90)),
        "gr_noise_std": noise_std,
    }


# ---- 仮説 2: formula oracle (b_well + plane-fit による TVT 復元誤差) ----------
# test では ANCC は plane-fit imputer で予測。R kernel report では per-formation
# RMSE 11-15 ft (= ANCC plane-fit imputer 単独の精度)。
# visible 区間で b_well を median で取り、ANCC を plane fit (= 我々 wide で計算する)
# として formula 適用したときの TVT 復元誤差を per-well で測る。

# 簡易版: 「visible 区間 全 wells について、各 well の b_well を visible で median で
# 推定したのを引き算して、Z + ANCC を hidden で観測 (train なら可) → b_well_hat 適用
# → TVT 推定誤差」を hidden 区間で測る。これは "b_well 推定誤差" だけが寄与する RMSE
# (= ANCC は train で完璧, 実 test では ANCC imputer 誤差が追加で乗る)。


def measure_formula_oracle_rmse(hw: pd.DataFrame, vmask: np.ndarray) -> dict:
    if "ANCC" not in hw.columns or vmask.sum() < 50 or (~vmask).sum() < 50:
        return {"formula_oracle_rmse": np.nan}
    sub_v = hw.loc[vmask]
    sub_h = hw.loc[~vmask]
    b_v = (sub_v["TVT"] + sub_v["Z"] - sub_v["ANCC"]).to_numpy()
    if not np.all(np.isfinite(b_v)):
        return {"formula_oracle_rmse": np.nan}
    b_hat = float(np.median(b_v))
    # hidden 区間で formula 適用
    tvt_pred = -sub_h["Z"].to_numpy() + sub_h["ANCC"].to_numpy() + b_hat
    tvt_true = sub_h["TVT"].to_numpy()
    if not np.all(np.isfinite(tvt_pred)) or not np.all(np.isfinite(tvt_true)):
        return {"formula_oracle_rmse": np.nan}
    rmse = float(np.sqrt(np.mean((tvt_pred - tvt_true) ** 2)))
    return {"formula_oracle_rmse": rmse}


# ---- 仮説 3: GR pattern matching consistency check -----------------------------
# visible 内で hidden を模擬: visible 末尾 N=200 ft を hidden_synthetic として、
# 残り visible で best-lag matching を行い、TVT 推定誤差を測る。


def measure_synthetic_gr_matching(hw: pd.DataFrame, tw: pd.DataFrame, vmask: np.ndarray, hold_ft: int = 200) -> dict:
    if vmask.sum() < hold_ft + 200:
        return {"gr_match_synth_rmse": np.nan}
    visible_idx = np.where(vmask)[0]
    if len(visible_idx) < hold_ft + 100:
        return {"gr_match_synth_rmse": np.nan}
    train_idx = visible_idx[:-hold_ft]
    holdout_idx = visible_idx[-hold_ft:]
    sub_train = hw.iloc[train_idx]
    sub_hold = hw.iloc[holdout_idx]
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy()
    tw_gr = tw_sorted["GR"].to_numpy()
    finite_t = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[finite_t]
    tw_gr = tw_gr[finite_t]
    if len(tw_tvt) < 20:
        return {"gr_match_synth_rmse": np.nan}
    # affine cal を train で fit
    train_h_gr = sub_train["GR"].to_numpy()
    train_h_tvt = sub_train["TVT"].to_numpy()
    valid_train = (
        np.isfinite(train_h_gr)
        & np.isfinite(train_h_tvt)
        & (train_h_tvt >= tw_tvt.min())
        & (train_h_tvt <= tw_tvt.max())
    )
    if valid_train.sum() < 50:
        return {"gr_match_synth_rmse": np.nan}
    interp_train = np.interp(train_h_tvt[valid_train], tw_tvt, tw_gr)
    if np.std(interp_train) < 1e-6:
        return {"gr_match_synth_rmse": np.nan}
    a = np.cov(interp_train, train_h_gr[valid_train], bias=True)[0, 1] / (np.var(interp_train) + 1e-9)
    b = np.mean(train_h_gr[valid_train]) - a * np.mean(interp_train)
    # holdout で TVT を grid search
    hold_gr = sub_hold["GR"].to_numpy()
    hold_tvt_true = sub_hold["TVT"].to_numpy()
    valid_h = np.isfinite(hold_gr) & np.isfinite(hold_tvt_true)
    if valid_h.sum() < 30:
        return {"gr_match_synth_rmse": np.nan}
    # 各 holdout 点で TVT を grid search (range = visible TVT range ± 50 ft)
    tvt_range_lo = max(tw_tvt.min(), train_h_tvt.min() - 50)
    tvt_range_hi = min(tw_tvt.max(), train_h_tvt.max() + 50)
    tvt_grid = np.arange(tvt_range_lo, tvt_range_hi + 0.5, 0.5)
    if len(tvt_grid) < 20:
        return {"gr_match_synth_rmse": np.nan}
    interp_grid = a * np.interp(tvt_grid, tw_tvt, tw_gr) + b
    # for each holdout 点 → best TVT
    preds = []
    trues = []
    for gr_i, tvt_i in zip(hold_gr[valid_h], hold_tvt_true[valid_h]):
        diff2 = (interp_grid - gr_i) ** 2
        best_idx = int(np.argmin(diff2))
        preds.append(tvt_grid[best_idx])
        trues.append(tvt_i)
    preds = np.array(preds)
    trues = np.array(trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return {"gr_match_synth_rmse": rmse, "n_holdout": int(valid_h.sum())}


def main():
    wids = list_wells()
    rows = []
    for i, wid in enumerate(wids):
        try:
            hw, tw = load(wid)
        except Exception:
            continue
        vmask = visible_mask(hw)
        rec = {"well_id": wid}
        rec.update(measure_gr_cramer_rao(hw, tw, vmask))
        rec.update(measure_formula_oracle_rmse(hw, vmask))
        rec.update(measure_synthetic_gr_matching(hw, tw, vmask))
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  done {i+1}/{len(wids)}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "per-well-bound.parquet", index=False)

    summary = {
        "gr_cr_lb_1pt_p10": float(df["gr_cr_lb_1pt"].quantile(0.10)),
        "gr_cr_lb_1pt_p50": float(df["gr_cr_lb_1pt"].quantile(0.50)),
        "gr_cr_lb_1pt_p90": float(df["gr_cr_lb_1pt"].quantile(0.90)),
        "gr_slope_p50_p50": float(df["gr_slope_p50"].quantile(0.50)),
        "gr_noise_std_p50": float(df["gr_noise_std"].quantile(0.50)),
        "formula_oracle_rmse_p10": float(df["formula_oracle_rmse"].quantile(0.10)),
        "formula_oracle_rmse_p50": float(df["formula_oracle_rmse"].quantile(0.50)),
        "formula_oracle_rmse_p90": float(df["formula_oracle_rmse"].quantile(0.90)),
        "formula_oracle_rmse_overall": float(np.sqrt(np.mean(df["formula_oracle_rmse"].dropna() ** 2))),
        "gr_match_synth_rmse_p10": float(df["gr_match_synth_rmse"].quantile(0.10)),
        "gr_match_synth_rmse_p50": float(df["gr_match_synth_rmse"].quantile(0.50)),
        "gr_match_synth_rmse_p90": float(df["gr_match_synth_rmse"].quantile(0.90)),
        "gr_match_synth_rmse_overall": float(np.sqrt(np.mean(df["gr_match_synth_rmse"].dropna() ** 2))),
        "n_wells": int(len(df)),
    }
    with open(OUT_DIR / "summary-bound.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
