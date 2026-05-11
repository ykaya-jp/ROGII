"""Deepest EDA Phase 5: failure pattern simulation.

各 train well を pseudo-test として扱い、以下の 4 種 simple model の per-well error を計測:
1. last-value extrapolation (= TVT_pred = visible 末端 TVT、constant)
2. linear extrap from tail-100 (= slope を末端 100 行で fit)
3. tvt_formula = -Z + ANCC_median + b_well_median (= formula が known な best-case)
4. typewell GR-NCC match (= visible 末端 GR profile を typewell GR にマッチ)

per-well の {RMSE, MAE, max abs error} を測定し、failure mode の signature 抽出.

実行: uv run python notebooks/_deepest_eda_failure_simulation.py
出力: outputs/eda/deepest_eda/pseudo-test-errors.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
TRAIN_DIR = REPO / "data/raw/train"
OUT_DIR = REPO / "outputs/eda/deepest_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_wells(d: Path):
    return sorted(p.stem.replace("__horizontal_well", "") for p in d.glob("*__horizontal_well.csv"))


def per_well_errors(well_id: str) -> dict:
    df = pd.read_csv(TRAIN_DIR / f"{well_id}__horizontal_well.csv")
    if "TVT" not in df.columns:
        return None
    n = len(df)
    is_vis = df["TVT_input"].notna().values
    if not is_vis.any() or is_vis.all() or is_vis.sum() < 20:
        return None
    boundary = int(np.where(is_vis)[0].max())
    tvt_true = df["TVT"].values
    md = df["MD"].values
    z = df["Z"].values
    gr = df["GR"].values

    # hidden region
    hidden_mask = ~is_vis
    if hidden_mask.sum() < 10:
        return None
    tvt_h = tvt_true[hidden_mask]
    md_h = md[hidden_mask]
    md_h_off = md_h - md[boundary]

    out = dict(well_id=well_id, n_visible=int(is_vis.sum()), n_hidden=int(hidden_mask.sum()),
               boundary_md=float(md[boundary]))

    # Model 1: last-value
    pred_last = np.full(hidden_mask.sum(), tvt_true[boundary])
    err1 = tvt_h - pred_last
    out["last_rmse"] = float(np.sqrt(np.mean(err1**2)))
    out["last_mae"] = float(np.mean(np.abs(err1)))
    out["last_max_err"] = float(np.max(np.abs(err1)))
    # per-MD-distance error: 100, 500, 1000, 2000, 4000 ft 先の err
    for k_ft in (100, 500, 1000, 2000, 4000):
        mask_k = (md_h_off >= k_ft - 50) & (md_h_off < k_ft + 50)
        if mask_k.any():
            out[f"last_err_at_{k_ft}"] = float(np.mean(err1[mask_k]))  # signed
            out[f"last_abs_at_{k_ft}"] = float(np.mean(np.abs(err1[mask_k])))
        else:
            out[f"last_err_at_{k_ft}"] = np.nan
            out[f"last_abs_at_{k_ft}"] = np.nan

    # Model 2: linear extrap from tail-100
    if is_vis.sum() >= 100:
        idx_vis = np.where(is_vis)[0]
        tail_idx = idx_vis[-100:]
        md_tail = md[tail_idx]
        tvt_tail = tvt_true[tail_idx]
        try:
            slope, intercept = np.polyfit(md_tail, tvt_tail, 1)
            pred_lin = slope * md_h + intercept
            err2 = tvt_h - pred_lin
            out["lin_rmse"] = float(np.sqrt(np.mean(err2**2)))
            out["lin_mae"] = float(np.mean(np.abs(err2)))
            out["lin_slope"] = float(slope)
        except Exception:
            out["lin_rmse"] = np.nan
            out["lin_mae"] = np.nan
            out["lin_slope"] = np.nan
    else:
        out["lin_rmse"] = np.nan
        out["lin_mae"] = np.nan
        out["lin_slope"] = np.nan

    # Model 3: tvt_formula = -Z + ANCC_median + b_well_median (= per-well best-case)
    if "ANCC" in df.columns:
        ancc = df["ANCC"].values
        # b_well_median from visible
        b_v = tvt_true[is_vis] + z[is_vis] - ancc[is_vis]
        b_med = float(np.nanmedian(b_v))
        # hidden で ANCC が ある場合 (= train であれば全部ある) を使う
        pred_form = -z[hidden_mask] + ancc[hidden_mask] + b_med
        err3 = tvt_h - pred_form
        out["formula_rmse"] = float(np.sqrt(np.mean(err3**2)))
        out["formula_mae"] = float(np.mean(np.abs(err3)))
        out["formula_max_err"] = float(np.max(np.abs(err3)))
    else:
        out["formula_rmse"] = np.nan
        out["formula_mae"] = np.nan
        out["formula_max_err"] = np.nan

    # Model 4: typewell GR-NCC match
    tw_path = TRAIN_DIR / f"{well_id}__typewell.csv"
    if tw_path.exists():
        tw = pd.read_csv(tw_path).sort_values("TVT").reset_index(drop=True)
        if len(tw) >= 50 and tw["GR"].notna().sum() >= 50:
            tw_tvt = tw["TVT"].values
            tw_gr = tw["GR"].values
            # visible 部の GR から typewell 上の最良 alignment を window NCC で探す
            # 簡易: hidden 各点で GR 値を typewell GR profile に対して NCC search
            # 31-point GR window around hidden point → typewell の対応 TVT を返す
            gr_h = gr[hidden_mask]
            # 各 hidden 点での GR を tw_gr 全体と比較して最 GR-近い TVT を採用 (= naive NCC)
            preds = np.full(hidden_mask.sum(), np.nan)
            win = 15
            gr_full = gr.copy()
            hidden_idx = np.where(hidden_mask)[0]
            tw_gr_clean = tw_gr[~np.isnan(tw_gr)]
            tw_tvt_clean = tw_tvt[~np.isnan(tw_gr)]
            for ii, idx in enumerate(hidden_idx):
                lo = max(0, idx - win)
                hi = min(len(gr_full), idx + win + 1)
                local = gr_full[lo:hi]
                local = local[~np.isnan(local)]
                if len(local) < 5 or len(tw_gr_clean) < len(local):
                    continue
                # rolling mean correlation
                wn = len(local)
                tw_smooth = pd.Series(tw_gr_clean).rolling(wn, center=True).mean().values
                # 簡易: 直接 GR 値の minabsdiff で TVT 推定 (rough NCC)
                local_mean = np.nanmean(local)
                diffs = np.abs(tw_gr_clean - local_mean)
                best = int(np.argmin(diffs))
                preds[ii] = tw_tvt_clean[best]
            valid = ~np.isnan(preds)
            if valid.sum() >= 10:
                err4 = tvt_h[valid] - preds[valid]
                out["tw_match_rmse"] = float(np.sqrt(np.mean(err4**2)))
                out["tw_match_mae"] = float(np.mean(np.abs(err4)))
                out["tw_match_pct"] = float(valid.mean())
            else:
                out["tw_match_rmse"] = np.nan
                out["tw_match_mae"] = np.nan
                out["tw_match_pct"] = float(valid.mean())
        else:
            out["tw_match_rmse"] = np.nan
            out["tw_match_mae"] = np.nan
            out["tw_match_pct"] = np.nan
    else:
        out["tw_match_rmse"] = np.nan
        out["tw_match_mae"] = np.nan
        out["tw_match_pct"] = np.nan

    return out


def main():
    train_wells = list_wells(TRAIN_DIR)
    print(f"train wells: {len(train_wells)}")
    rows = []
    for i, w in enumerate(train_wells):
        if i % 50 == 0:
            print(f"  [{i:4d}/{len(train_wells)}] {w}")
        try:
            r = per_well_errors(w)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR {w}: {e}")
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "pseudo-test-errors.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df)} rows -> {out_path}")

    # summary
    summary = {}
    for col in df.columns:
        if col == "well_id":
            continue
        s = df[col].dropna()
        if len(s) == 0 or not np.issubdtype(s.dtype, np.number):
            continue
        summary[col] = dict(
            n=int(len(s)),
            mean=float(s.mean()),
            std=float(s.std()),
            p10=float(s.quantile(0.10)),
            p50=float(s.quantile(0.50)),
            p90=float(s.quantile(0.90)),
            p99=float(s.quantile(0.99)),
            max=float(s.max()),
        )
    (OUT_DIR / "summary-pseudo-test.json").write_text(json.dumps(summary, indent=2))

    # === 主要な観察 ===
    print("\n=== RMSE 分布 (per well, pseudo-test) ===")
    for m in ("last_rmse", "lin_rmse", "formula_rmse", "tw_match_rmse"):
        if m in df.columns:
            s = df[m].dropna()
            print(f"  {m}: n={len(s)}, p10={s.quantile(0.1):.3f}, p50={s.quantile(0.5):.3f}, p90={s.quantile(0.9):.3f}, p99={s.quantile(0.99):.3f}, max={s.max():.3f}")

    print("\n=== 'failure wells' = last_rmse > p99 ===")
    if "last_rmse" in df.columns:
        thresh = df["last_rmse"].quantile(0.99)
        fail = df[df["last_rmse"] > thresh].sort_values("last_rmse", ascending=False)
        print(fail[["well_id", "n_hidden", "last_rmse", "last_max_err", "lin_rmse",
                     "formula_rmse", "lin_slope"]].to_string())

    print("\n=== MD-distance dependent abs error mean ===")
    for k in (100, 500, 1000, 2000, 4000):
        col = f"last_abs_at_{k}"
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {k} ft 先: n={len(s)}, p50={s.quantile(0.5):.3f}, p90={s.quantile(0.9):.3f}, max={s.max():.3f}")


if __name__ == "__main__":
    main()
