"""Submission Data Deep Mining (2026-05-11).

Extracts per-fold OOF RMSE, Ridge weights, parameters, runtime breakdown,
per-well submission predictions, visible TVT_input residuals, and inter-well
prediction variance for all SCORED submissions whose kernel log is available.

Outputs JSON tables consumed by `docs/research/submission-data-mining.dense.md`.

Sources:
- outputs/kernel_logs/exp005/rogii-exp005-cache-blend.log   (= exp005 v2, Edge S)
- outputs/kernel_logs/exp006/rogii-exp006-tabicl-pflite.log (= exp006)
- outputs/kernel_logs/exp007/rogii-exp007-edge-qm.log       (= exp007)
- submissions/exp005/rogii-exp005-cache-blend.log           (= exp005 v1, no Edge S)
- submissions/exp005/submission.csv                          (= exp005 v1 predictions)
- data/raw/test/<well>__horizontal_well.csv                  (= visible TVT_input)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
OUT_DIR = ROOT / "outputs" / "eda" / "submission_mining"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILES = {
    "exp005_v1": ROOT / "submissions" / "exp005" / "rogii-exp005-cache-blend.log",
    "exp005_v2": ROOT / "outputs" / "kernel_logs" / "exp005" / "rogii-exp005-cache-blend.log",
    "exp006":    ROOT / "outputs" / "kernel_logs" / "exp006" / "rogii-exp006-tabicl-pflite.log",
    "exp007":    ROOT / "outputs" / "kernel_logs" / "exp007" / "rogii-exp007-edge-qm.log",
}

SUB_FILES = {
    "exp002":     ROOT / "submissions" / "exp002_lgb.csv",
    "exp005_v1":  ROOT / "submissions" / "exp005" / "submission.csv",
}

LB_VALUES = {
    "exp002": 14.695,
    "exp003": 17.510,
    "exp005_v1": 10.317,
    "exp006": 10.503,
    "exp007": 10.677,
    "exp005_v2": 10.203,
}


def load_log_events(path: Path) -> list[dict]:
    text = path.read_text()
    return json.loads(text)


def collect_log_messages(events: list[dict]) -> list[tuple[float, str]]:
    """Return list of (time, msg) sorted by time."""
    out = []
    for ev in events:
        msg = ev["data"].rstrip("\n")
        out.append((ev["time"], msg))
    return out


PER_FOLD_RE = re.compile(
    r"fold(?P<fold>\d+)\s+(?P<base>\w+)(?:\s+lr=(?P<lr>[\d.]+)\s+ne_used=(?P<ne>\d+))?:?\s*va RMSE=(?P<rmse>[\d.]+)"
)
EARLY_STOP_RE = re.compile(r"Early stopping, best iteration is:")
BEST_ITER_RE = re.compile(r"\[(?P<iter>\d+)\]\s*valid_0's l2:\s*(?P<l2>[\d.]+)")
OOF_RE = re.compile(r"own OOF (?P<base>\w+): RMSE=(?P<rmse>[\d.]+)")
RIDGE_WT_RE = re.compile(r"Ridge weights\s*:\s*({.+})")
RIDGE_OOF_RE = re.compile(r"Ridge \d+-stk OOF RMSE:\s*([\d.]+)")
SIMPLE_AVG_RE = re.compile(r"Simple \d+-avg OOF RMSE:\s*([\d.]+)")
DELTA_RE = re.compile(r"(?P<base>\w+)\s+delta range:\s*\[?(-?[\d.]+)[–\-,](-?[\d.]+)\]?")
EDGE_S_DIFF_MEAN = re.compile(r"\[Edge S\]\s+diff abs mean:\s*([\d.]+)")
EDGE_S_DIFF_MAX = re.compile(r"\[Edge S\]\s+diff abs max:\s*([\d.]+)")
EDGE_S_UNIQUE = re.compile(r"\[Edge S\]\s+unique tvt count:\s*(\d+)")
PHASE_DONE_RE = re.compile(r"✔\s+(.+?)\s+—\s+([\d.]+)s")


def parse_log_metrics(events: list[dict]) -> dict:
    msgs = collect_log_messages(events)

    per_fold = []  # list of dict(fold, base, lr, ne_used, va_rmse, t)
    own_oof = {}
    ridge_oof = None
    ridge_simple_avg = None
    ridge_weights = None
    delta_ranges = {}  # base -> (lo, hi)
    edge_s = {}
    phases = []  # list of (name, seconds)
    total_runtime_s = msgs[-1][0] if msgs else 0.0
    last_best_iter = None  # tracks LGB best iteration for matching with next va RMSE
    last_best_l2 = None

    for t, msg in msgs:
        if "Early stopping" in msg:
            continue
        m = BEST_ITER_RE.search(msg)
        if m:
            last_best_iter = int(m.group("iter"))
            last_best_l2 = float(m.group("l2"))
            continue
        m = PER_FOLD_RE.search(msg)
        if m and "va RMSE=" in msg:
            d = m.groupdict()
            entry = {
                "fold": int(d["fold"]),
                "base": d["base"],
                "lr": float(d["lr"]) if d["lr"] else None,
                "ne_used": int(d["ne"]) if d["ne"] else None,
                "va_rmse": float(d["rmse"]),
                "t_seconds": t,
            }
            # attach best l2 if this is an LGB log line preceded by best_iter
            if d["base"].startswith("lgb") and last_best_l2 is not None:
                entry["best_l2_train_valid"] = last_best_l2
                last_best_l2 = None
                last_best_iter = None
            per_fold.append(entry)
            continue
        m = OOF_RE.search(msg)
        if m:
            own_oof[m.group("base")] = float(m.group("rmse"))
            continue
        m = RIDGE_OOF_RE.search(msg)
        if m:
            ridge_oof = float(m.group(1))
            continue
        m = SIMPLE_AVG_RE.search(msg)
        if m:
            ridge_simple_avg = float(m.group(1))
            continue
        m = RIDGE_WT_RE.search(msg)
        if m:
            raw = m.group(1)
            # convert np.float32(...) → number
            cleaned = re.sub(r"np\.float32\(([\d.\-]+)\)", r"\1", raw)
            cleaned = cleaned.replace("'", '"')
            try:
                ridge_weights = json.loads(cleaned)
            except Exception:
                ridge_weights = {"raw": raw}
            continue
        m = DELTA_RE.search(msg)
        if m and "delta range" in msg:
            base = m.group("base")
            lo = float(m.group(2))
            hi = float(m.group(3))
            if base not in delta_ranges:
                delta_ranges[base] = (lo, hi)
            continue
        m = EDGE_S_DIFF_MEAN.search(msg)
        if m:
            edge_s["diff_abs_mean"] = float(m.group(1))
            continue
        m = EDGE_S_DIFF_MAX.search(msg)
        if m:
            edge_s["diff_abs_max"] = float(m.group(1))
            continue
        m = EDGE_S_UNIQUE.search(msg)
        if m:
            edge_s["unique_tvt_count"] = int(m.group(1))
            continue
        m = PHASE_DONE_RE.search(msg)
        if m:
            phases.append({"name": m.group(1), "seconds": float(m.group(2))})
            continue

    return {
        "total_runtime_s": total_runtime_s,
        "per_fold": per_fold,
        "own_oof": own_oof,
        "ridge_oof": ridge_oof,
        "ridge_simple_avg": ridge_simple_avg,
        "ridge_weights": ridge_weights,
        "delta_ranges": delta_ranges,
        "edge_s": edge_s,
        "phases": phases,
    }


def per_fold_stats(per_fold: list[dict]) -> dict:
    """Aggregate per-fold RMSE → per-fold mean across bases + per-base mean across folds."""
    df = pd.DataFrame(per_fold)
    if df.empty:
        return {}
    by_fold = df.groupby("fold")["va_rmse"].agg(["mean", "std", "min", "max"]).round(4)
    by_base = df.groupby("base")["va_rmse"].agg(["mean", "std", "min", "max"]).round(4)
    return {
        "by_fold": by_fold.reset_index().to_dict(orient="records"),
        "by_base": by_base.reset_index().to_dict(orient="records"),
    }


def load_visible_tvt() -> dict[str, pd.DataFrame]:
    """Return per-well dataframe of (MD, TVT_input, has_visible_flag)."""
    out = {}
    for well in ["000d7d20", "00bbac68", "00e12e8b"]:
        p = ROOT / "data" / "raw" / "test" / f"{well}__horizontal_well.csv"
        df = pd.read_csv(p)
        df["has_visible"] = df["TVT_input"].notna()
        out[well] = df
    return out


def build_id_index(visible: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """sub id format = <well>_<row_index> (0-based row in horizontal_well.csv)."""
    rows = []
    for well, df in visible.items():
        for row_idx, r in df.iterrows():
            rows.append({
                "id": f"{well}_{row_idx}",
                "well": well,
                "row_idx": int(row_idx),
                "md": float(r["MD"]),
                "tvt_input": r["TVT_input"],
                "has_visible": bool(r["has_visible"]),
            })
    return pd.DataFrame(rows)


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_well_visible_anchor(visible_df: dict[str, pd.DataFrame]) -> dict:
    """For each well, compute last visible MD/TVT_input, hidden length, etc."""
    out = {}
    for well, df in visible_df.items():
        vis_mask = df["TVT_input"].notna()
        vis = df[vis_mask]
        last_md = float(vis["MD"].max())
        last_tvt = float(vis.loc[vis["MD"].idxmax(), "TVT_input"])
        last_visible_row = int(vis.index.max())
        first_hidden_row = last_visible_row + 1
        max_md = float(df["MD"].max())
        out[well] = {
            "n_rows": int(len(df)),
            "n_visible": int(vis_mask.sum()),
            "n_hidden": int((~vis_mask).sum()),
            "visible_last_md": last_md,
            "visible_last_tvt": last_tvt,
            "last_visible_row": last_visible_row,
            "first_hidden_row": first_hidden_row,
            "hidden_max_md": max_md,
            "hidden_len_ft": int(max_md - last_md),
        }
    return out


def cross_well_pred_stats(sub_df: pd.DataFrame, id_idx: pd.DataFrame, anchors: dict) -> dict:
    """Per-well hidden-region prediction summary (sub only covers hidden rows).

    Key metrics:
    - prediction range (min / max / median)
    - monotonic delta (= predict trend = sign of dtvt)
    - tail_delta_vs_visible_last (= last_pred - visible_last_tvt; proxy of cumulative drift)
    - first_hidden_pred_vs_visible_last (= continuity at boundary)
    """
    merged = sub_df.merge(id_idx, on="id", how="left")
    out = {}
    for well, g in merged.groupby("well"):
        g = g.sort_values("row_idx").reset_index(drop=True)
        anchor = anchors[well]
        visible_last_tvt = anchor["visible_last_tvt"]
        hidden_len_ft = anchor["hidden_len_ft"]
        n = len(g)
        if n == 0:
            continue
        first_pred = float(g.iloc[0]["tvt"])
        last_pred = float(g.iloc[-1]["tvt"])
        dtvt = g["tvt"].diff().dropna()
        # On-grid (Edge S signal): unique TVT count vs n
        unique_tvt = int(g["tvt"].round(2).nunique())
        out[well] = {
            "n_hidden_rows": int(n),
            "hidden_len_ft": hidden_len_ft,
            "pred_min": float(g["tvt"].min()),
            "pred_max": float(g["tvt"].max()),
            "pred_median": float(g["tvt"].median()),
            "pred_range": float(g["tvt"].max() - g["tvt"].min()),
            "visible_last_tvt": visible_last_tvt,
            "first_hidden_pred": first_pred,
            "boundary_jump": first_pred - visible_last_tvt,
            "last_hidden_pred": last_pred,
            "tail_delta_vs_visible_last": last_pred - visible_last_tvt,
            "dtvt_mean": float(dtvt.mean()),
            "dtvt_std": float(dtvt.std()),
            "dtvt_min": float(dtvt.min()),
            "dtvt_max": float(dtvt.max()),
            "dtvt_p95_abs": float(np.percentile(dtvt.abs(), 95)),
            "unique_tvt_round2": unique_tvt,
            "on_grid_ratio_round2": unique_tvt / n,
        }
    return out


def cross_sub_diff(sub_a: pd.DataFrame, sub_b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Compare two submission CSVs id-wise."""
    merged = sub_a.merge(sub_b, on="id", suffixes=(f"_{name_a}", f"_{name_b}"))
    d = merged[f"tvt_{name_a}"] - merged[f"tvt_{name_b}"]
    return {
        "n": int(len(merged)),
        "diff_mean": float(d.mean()),
        "diff_std": float(d.std()),
        "diff_min": float(d.min()),
        "diff_max": float(d.max()),
        "diff_abs_mean": float(d.abs().mean()),
        "diff_abs_p95": float(np.percentile(d.abs(), 95)),
        "diff_abs_max": float(d.abs().max()),
    }


def main() -> None:
    summary = {}

    # 1. Parse all kernel logs
    log_metrics = {}
    for sub_id, log_path in LOG_FILES.items():
        if not log_path.exists():
            log_metrics[sub_id] = {"error": "log_not_found"}
            continue
        events = load_log_events(log_path)
        metrics = parse_log_metrics(events)
        metrics["per_fold_stats"] = per_fold_stats(metrics["per_fold"])
        log_metrics[sub_id] = metrics
    summary["log_metrics"] = log_metrics

    # 2. Visible TVT_input + id index
    visible = load_visible_tvt()
    id_idx = build_id_index(visible)
    anchors = compute_well_visible_anchor(visible)
    summary["well_anchors"] = anchors

    # 3. Per-well prediction stats for available submissions
    sub_dfs = {}
    sub_stats = {}
    for sub_id, sub_path in SUB_FILES.items():
        if sub_path.exists():
            sd = load_submission(sub_path)
            sub_dfs[sub_id] = sd
            sub_stats[sub_id] = cross_well_pred_stats(sd, id_idx, anchors)
    summary["per_well_pred_stats"] = sub_stats

    # 4. Cross-sub diff (only what we have on disk)
    cross_diff = {}
    if "exp002" in sub_dfs and "exp005_v1" in sub_dfs:
        cross_diff["exp005_v1_vs_exp002"] = cross_sub_diff(
            sub_dfs["exp005_v1"], sub_dfs["exp002"], "exp005_v1", "exp002"
        )
    summary["cross_sub_diff"] = cross_diff

    # 5. CV-LB analysis
    cv_lb = {}
    for sub_id, metrics in log_metrics.items():
        if "error" in metrics:
            continue
        lb = LB_VALUES.get(sub_id)
        rec = {
            "lb": lb,
            "ridge_oof": metrics.get("ridge_oof"),
            "ridge_simple_avg": metrics.get("ridge_simple_avg"),
            "own_oof": metrics.get("own_oof"),
            "ridge_weights": metrics.get("ridge_weights"),
        }
        if metrics.get("ridge_oof") is not None and lb is not None:
            rec["cv_lb_diff"] = round(lb - metrics["ridge_oof"], 4)
        cv_lb[sub_id] = rec
    summary["cv_lb"] = cv_lb

    # Write out
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_path}")

    # Pretty print key tables
    print("\n=== Per-fold RMSE matrix (exp007) ===")
    if "exp007" in log_metrics and "per_fold_stats" in log_metrics["exp007"]:
        pfs = log_metrics["exp007"]["per_fold_stats"]
        for r in pfs.get("by_fold", []):
            print(r)
        print("--- by_base ---")
        for r in pfs.get("by_base", []):
            print(r)

    print("\n=== Edge S effect (exp005 v2) ===")
    print(log_metrics.get("exp005_v2", {}).get("edge_s"))

    print("\n=== Ridge weights (exp007) ===")
    print(log_metrics.get("exp007", {}).get("ridge_weights"))

    print("\n=== Delta ranges per sub ===")
    for sub_id, m in log_metrics.items():
        if "delta_ranges" in m:
            print(f"  {sub_id}: {m['delta_ranges']}")

    print("\n=== Per-well anchors ===")
    for w, a in summary["well_anchors"].items():
        print(f"  {w}: {a}")

    print("\n=== Per-well prediction stats (exp005 v1) ===")
    for w, st in summary["per_well_pred_stats"].get("exp005_v1", {}).items():
        print(f"  {w}: {st}")


if __name__ == "__main__":
    main()
