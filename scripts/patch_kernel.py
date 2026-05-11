"""Inject D1 cluster map + LGB/CB best params into the kernel script.

Reads:
  - per-well-outlier.parquet -> D1 cluster map literal
  - outputs/exp004/hpo_lgb.json -> LGB best params
  - outputs/exp004/hpo_cb.json  -> CB best params
  - outputs/exp004/loss_best.json -> objective name + huber alpha

Patches:
  - %%D1_MAP_INSERT%% marker in the kernel script with `D1_CLUSTER_MAP = {...}`
  - %%LGB_BEST_INSERT%% marker with LGB_BASE / LGB_CONFIGS dict update
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KERNEL = ROOT / "kaggle_kernels/exp004_numba_beam_pf/exp004_numba_beam_pf.py"
PARQUET = Path("/home/yusuke_kaya/projects/kaggle/ROGII/outputs/eda/deepest_eda/per-well-outlier.parquet")
HPO_LGB = ROOT / "outputs/exp004/hpo_lgb.json"
HPO_CB = ROOT / "outputs/exp004/hpo_cb.json"
LOSS_BEST = ROOT / "outputs/exp004/loss_best.json"


def build_d1_literal() -> str:
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    train = df[df["split"] == "train"].dropna(subset=["b_med"]).copy()
    train["b_round"] = train["b_med"].round(2)
    distinct = sorted(train["b_round"].unique())
    b_to_cluster = {b: i for i, b in enumerate(distinct)}
    mapping = {}
    for wid, b in zip(train["well_id"].to_list(), train["b_round"].to_list(), strict=True):
        mapping[str(wid)] = b_to_cluster[b]
    out = ["D1_CLUSTER_MAP = {"]
    for wid, cid in sorted(mapping.items()):
        out.append(f'    "{wid}": {cid},')
    out.append("}")
    return "\n".join(out)


def build_lgb_block() -> str:
    """Render LGB/CB best params + objective. If HPO output missing, keep defaults."""
    lines = []
    if HPO_LGB.exists():
        with open(HPO_LGB) as f:
            lgb_best = json.load(f)
        bp = lgb_best["best_params"]
        objective = "regression"
        huber_alpha = None
        if LOSS_BEST.exists():
            with open(LOSS_BEST) as f:
                lb = json.load(f)
            objective = lb.get("best_objective", "regression")
            huber_alpha = lb.get("best_huber_alpha", None)
        lines.append("# Optuna HPO best params (AC-14) + loss ablation (AC-15) applied.")
        lines.append(f"LGB_HPO_BEST = {bp!r}")
        if objective != "regression":
            lines.append(f'_HPO_OBJECTIVE = {objective!r}')
            if huber_alpha is not None:
                lines.append(f"_HPO_HUBER_ALPHA = {huber_alpha}")
        else:
            lines.append('_HPO_OBJECTIVE = "regression"')
            lines.append("_HPO_HUBER_ALPHA = None")
    else:
        lines.append("# (HPO not yet run; using baseline params.)")
        lines.append("LGB_HPO_BEST = None")
        lines.append('_HPO_OBJECTIVE = "regression"')
        lines.append("_HPO_HUBER_ALPHA = None")

    if HPO_CB.exists():
        with open(HPO_CB) as f:
            cb_best = json.load(f)
        lines.append(f"CB_HPO_BEST = {cb_best['best_params']!r}")
    else:
        lines.append("CB_HPO_BEST = None")
    return "\n".join(lines)


def patch(text: str) -> str:
    d1_block = build_d1_literal()
    lgb_block = build_lgb_block()
    text = text.replace("# %%D1_MAP_INSERT%%", d1_block)
    text = text.replace("# %%LGB_BEST_INSERT%%", lgb_block)
    return text


def main() -> int:
    if not KERNEL.exists():
        print(f"ERROR: kernel script missing at {KERNEL}", flush=True)
        return 1
    src = KERNEL.read_text()
    if "# %%D1_MAP_INSERT%%" not in src:
        print("WARN: kernel script already patched (no D1 marker)", flush=True)
    out = patch(src)
    KERNEL.write_text(out)
    print(f"Patched {KERNEL} ({len(out)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
