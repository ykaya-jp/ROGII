"""Aggregate per-(exp, cv) kernel outputs into a single oof_table.parquet
that ``tools/measure_cv_lb_correlation.py`` consumes.

Expected input directory layout (= produced by scripts/regenerate_oof.py):

    outputs/oof/cv-lb-correlation/
        exp007_edge_q_m__cv-baseline/
            oof_predictions_self.parquet   ← kernel-saved self OOF
            submission.csv                 ← kernel-saved test predictions
            *.log                          ← kernel runtime log
        exp007_edge_q_m__cv-C1/
            ...
        ...

The script:
    1. Walks every <exp>__cv-<cv>/ subdir.
    2. Reads ``oof_predictions_self.parquet`` (= patched kernel output).
       Schema: well, y_true, own_oof_blend, fold_id, [own_oof_<key>...]
    3. Computes ``oof_rmse = sqrt(mean((y_true - own_oof_blend)**2))``.
    4. Looks up ``public_lb`` from a CLI-supplied JSON (or PUBLIC_LB defaults).
    5. Optionally computes ``sigma_fold`` = std(per-fold RMSE) for AC-10.
    6. Writes ``oof_table.parquet`` to ``outputs/oof/cv-lb-correlation/``.

NOTE: As of 2026-05-11, none of the 4 self-base kernels emit
``oof_predictions_self.parquet`` yet — that patch is the remaining work in
Phase 2.1.f. For now this script is a *contract* describing what the
kernel-side patch must produce. Run with --probe to enumerate which
output dirs exist but are missing the parquet.

Usage:
    .venv/bin/python scripts/aggregate_oof.py --probe
    .venv/bin/python scripts/aggregate_oof.py --build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "outputs" / "oof" / "cv-lb-correlation"
OOF_TABLE = OOF_DIR / "oof_table.parquet"

PUBLIC_LB = {
    "exp002_lgb":            14.695,
    "exp003_lgb":            17.510,
    "exp005_cache_blend":    10.317,
    "exp005_cache_blend_v2": 10.203,
    "exp005_cache_blend_v3": 10.387,
    "exp006_tabicl_pflite":  10.503,
    "exp007_edge_q_m":       10.677,
    "exp008_case_d_kalman":   9.957,   # v2 = current best
    "exp009_case_e_edge_o":  None,     # PENDING
    "exp010_fold_reform":    None,     # kernel still RUNNING
}


def parse_dir_name(name: str) -> tuple[str, str] | None:
    """Return (exp, cv) from `<exp>__cv-<cv>`, or None if it doesn't fit."""
    marker = "__cv-"
    if marker not in name:
        return None
    exp, cv = name.split(marker, 1)
    return exp, cv


def aggregate_one_dir(d: Path) -> dict | None:
    parsed = parse_dir_name(d.name)
    if parsed is None:
        return None
    exp, cv = parsed
    oof_parquet = d / "oof_predictions_self.parquet"
    if not oof_parquet.exists():
        return {"exp": exp, "cv": cv, "status": "MISSING_OOF_PARQUET", "dir": str(d)}
    try:
        df = pd.read_parquet(oof_parquet)
    except Exception as e:
        return {"exp": exp, "cv": cv, "status": f"READ_FAILED:{type(e).__name__}",
                "dir": str(d)}
    needed = {"y_true", "own_oof_blend"}
    missing = needed - set(df.columns)
    if missing:
        return {"exp": exp, "cv": cv, "status": f"MISSING_COLS:{sorted(missing)}",
                "dir": str(d)}
    y_true = df["y_true"].astype(float).values
    y_pred = df["own_oof_blend"].astype(float).values
    oof_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sigma_fold = None
    if "fold_id" in df.columns:
        per_fold = []
        for k in sorted(df["fold_id"].unique()):
            mask = df["fold_id"] == k
            if mask.sum() < 5:
                continue
            per_fold.append(
                float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
            )
        if len(per_fold) >= 3:
            sigma_fold = float(np.std(per_fold))
    return {
        "exp": exp,
        "cv": cv,
        "status": "OK",
        "n_rows": int(len(df)),
        "oof_rmse": oof_rmse,
        "public_lb": PUBLIC_LB.get(exp),
        "sigma_fold": sigma_fold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true",
                        help="list output dirs and report which are ready / missing")
    parser.add_argument("--build", action="store_true",
                        help="aggregate OOFs into oof_table.parquet")
    parser.add_argument("--public-lb-json", default=None,
                        help="optional JSON file overriding PUBLIC_LB defaults")
    args = parser.parse_args()
    if not (args.probe or args.build):
        parser.error("specify --probe or --build")

    if args.public_lb_json:
        PUBLIC_LB.update(json.loads(Path(args.public_lb_json).read_text()))

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    subdirs = sorted(p for p in OOF_DIR.iterdir() if p.is_dir())
    print(f"== aggregate_oof ==  scanning {OOF_DIR}  ({len(subdirs)} subdirs)")

    results = [aggregate_one_dir(d) for d in subdirs]
    results = [r for r in results if r is not None]

    if args.probe:
        for r in results:
            line = f"  {r['exp']:30s}  cv={r['cv']:8s}  status={r['status']}"
            if r["status"] == "OK":
                line += f"  rmse={r['oof_rmse']:.4f}  n={r['n_rows']}"
            print(line)
        n_ok = sum(1 for r in results if r["status"] == "OK")
        print(f"\n  {n_ok}/{len(results)} dirs have a valid oof_predictions_self.parquet")
        return

    # --build
    rows_ok = [r for r in results if r["status"] == "OK"]
    if not rows_ok:
        print("ERROR: no valid OOF parquets found. Run --probe to diagnose.",
              file=sys.stderr)
        sys.exit(2)
    df = pd.DataFrame([{k: r[k] for k in ("exp", "cv", "n_rows", "oof_rmse",
                                          "public_lb", "sigma_fold")}
                       for r in rows_ok])
    df.to_parquet(OOF_TABLE)
    print(f"  wrote {len(df)} rows to {OOF_TABLE.relative_to(REPO_ROOT)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
