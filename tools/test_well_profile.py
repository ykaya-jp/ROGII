"""Test wells (3 件) の per-well metric 分布を train wells と対比する。

既存 deepest EDA parquet (outputs/eda/deepest_eda/*) を流用し、
- test 3 wells の各 metric の値
- train 分布における percentile rank (= 0-100%)
を出力。 CV split 候補設計の前提資料。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]
EDA_DIR = Path("outputs/eda/deepest_eda")
FP_DIR = Path("outputs/eda/first_principles")
PER_WELL_TOP = Path("outputs/eda/per-well-stats.parquet")

PARQUETS = [
    ("high_order", EDA_DIR / "per-well-high-order.parquet"),
    ("outlier", EDA_DIR / "per-well-outlier.parquet"),
    ("tail_stats", EDA_DIR / "tail-stats.parquet"),
    ("gr_pca", EDA_DIR / "gr-pca.parquet"),
    ("typewell_stats", EDA_DIR / "typewell-stats.parquet"),
    ("formation_uniqueness", EDA_DIR / "formation-uniqueness.parquet"),
    ("b_cluster_xy", EDA_DIR / "b-cluster-xy.parquet"),
    ("per_well_top", PER_WELL_TOP),
]


def find_well_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ("well", "well_id", "wellname", "wellid"):
            return c
    return None


def main() -> None:
    out_lines: list[str] = []
    out_lines.append(f"# Test wells profile vs train distribution\n")
    out_lines.append(f"test wells: {TEST_WELLS}\n")
    out_lines.append("\n各 parquet で test 3 wells の値と train 内 percentile rank を併記。\n")

    for name, p in PARQUETS:
        if not p.exists():
            out_lines.append(f"\n## {name} — {p} 不在 (skip)\n")
            continue
        df = pd.read_parquet(p)
        well_col = find_well_col(df)
        if well_col is None and len(df) and isinstance(df.iloc[0, 0], str) and len(df.iloc[0, 0]) == 8:
            well_col = df.columns[0]
        out_lines.append(f"\n## {name} ({p.name}) — shape={df.shape}, well_col={well_col!r}\n")
        if well_col is None:
            out_lines.append(f"well_col 不在、skip\n")
            continue

        test_df = df[df[well_col].isin(TEST_WELLS)].copy()
        train_df = df[~df[well_col].isin(TEST_WELLS)].copy()
        out_lines.append(
            f"test rows found: {len(test_df)} (= {test_df[well_col].tolist()})\n"
        )
        out_lines.append(f"train rows: {len(train_df)}\n")

        numeric_cols = [
            c for c in df.columns
            if c != well_col
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]
        if not numeric_cols or len(test_df) == 0:
            continue

        # build table: metric | test_w1 | test_w2 | test_w3 | train p10 | p50 | p90 | test pct ranks
        rows: list[dict] = []
        for c in numeric_cols:
            train_vals = train_df[c].dropna()
            if len(train_vals) < 50:
                continue
            test_vals = []
            test_pcts = []
            for w in TEST_WELLS:
                sub = test_df.loc[test_df[well_col] == w, c]
                v = sub.iloc[0] if len(sub) and pd.notna(sub.iloc[0]) else None
                test_vals.append(v)
                if v is not None:
                    pct = (train_vals < v).sum() / len(train_vals) * 100
                    test_pcts.append(pct)
                else:
                    test_pcts.append(None)
            rows.append({
                "metric": c,
                "w0": test_vals[0],
                "w1": test_vals[1],
                "w2": test_vals[2],
                "pct0": test_pcts[0],
                "pct1": test_pcts[1],
                "pct2": test_pcts[2],
                "tr_p10": float(train_vals.quantile(0.10)),
                "tr_p50": float(train_vals.quantile(0.50)),
                "tr_p90": float(train_vals.quantile(0.90)),
            })

        # render as markdown table (cap 80 rows per parquet)
        rows_sorted = sorted(
            rows,
            key=lambda r: max(
                abs((r["pct0"] or 50) - 50),
                abs((r["pct1"] or 50) - 50),
                abs((r["pct2"] or 50) - 50),
            ),
            reverse=True,
        )
        out_lines.append(
            "\n| metric | w0=000d7d20 | w1=00bbac68 | w2=00e12e8b | tr p10 | tr p50 | tr p90 | w0 pct | w1 pct | w2 pct |\n"
        )
        out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        def fmt(v):
            if v is None:
                return "-"
            if abs(v) >= 1000:
                return f"{v:.0f}"
            if abs(v) >= 10:
                return f"{v:.2f}"
            return f"{v:.4f}"
        def fmtpct(v):
            return "-" if v is None else f"{v:.0f}%"
        for r in rows_sorted[:80]:
            out_lines.append(
                f"| `{r['metric']}` | {fmt(r['w0'])} | {fmt(r['w1'])} | {fmt(r['w2'])} | "
                f"{fmt(r['tr_p10'])} | {fmt(r['tr_p50'])} | {fmt(r['tr_p90'])} | "
                f"{fmtpct(r['pct0'])} | {fmtpct(r['pct1'])} | {fmtpct(r['pct2'])} |\n"
            )

    out_dir = Path("docs/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2026-05-11-test-distribution.dense.md"
    out_path.write_text("".join(out_lines))
    print(f"wrote: {out_path} ({sum(len(l) for l in out_lines)} chars, {len(out_lines)} lines)")


if __name__ == "__main__":
    main()
