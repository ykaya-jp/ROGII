"""EDA aggregator — walk all wells in data/raw/{train,test}/ and compute per-well stats.

Run:
    uv run python -m rogii.eda

Outputs:
    outputs/eda/per-well-stats.parquet — one row per well, both splits combined
    outputs/eda/aggregate-summary.json — cross-cutting statistics for docs/research/data-spec.dense.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")
OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)


def _consecutive_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_inclusive) for runs where mask is True."""
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends, strict=True))


def per_well_stats(split: str, well: str) -> dict:
    """Compute per-well stats for one (split, well)."""
    base = DATA_RAW / split
    h = pd.read_csv(base / f"{well}__horizontal_well.csv")
    t = pd.read_csv(base / f"{well}__typewell.csv")

    md = h["MD"].to_numpy()
    md_steps = np.diff(md)
    visible_mask = h["TVT_input"].notna().to_numpy()
    hidden_mask = ~visible_mask

    visible_runs = _consecutive_runs(visible_mask)
    hidden_runs = _consecutive_runs(hidden_mask)

    # is the entire hidden region a single trailing block?
    trailing_hidden_only = (
        len(hidden_runs) == 1 and hidden_runs[0][1] == hidden_mask.size - 1 and hidden_mask[-1]
    )

    out = {
        "split": split,
        "well": well,
        # horizontal_well row counts
        "n_rows": len(h),
        "n_visible": int(visible_mask.sum()),
        "n_hidden": int(hidden_mask.sum()),
        "visible_ratio": float(visible_mask.mean()),
        # MD geometry
        "MD_min": float(md.min()),
        "MD_max": float(md.max()),
        "MD_range": float(md.max() - md.min()),
        "MD_step_modal": float(np.bincount(np.round(md_steps).astype(int)).argmax())
        if md_steps.size
        else float("nan"),
        "MD_step_min": float(md_steps.min()) if md_steps.size else float("nan"),
        "MD_step_max": float(md_steps.max()) if md_steps.size else float("nan"),
        "MD_steps_unique_count": int(np.unique(md_steps.round(3)).size) if md_steps.size else 0,
        # eval zone shape
        "n_visible_runs": len(visible_runs),
        "n_hidden_runs": len(hidden_runs),
        "trailing_hidden_only": trailing_hidden_only,
        "first_hidden_idx": int(np.argmax(hidden_mask)) if hidden_mask.any() else -1,
        "first_hidden_md": float(md[np.argmax(hidden_mask)]) if hidden_mask.any() else float("nan"),
        # XYZ
        "X_min": float(h["X"].min()),
        "X_max": float(h["X"].max()),
        "Y_min": float(h["Y"].min()),
        "Y_max": float(h["Y"].max()),
        "Z_min": float(h["Z"].min()),
        "Z_max": float(h["Z"].max()),
        "Z_range": float(h["Z"].max() - h["Z"].min()),
        # GR
        "GR_min": float(h["GR"].min()),
        "GR_max": float(h["GR"].max()),
        "GR_mean": float(h["GR"].mean()),
        "GR_std": float(h["GR"].std()),
        # typewell
        "typewell_n_rows": len(t),
        "typewell_TVT_min": float(t["TVT"].min()),
        "typewell_TVT_max": float(t["TVT"].max()),
        "typewell_GR_mean": float(t["GR"].mean()),
        "typewell_GR_std": float(t["GR"].std()),
        "typewell_TVT_step_modal": float(np.bincount(np.round(np.diff(t["TVT"]) * 10).astype(int)).argmax() / 10)
        if len(t) > 1
        else float("nan"),
    }

    # train-only columns
    if "TVT" in h.columns:
        tvt = h["TVT"].dropna()
        out.update(
            {
                "TVT_min": float(tvt.min()),
                "TVT_max": float(tvt.max()),
                "TVT_range": float(tvt.max() - tvt.min()),
                "TVT_input_matches_TVT_on_visible": bool(
                    np.allclose(
                        h.loc[visible_mask, "TVT_input"].to_numpy(),
                        h.loc[visible_mask, "TVT"].to_numpy(),
                        atol=1e-3,
                    )
                ),
            }
        )
    if "Geology" in t.columns:
        geo = t["Geology"].dropna()
        out["typewell_geology_unique_count"] = int(geo.nunique())
        out["typewell_geology_n_labeled"] = int(geo.size)

    # typewell covers horizontal TVT range?
    if "TVT" in h.columns:
        out["typewell_covers_horizontal_tvt"] = bool(
            t["TVT"].min() <= h["TVT"].min() and t["TVT"].max() >= h["TVT"].max()
        )

    return out


def main() -> None:
    rows = []
    for split in ("train", "test"):
        split_dir = DATA_RAW / split
        if not split_dir.exists():
            continue
        wells = sorted({p.stem.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv")})
        print(f"[{split}] {len(wells)} wells")
        for i, w in enumerate(wells):
            rows.append(per_well_stats(split, w))
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(wells)}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "per-well-stats.parquet", index=False)
    print(f"\nwrote {OUT / 'per-well-stats.parquet'} ({len(df)} rows)")

    # cross-cutting summary
    summary = {}
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        summary[split] = {
            "n_wells": len(sub),
            "MD_step_modal_unique": Counter(sub["MD_step_modal"].round(3).tolist()).most_common(),
            "MD_step_max_max": float(sub["MD_step_max"].max()),
            "MD_step_min_min": float(sub["MD_step_min"].min()),
            "n_rows_min": int(sub["n_rows"].min()),
            "n_rows_max": int(sub["n_rows"].max()),
            "n_rows_p50": int(sub["n_rows"].quantile(0.5)),
            "n_rows_p95": int(sub["n_rows"].quantile(0.95)),
            "visible_ratio_min": float(sub["visible_ratio"].min()),
            "visible_ratio_max": float(sub["visible_ratio"].max()),
            "visible_ratio_p50": float(sub["visible_ratio"].quantile(0.5)),
            "visible_ratio_p25": float(sub["visible_ratio"].quantile(0.25)),
            "visible_ratio_p75": float(sub["visible_ratio"].quantile(0.75)),
            "wells_with_eval_zone": int((sub["n_hidden"] > 0).sum()),
            "wells_with_trailing_hidden_only": int(sub["trailing_hidden_only"].sum()),
            "wells_with_visible_runs_gt_1": int((sub["n_visible_runs"] > 1).sum()),
            "wells_with_hidden_runs_gt_1": int((sub["n_hidden_runs"] > 1).sum()),
            "wells_zero_hidden": int((sub["n_hidden"] == 0).sum()),
            "GR_mean_min": float(sub["GR_mean"].min()),
            "GR_mean_max": float(sub["GR_mean"].max()),
            "GR_mean_p50": float(sub["GR_mean"].quantile(0.5)),
            "GR_std_p50": float(sub["GR_std"].quantile(0.5)),
            "typewell_n_rows_min": int(sub["typewell_n_rows"].min()),
            "typewell_n_rows_max": int(sub["typewell_n_rows"].max()),
            "typewell_n_rows_p50": int(sub["typewell_n_rows"].quantile(0.5)),
            "typewell_TVT_step_modal_unique": Counter(
                sub["typewell_TVT_step_modal"].round(2).tolist()
            ).most_common(5),
            "Z_range_p50": float(sub["Z_range"].quantile(0.5)),
            "Z_range_p95": float(sub["Z_range"].quantile(0.95)),
        }
        if "TVT_min" in sub.columns and sub["TVT_min"].notna().any():
            summary[split].update(
                {
                    "TVT_min_min": float(sub["TVT_min"].min()),
                    "TVT_max_max": float(sub["TVT_max"].max()),
                    "TVT_range_p50": float(sub["TVT_range"].quantile(0.5)),
                    "TVT_range_p95": float(sub["TVT_range"].quantile(0.95)),
                    "TVT_input_eq_TVT_on_visible_all_wells": bool(
                        sub["TVT_input_matches_TVT_on_visible"].all()
                    ),
                    "typewell_covers_horizontal_tvt_count": int(
                        sub["typewell_covers_horizontal_tvt"].sum()
                    ),
                }
            )
        if "typewell_geology_unique_count" in sub.columns:
            geo_n = sub["typewell_geology_n_labeled"].dropna()
            geo_u = sub["typewell_geology_unique_count"].dropna()
            if len(geo_n):
                summary[split]["typewell_geology_n_labeled_p50"] = int(geo_n.quantile(0.5))
                summary[split]["typewell_geology_n_labeled_min"] = int(geo_n.min())
                summary[split]["typewell_geology_n_labeled_max"] = int(geo_n.max())
            if len(geo_u):
                summary[split]["typewell_geology_unique_count_p50"] = int(geo_u.quantile(0.5))
                summary[split]["typewell_geology_unique_count_min"] = int(geo_u.min())
                summary[split]["typewell_geology_unique_count_max"] = int(geo_u.max())

    with open(OUT / "aggregate-summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT / 'aggregate-summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
