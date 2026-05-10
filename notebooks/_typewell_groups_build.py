"""Compute typewell content-hash groups for all 773 train wells.

Run: `uv run python notebooks/_typewell_groups_build.py`

Output: data/processed/typewell_groups.{parquet,csv}

Background: discussion topic 698449 (ROGII official reply, 2026-05-10) reveals
that some horizontal wells share the same typewell file because the geologist
manually picks the closest available reference (and "pseudo-typewell" =
interpretation from a nearby lateral) is reused across multiple laterals.

This script computes a content fingerprint (md5 of TVT + GR + Geology bytes,
NaN-normalised, rounded to 4 decimals) for every train typewell and groups
wells sharing the same fingerprint. The 13 groups disclosed in discussion
698449 are recovered exactly (13/13 matched) and no extra hidden duplicates
are found beyond those 13.

Implication for CV: GroupKFold by well_id is insufficient. To prevent leakage,
fold assignment must respect typewell_hash — i.e. wells in the same group
must land in the same fold. Use the `group_id` column for sklearn's
GroupKFold(groups=group_id).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/train")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    for col in ("TVT", "GR", "Geology"):
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
            payload = df[col].fillna("NaN").astype(str).str.cat(sep="|").encode()
        else:
            payload = df[col].fillna(-99999.0).round(4).values.tobytes()
        h.update(payload)
    return h.hexdigest()


def main() -> None:
    typewells = sorted(RAW_DIR.glob("*__typewell.csv"))
    if not typewells:
        raise SystemExit(f"no typewell files under {RAW_DIR}")
    rows = []
    for path in typewells:
        well_id = path.name.split("__")[0]
        df = pd.read_csv(path)
        rows.append(
            {
                "well_id": well_id,
                "typewell_hash": fingerprint(df),
                "n_rows": len(df),
                "tvt_min": df["TVT"].min() if "TVT" in df.columns else None,
                "tvt_max": df["TVT"].max() if "TVT" in df.columns else None,
                "has_geology": ("Geology" in df.columns and df["Geology"].notna().any()),
            }
        )

    out = pd.DataFrame(rows)
    out["group_size"] = out.groupby("typewell_hash")["well_id"].transform("count")
    out["is_duplicate"] = out["group_size"] > 1
    out["group_id"] = out.groupby("typewell_hash")["well_id"].transform("min")
    out = out.sort_values(
        ["group_size", "group_id", "well_id"], ascending=[False, True, True]
    ).reset_index(drop=True)
    out = out[
        [
            "well_id",
            "typewell_hash",
            "group_id",
            "group_size",
            "is_duplicate",
            "n_rows",
            "tvt_min",
            "tvt_max",
            "has_geology",
        ]
    ]
    out.to_parquet(OUT_DIR / "typewell_groups.parquet", index=False)
    out.to_csv(OUT_DIR / "typewell_groups.csv", index=False)

    n_total = len(out)
    n_unique = out["typewell_hash"].nunique()
    n_dupe = int(out["is_duplicate"].sum())
    print(f"wells          : {n_total}")
    print(f"unique typewells: {n_unique}")
    print(f"wells in dup group: {n_dupe} ({n_dupe / n_total:.1%})")
    print("\ngroup_size distribution:")
    print(out.groupby("group_size").size().to_string())


if __name__ == "__main__":
    main()
