"""Compute typewell content-hash groups + azimuth dispersion for 773 train wells.

Run: `uv run python notebooks/_typewell_groups_build.py`

Output: data/processed/typewell_groups.{parquet,csv} (v2, 14 cols)

Background: discussion topic 698449 (ROGII official reply, 2026-05-10) reveals
that some horizontal wells share the same typewell file because the geologist
manually picks the closest available reference (and "pseudo-typewell" =
interpretation from a nearby lateral) is reused across multiple laterals.

This script
1. computes a content fingerprint (md5 of TVT + GR + Geology bytes,
   NaN-normalised, rounded to 4 decimals) for every train typewell and groups
   wells sharing the same fingerprint;
2. computes the per-well azimuth (PCA first axis of the visible-zone XY
   trajectory and the simpler full-trajectory end-minus-start direction);
3. for each duplicate group, computes the circular standard deviation of
   the wells' azimuths;
4. flags wells as is_pseudo_likely when they sit in a duplicate group whose
   azimuth circular std exceeds 60 deg — empirically these groups have
   wells running in opposite directions yet sharing one typewell, the
   strongest signal that the typewell is actually a pseudo-typewell
   (interpretation propagated from a previously drilled lateral).

The 13 duplicate groups disclosed in discussion 698449 are recovered exactly
(13/13 matched). 5 of those 13 groups (10 wells) are flagged is_pseudo_likely.

Implication for CV: GroupKFold by well_id is insufficient. To prevent leakage,
fold assignment must respect typewell_hash — i.e. wells in the same group
must land in the same fold. Use the `group_id` column for sklearn's
GroupKFold(groups=group_id). For loss weighting, downweight wells where
is_pseudo_likely is True.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/train")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PSEUDO_AZIMUTH_STD_THRESH_DEG = 60.0


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


def well_azimuth(horizontal_csv: Path) -> tuple[float, float, float]:
    """Return (azimuth_visible_pca_deg, azimuth_full_deg, straightness).

    azimuth is in degrees clockwise from north (= +Y axis).
    straightness is S2/S1 of XY SVD (lower = straighter, typical < 0.05).
    """
    df = pd.read_csv(horizontal_csv, usecols=["MD", "X", "Y", "TVT_input"])
    visible = df[df["TVT_input"].notna()]
    xy = visible[["X", "Y"]].dropna().values
    if len(xy) < 50:
        return float("nan"), float("nan"), float("nan")
    centered = xy - xy.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    az_pca = float(np.degrees(np.arctan2(pc1[0], pc1[1])) % 360)
    full_xy = df[["X", "Y"]].dropna().values
    head = full_xy[: max(50, len(full_xy) // 50)].mean(axis=0)
    tail = full_xy[-max(50, len(full_xy) // 50) :].mean(axis=0)
    dxy = tail - head
    az_full = float(np.degrees(np.arctan2(dxy[0], dxy[1])) % 360)
    straight = float(S[1] / S[0]) if S[0] > 0 else 0.0
    return az_pca, az_full, straight


def circular_std_deg(angles_deg: np.ndarray) -> float:
    if len(angles_deg) < 2:
        return 0.0
    ang = np.radians(angles_deg)
    R = np.sqrt(np.cos(ang).sum() ** 2 + np.sin(ang).sum() ** 2) / len(angles_deg)
    if R <= 1e-9:
        return 999.0
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def main() -> None:
    typewells = sorted(RAW_DIR.glob("*__typewell.csv"))
    if not typewells:
        raise SystemExit(f"no typewell files under {RAW_DIR}")
    rows = []
    for path in typewells:
        well_id = path.name.split("__")[0]
        df = pd.read_csv(path)
        horizontal_path = RAW_DIR / f"{well_id}__horizontal_well.csv"
        az_pca, az_full, straight = (
            well_azimuth(horizontal_path) if horizontal_path.exists() else (float("nan"),) * 3
        )
        rows.append(
            {
                "well_id": well_id,
                "typewell_hash": fingerprint(df),
                "n_rows": len(df),
                "tvt_min": df["TVT"].min() if "TVT" in df.columns else None,
                "tvt_max": df["TVT"].max() if "TVT" in df.columns else None,
                "has_geology": ("Geology" in df.columns and df["Geology"].notna().any()),
                "azimuth_visible_pca_deg": az_pca,
                "azimuth_full_deg": az_full,
                "straightness_visible": straight,
            }
        )

    out = pd.DataFrame(rows)
    out["group_size"] = out.groupby("typewell_hash")["well_id"].transform("count")
    out["is_duplicate"] = out["group_size"] > 1
    out["group_id"] = out.groupby("typewell_hash")["well_id"].transform("min")
    out["azimuth_circ_std_deg"] = out.groupby("group_id")["azimuth_full_deg"].transform(
        lambda s: circular_std_deg(s.values)
    )
    out["is_pseudo_likely"] = out["is_duplicate"] & (
        out["azimuth_circ_std_deg"] >= PSEUDO_AZIMUTH_STD_THRESH_DEG
    )
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
            "azimuth_visible_pca_deg",
            "azimuth_full_deg",
            "straightness_visible",
            "azimuth_circ_std_deg",
            "is_pseudo_likely",
        ]
    ]
    out.to_parquet(OUT_DIR / "typewell_groups.parquet", index=False)
    out.to_csv(OUT_DIR / "typewell_groups.csv", index=False)

    n_total = len(out)
    n_unique = out["typewell_hash"].nunique()
    n_dupe = int(out["is_duplicate"].sum())
    n_pseudo = int(out["is_pseudo_likely"].sum())
    print(f"wells           : {n_total}")
    print(f"unique typewells : {n_unique}")
    print(f"wells in dup group: {n_dupe} ({n_dupe / n_total:.1%})")
    print(f"is_pseudo_likely : {n_pseudo} wells in {out[out['is_pseudo_likely']]['group_id'].nunique()} groups")
    print("\ngroup_size distribution:")
    print(out.groupby("group_size").size().to_string())


if __name__ == "__main__":
    main()
