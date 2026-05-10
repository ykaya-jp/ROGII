"""Data IO for ROGII — walk per-well CSVs and assemble into long DataFrames.

Each well is identified by an 8-char hash. Both train/ and test/ directories
contain `{HASH}__horizontal_well.csv` and `{HASH}__typewell.csv`.

Train horizontal_well columns:
    MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input

Test horizontal_well columns (drops formation depths and TVT target):
    MD, X, Y, Z, GR, TVT_input

All wells use 1.0 ft MD step (verified Phase 1 EDA).
Evaluation zone is always the trailing block where TVT_input is NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")
TRAIN_DIR = DATA_RAW / "train"
TEST_DIR = DATA_RAW / "test"


def list_wells(split_dir: Path) -> list[str]:
    return sorted({p.stem.split("__")[0] for p in split_dir.glob("*__horizontal_well.csv")})


def load_horizontal(split_dir: Path, well: str) -> pd.DataFrame:
    df = pd.read_csv(split_dir / f"{well}__horizontal_well.csv")
    df.insert(0, "well", well)
    df["row_idx"] = np.arange(len(df), dtype=np.int32)
    return df


def load_typewell(split_dir: Path, well: str) -> pd.DataFrame:
    df = pd.read_csv(split_dir / f"{well}__typewell.csv")
    df.insert(0, "well", well)
    return df


def load_split_horizontals(split_dir: Path) -> pd.DataFrame:
    """Load and concatenate all horizontal_well.csv files in a split."""
    wells = list_wells(split_dir)
    dfs = [load_horizontal(split_dir, w) for w in wells]
    out = pd.concat(dfs, ignore_index=True)
    return out


def load_split_typewells(split_dir: Path) -> pd.DataFrame:
    """Load and concatenate all typewell.csv files in a split."""
    wells = list_wells(split_dir)
    dfs = [load_typewell(split_dir, w) for w in wells]
    return pd.concat(dfs, ignore_index=True)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_h, train_tw, test_h, test_tw)."""
    return (
        load_split_horizontals(TRAIN_DIR),
        load_split_typewells(TRAIN_DIR),
        load_split_horizontals(TEST_DIR),
        load_split_typewells(TEST_DIR),
    )


def submission_id_format(well: str, row_idx: int) -> str:
    return f"{well}_{row_idx}"


def make_submission_template(test_h: pd.DataFrame) -> pd.DataFrame:
    """Build the (id, tvt) skeleton matching sample_submission.csv (only hidden rows)."""
    hidden_mask = test_h["TVT_input"].isna()
    sub = pd.DataFrame(
        {
            "id": [
                submission_id_format(w, i)
                for w, i in zip(
                    test_h.loc[hidden_mask, "well"].to_numpy(),
                    test_h.loc[hidden_mask, "row_idx"].to_numpy(),
                    strict=True,
                )
            ],
            "tvt": np.zeros(int(hidden_mask.sum())),
        }
    )
    return sub
