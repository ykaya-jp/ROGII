"""Data IO — competition-specific reading. Override per comp."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglib.data import read_csv_polars

DATA_RAW = Path("data/raw")


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv from data/raw/. Override if filenames differ."""
    train = read_csv_polars(DATA_RAW / "train.csv").to_pandas()
    test = read_csv_polars(DATA_RAW / "test.csv").to_pandas()
    return train, test
