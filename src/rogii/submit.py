"""Generate submission.csv from saved test preds and submit via Kaggle CLI.

Usage:
    uv run python -m rogii.submit "lgb baseline cv=0.842"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kagglib.submit import submit_to_kaggle

from .config import load_config
from .data_io import load_train_test


def main(message: str) -> None:
    cfg = load_config()
    _, test = load_train_test()
    pred_path = Path("outputs/preds/test_lgb.npy")
    if not pred_path.exists():
        print(f"missing {pred_path}; run `make baseline` first", file=sys.stderr)
        sys.exit(1)

    preds = np.load(pred_path)
    submission = pd.DataFrame({cfg["id_col"]: test[cfg["id_col"]], cfg["target"]: preds})
    out_csv = Path("submissions") / "latest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)

    submit_to_kaggle(out_csv, competition=cfg["slug"], message=message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m rogii.submit '<message>'", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
