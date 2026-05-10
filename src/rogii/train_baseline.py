"""Run 4 diverse baselines × 5-fold CV. Saves per-model OOF + test preds.

Usage:
    uv run python -m rogii.train_baseline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from kagglib.baselines import run_all
from kagglib.cv import make_folds
from kagglib.seed import seed_everything
from kagglib.tracking import start_run

from .config import load_config
from .data_io import load_train_test

OUT_DIR = Path("outputs")


def _metric_fn(metric: str):
    if metric == "roc_auc":
        return roc_auc_score
    if metric == "logloss":
        return lambda y, p: -log_loss(y, p)  # negate so higher = better
    if metric == "rmse":
        return lambda y, p: -np.sqrt(mean_squared_error(y, p))
    raise ValueError(f"unknown metric: {metric}")


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["cv"]["seed"])

    train, test = load_train_test()
    y = train[cfg["target"]].to_numpy()
    drop_cols = [cfg["target"], cfg["id_col"]] + cfg["features"].get("drop", [])
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    metric = _metric_fn(cfg["metric"])
    folds = list(make_folds(X, y, kind=cfg["cv"]["kind"], n_splits=cfg["cv"]["n_splits"], seed=cfg["cv"]["seed"]))

    with start_run(experiment=cfg["slug"], run_name="baseline-bank") as logger:
        results = run_all(X, y, folds, metric=metric, task=cfg["task"], X_test=X_test)
        for name, r in results.items():
            logger.log_metric(f"{name}_cv", float(r["score"]))
            (OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "preds").mkdir(parents=True, exist_ok=True)
            np.save(OUT_DIR / "oof" / f"oof_{name}.npy", r["oof"])
            if r["test"] is not None:
                np.save(OUT_DIR / "preds" / f"test_{name}.npy", r["test"])
            print(f"{name:8s}  cv={r['score']:.6f}")


if __name__ == "__main__":
    main()
