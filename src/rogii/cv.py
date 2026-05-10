"""Cross-validation for ROGII.

Strategy: GroupKFold by well (so a well never appears in both train and val).
Validation metric is RMSE on hidden rows of held-out wells only,
which mirrors the public leaderboard scoring exactly.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def make_well_folds(
    well_ids: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) splits where each well is in exactly one val fold.

    Wells are shuffled deterministically using `seed` before being assigned to folds.
    """
    rng = np.random.RandomState(seed)
    unique_wells = np.array(sorted(np.unique(well_ids)))
    rng.shuffle(unique_wells)

    # Build a mapping: well → fold index. Spread wells round-robin.
    fold_of = {w: i % n_splits for i, w in enumerate(unique_wells)}
    well_fold = np.array([fold_of[w] for w in well_ids])

    folds = []
    for k in range(n_splits):
        val_idx = np.where(well_fold == k)[0]
        train_idx = np.where(well_fold != k)[0]
        folds.append((train_idx, val_idx))
    return folds


def rmse_hidden(
    y_true: np.ndarray, y_pred: np.ndarray, hidden_mask: np.ndarray
) -> float:
    """RMSE evaluated only on rows where hidden_mask is True."""
    err = y_true[hidden_mask] - y_pred[hidden_mask]
    return float(np.sqrt(np.mean(err**2)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err**2)))
