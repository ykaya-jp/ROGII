"""Adversarial validation for ROGII (= C3 CV strategy).

Reweights / drops train wells whose visible-only per-well features look
*less* like the test distribution than average. Implements Chris Deotte's
adversarial validation playbook adapted to ROGII's well-level setting.

Algorithm:
    1. ``compute_adversarial_well_features`` — per-well visible-region
       features (no TVT, no hidden region, no target).
    2. ``fit_adversarial_classifier`` — 5-fold StratifiedKFold + LightGBM
       binary classifier predicting ``is_test``. Returns OOF AUC and the
       train-side test-class probability.
    3. ``apply_adversarial_reweight`` — bottom ``drop_quantile`` train
       wells (= lowest test-likelihood) get ``drop_weight`` (default 0.5).

Leak guard (3 layers, mandatory):
    a. Feature builder reads only visible-region rows (= ``TVT_input``
       finite) — no TVT, no hidden, no target.
    b. Forbidden-column assertion in caller (= test_cv_strategies +
       kernel-side ``assert``).
    c. Returns only ``oof_prob[:n_train]`` from the classifier; test rows
       are never used to reweight themselves.

Originated in ``kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py``
(commit 8cb92f7). This module is the canonical src/ home; kernels keep
an inline copy for self-containedness — **mirror any change here into
those kernel inline copies**.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Feature builder (visible-only, leak-safe)
# ────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_FEATURE_COLS_DEFAULT: Tuple[str, ...] = (
    "visible_ratio",
    "gr_noise_std",
    "gr_mean",
    "gr_std",
    "gr_nan_frac",
    "n_rows",
    "n_visible",
    "z_range",
)
"""Default feature set for adversarial classification. All visible-region
metrics — explicitly excludes TVT, TVT_input, target, tvt_range."""

FORBIDDEN_FEATURE_COLS: Tuple[str, ...] = (
    "target", "TVT", "TVT_input", "tvt_range",
)
"""Hard-deny list. Callers must assert these never appear in their
feature_cols tuple — runtime guard against leak."""


def compute_adversarial_well_features(
    raw_dir: Path,
    well_ids: List[str],
    feature_cols: Tuple[str, ...] = ADVERSARIAL_FEATURE_COLS_DEFAULT,
) -> pd.DataFrame:
    """Build per-well visible-only features for adversarial classification.

    For each well, reads ``{wid}__horizontal_well.csv`` and computes:
        - n_rows, n_visible, visible_ratio
        - gr_mean, gr_std, gr_nan_frac, gr_noise_std (= rolling median resid)
        - z_range
    All from **visible region only** (= TVT_input is finite).

    Returns DataFrame indexed by well_id with columns matching feature_cols.
    Missing files / unreadable wells get a NaN row.
    """
    raw_dir = Path(raw_dir)
    rows: List[dict] = []
    for wid in well_ids:
        path = raw_dir / f"{wid}__horizontal_well.csv"
        if not path.exists():
            rows.append({"well_id": wid, **{c: np.nan for c in feature_cols}})
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            rows.append({"well_id": wid, **{c: np.nan for c in feature_cols}})
            continue
        n_rows = len(df)
        if "TVT_input" in df.columns:
            visible_mask = df["TVT_input"].notna().values
        else:
            visible_mask = np.ones(n_rows, dtype=bool)
        n_vis = int(visible_mask.sum())
        gr = df["GR"].values if "GR" in df.columns else np.full(n_rows, np.nan)
        gr_vis = gr[visible_mask]
        z_col = (
            df["Z"]
            if "Z" in df.columns
            else (df["MD"] if "MD" in df.columns else None)
        )
        z_range = (
            float(z_col.max() - z_col.min()) if z_col is not None else np.nan
        )
        gr_mean = float(np.nanmean(gr_vis)) if len(gr_vis) > 0 else np.nan
        gr_std = float(np.nanstd(gr_vis)) if len(gr_vis) > 0 else np.nan
        gr_nan_frac = float(np.isnan(gr).mean())
        if len(gr_vis) > 30:
            gr_med = (
                pd.Series(gr_vis)
                .rolling(11, min_periods=1, center=True)
                .median()
                .values
            )
            gr_noise_std = float(np.nanstd(gr_vis - gr_med))
        else:
            gr_noise_std = np.nan
        row = {
            "well_id":       wid,
            "n_rows":        n_rows,
            "n_visible":     n_vis,
            "visible_ratio": n_vis / max(n_rows, 1),
            "gr_mean":       gr_mean,
            "gr_std":        gr_std,
            "gr_nan_frac":   gr_nan_frac,
            "gr_noise_std":  gr_noise_std,
            "z_range":       z_range,
        }
        for c in feature_cols:
            row.setdefault(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("well_id")


# ────────────────────────────────────────────────────────────────────────────
# Classifier (= LightGBM binary, 5-fold StratifiedKFold OOF)
# ────────────────────────────────────────────────────────────────────────────


def fit_adversarial_classifier(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    lgb_params: dict,
    n_estimators: int = 200,
    early_stopping: int = 20,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Fit a binary classifier (train=0, test=1) and return (OOF AUC, train OOF prob).

    Algorithm:
        - Concatenate train + test feature rows; assign y=0/1 by source.
        - 5-fold StratifiedKFold (= min(5, n_test // 2) if test is tiny).
        - LightGBM binary with early stopping on OOF AUC.
        - Returns ``(AUC, oof_prob_train_only)`` — only train rows for reweight.

    Leak guard:
        - Assert no forbidden column in feature_cols.
        - Features are visible-only per-well stats (see
          ``compute_adversarial_well_features``).

    Args:
        feat_train: DataFrame indexed by train well_id, columns ⊇ feature_cols.
        feat_test: Same shape, but for test wells.
        feature_cols: Tuple of column names. Must exclude FORBIDDEN_FEATURE_COLS.
        lgb_params: LightGBM hyperparams (objective='binary', metric='auc', etc.).
        n_estimators: Max boosting rounds.
        early_stopping: Patience for OOF AUC.
        seed: Random seed for KFold.

    Returns:
        ``(auc, train_likelihood)`` where ``train_likelihood[i]`` is the OOF
        probability that train well i looks like a test well.
    """
    # Leak guard
    bad = [c for c in feature_cols if c in FORBIDDEN_FEATURE_COLS]
    if bad:
        raise ValueError(
            f"adversarial feature_cols contain forbidden leak columns: {bad}"
        )

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as _lgb

    n_train = len(feat_train)
    n_test = len(feat_test)
    X = pd.concat(
        [feat_train[list(feature_cols)], feat_test[list(feature_cols)]],
        axis=0,
        ignore_index=False,
    )
    y = np.concatenate([np.zeros(n_train), np.ones(n_test)]).astype(np.int8)

    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    n_splits = min(5, max(2, n_test // 2))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(X), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        Xt, yt = X.iloc[tr_idx], y[tr_idx]
        Xv, yv = X.iloc[va_idx], y[va_idx]
        clf = _lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **{k: v for k, v in lgb_params.items() if k != "metric"},
        )
        clf.fit(
            Xt.values, yt,
            eval_set=[(Xv.values, yv)],
            callbacks=[
                _lgb.early_stopping(early_stopping),
                _lgb.log_evaluation(0),
            ],
        )
        oof_prob[va_idx] = clf.predict_proba(Xv.values)[:, 1].astype(np.float32)
    auc = float(roc_auc_score(y, oof_prob))
    return auc, oof_prob[:n_train]


# ────────────────────────────────────────────────────────────────────────────
# Reweight (= row-level sample_weight modifier)
# ────────────────────────────────────────────────────────────────────────────


def apply_adversarial_reweight(
    w_train: np.ndarray,
    train_well_ids: np.ndarray,
    well_to_test_likelihood: dict,
    drop_quantile: float = 0.20,
    drop_weight: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """Reweight ``w_train``: bottom ``drop_quantile`` wells → ``drop_weight``.

    "Bottom" here means lowest test-likelihood = least test-like = most
    safely dropped without sacrificing prediction signal on test-like wells.

    Args:
        w_train: Per-row sample_weight, shape (n_train_rows,).
        train_well_ids: Per-row well_id, shape (n_train_rows,).
        well_to_test_likelihood: ``{well_id: float P(test|well)}`` from
            ``fit_adversarial_classifier``.
        drop_quantile: Fraction of wells to reweight (default 0.20).
        drop_weight: Multiplier applied to the rows of reweighted wells
            (default 0.5; use 0.0 for hard drop).

    Returns:
        ``(new_w_train, n_wells_reweighted)``.
    """
    n_wells = len(well_to_test_likelihood)
    if n_wells == 0:
        return w_train, 0
    vals = np.array(list(well_to_test_likelihood.values()))
    threshold = np.quantile(vals, drop_quantile)
    low_wells = {
        w for w, v in well_to_test_likelihood.items() if v <= threshold
    }
    new_w = w_train.copy()
    mask = np.array([w in low_wells for w in train_well_ids])
    new_w[mask] = new_w[mask] * drop_weight
    return new_w, int(len(low_wells))


# ────────────────────────────────────────────────────────────────────────────
# Fold-level wrapper: drop wells outright (= treat as 0-weight in CV)
# ────────────────────────────────────────────────────────────────────────────


def compute_adversarial_classifier_oof(
    train_dir: Path,
    test_dir: Path,
    train_well_ids: List[str],
    test_well_ids: List[str],
    feature_cols: Tuple[str, ...] = ADVERSARIAL_FEATURE_COLS_DEFAULT,
    lgb_params: dict | None = None,
    n_estimators: int = 200,
    early_stopping: int = 20,
    seed: int = 42,
) -> Tuple[float, dict[str, float]]:
    """One-shot convenience wrapper: build features + fit classifier.

    Returns:
        ``(auc, {well_id: P(test|well)})`` for train wells only.
    """
    if lgb_params is None:
        lgb_params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=2,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
            seed=seed,
        )
    feat_tr = compute_adversarial_well_features(
        train_dir, train_well_ids, feature_cols
    )
    feat_te = compute_adversarial_well_features(
        test_dir, test_well_ids, feature_cols
    )
    auc, train_likelihood = fit_adversarial_classifier(
        feat_tr, feat_te, feature_cols, lgb_params,
        n_estimators=n_estimators, early_stopping=early_stopping, seed=seed,
    )
    return auc, dict(zip(feat_tr.index.astype(str), train_likelihood))


def apply_adversarial_drop_fold(
    train_df: pd.DataFrame,
    base_fold_id: np.ndarray,
    train_likelihood: dict[str, float],
    drop_quantile: float = 0.20,
    well_col: str = "well",
) -> np.ndarray:
    """Mark bottom ``drop_quantile`` wells as fold = -1 (= excluded).

    This is the **fold-level** counterpart to ``apply_adversarial_reweight``.
    Instead of reweighting rows, it marks low-test-likelihood wells with
    fold_id = -1, signaling "exclude from all folds" to downstream training.

    Use this when you want C3 = adversarial validation **drop** (= hard
    removal) rather than soft reweight.

    Args:
        train_df: DataFrame with ``well_col`` column.
        base_fold_id: Existing fold assignment, shape (n_rows,).
        train_likelihood: ``{well_id: P(test|well)}``.
        drop_quantile: Fraction of wells to mark fold = -1.
        well_col: Column in train_df with well_id.

    Returns:
        New fold_id array with low-likelihood wells set to -1.
    """
    vals = np.array(list(train_likelihood.values()))
    if len(vals) == 0:
        return base_fold_id
    threshold = np.quantile(vals, drop_quantile)
    low_wells = {
        w for w, v in train_likelihood.items() if v <= threshold
    }
    new_fold_id = base_fold_id.copy()
    well_arr = train_df[well_col].astype(str).values
    mask = np.array([w in low_wells for w in well_arr])
    new_fold_id[mask] = -1
    return new_fold_id


__all__ = [
    "ADVERSARIAL_FEATURE_COLS_DEFAULT",
    "FORBIDDEN_FEATURE_COLS",
    "compute_adversarial_well_features",
    "fit_adversarial_classifier",
    "apply_adversarial_reweight",
    "compute_adversarial_classifier_oof",
    "apply_adversarial_drop_fold",
]
