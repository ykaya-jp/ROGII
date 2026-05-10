"""Formation top imputers — predict ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA at any (X, Y).

The formation columns are train-only. To use the closed-form
    tvt_formula = -Z + ANCC + b_well
on test wells we must impute ANCC (and friends) from neighboring training wells.

FormationPlaneKNN: per-row weighted 2D plane fit using K=10 nearest non-self
training-well centroids. Public-notebook reference (`konbu17`,
`needless090`) reports plane-fit RMSE ≈ 17 ft per formation vs IDW 47 ft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


class FormationPlaneKNN:
    """K-nearest non-self centroid plane-fit for each formation top."""

    def __init__(self, train_dir: Path, k: int = 10):
        self.k = k
        rows: list[dict] = []
        for p in sorted(Path(train_dir).glob("*__horizontal_well.csv")):
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y", *FORMATIONS]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_med"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"].to_numpy())}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(axis=0) < 1e-3, 1.0, xy.std(axis=0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_med" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (formation_pred[N, 6], min_distance[N]).

        For each query row, fits a 2D plane (z = a*X + b*Y + c) using K=k nearest
        non-self centroid wells, weighted by 1/distance.
        """
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q = xy_q / self.scale
        nf = min(self.k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid is not None and self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        order = np.argpartition(dist, min(self.k - 1, nf - 1), 1)[:, : self.k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        wx = w * xn
        wy = w * yn
        # Solve weighted normal equations: A @ coef = rhs, A is (3, 3) per row
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        # Tikhonov for numerical stability (formation values ~ thousands)
        for i in range(3):
            A[:, i, i] += 1e-9
        fn = self.fa[ik]  # (N, K, 6)
        rhs = np.stack(
            [(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)],
            1,
        )
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (
            Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]
        ).astype(np.float32)
        # Fallback for rows with no valid neighbors: global mean per formation
        nofit = ~vk.any(1)
        if nofit.any():
            pred[nofit] = self.fa.mean(0).astype(np.float32)
        min_dist = np.where(vk, dk, np.inf).min(1).astype(np.float32)
        return pred, min_dist
