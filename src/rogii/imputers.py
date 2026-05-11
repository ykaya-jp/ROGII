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


# Dense per-well ANCC subsampler + KDTree IDW — used by exp004 (= Approach A).
# Inspired by romantamrazov/rogii-super-solution-lb-top-3 lines 364-394 (Apache 2.0).
DENSE_SPW = 60
DENSE_K = 20


class DenseANCCImputer:
    """KDTree + IDW over uniformly-subsampled per-well ANCC points.

    Self-exclusion enforced: when `self_wid` is passed, all points belonging
    to that well are masked to inf before top-K selection.
    """

    def __init__(self, train_dir: Path, spw: int = DENSE_SPW):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        anccs: list[np.ndarray] = []
        wids: list[str] = []
        for p in sorted(Path(train_dir).glob("*__horizontal_well.csv")):
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(
        self,
        xy_q: np.ndarray,
        self_wid: str | None = None,
        k: int = DENSE_K,
        nfetch: int = 3000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (
            ap.astype(np.float32),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            np.where(vk, dk, np.inf).min(1).astype(np.float32),
        )
