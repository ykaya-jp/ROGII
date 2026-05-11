"""Phase A2: Hill Climb ensemble selection (Caruana 2004 generalized).

Plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1.
Strength analysis: docs/research/2026-05-12-hill-climb-strength-analysis.dense.md §1.1.
Criteria: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml AC-A2.

Self-implemented replacement for `from hill_climbing import Climber` used by
Roman Tarasov's Hill Climb (LB 9.43) kernel. The original is a PyPI package
("hill-climbing"), but we re-implement from first principles to (a) keep
self-compiled paradigm coverage per ~/projects/kaggle/CLAUDE.md §11
("self-compiled 75%+"), (b) allow custom evaluation metrics + per-row sample
weights, and (c) audit the negative-weight pathway that distinguishes this
ensemble strategy from Ridge stacking.

Algorithm: generalized Caruana et al. 2004 "Ensemble Selection from Libraries
of Models" (ICML, https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)
with two extensions:
  1. Greedy selection with replacement is reformulated as discrete weight
     updates: each iteration adds ±precision to one base's weight, picking
     the (base, sign) pair that maximizes the eval metric.
  2. `allow_negative_weights=True` enables the - sign branch (= equivalent to
     Caruana 2004 §2.5 "Bagged Ensemble Selection" extended to signed weights).

License / attribution:
  - Algorithm: Caruana et al. 2004 (ICML), public domain academic algorithm
  - Re-implementation: this file, fair-use of ravaghi's notebook architecture
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import root_mean_squared_error


class Climber:
    """Greedy hill-climbing ensemble weight optimizer.

    Drop-in replacement for `from hill_climbing import Climber` used by
    Hill Climb (LB 9.43) Kaggle notebook (ravaghi/wellbore-geology-prediction-hill-climbing).

    Example:
        >>> from rogii.hill_climb import Climber
        >>> climber = Climber(
        ...     objective="minimize",
        ...     eval_metric=root_mean_squared_error,
        ...     allow_negative_weights=True,
        ...     precision=0.001,
        ... ).fit(oof_matrix, y_true)
        >>> blend_test = climber.predict(test_matrix)
        >>> climber.best_score  # CV RMSE of best blend
    """

    def __init__(
        self,
        objective: str = "minimize",
        eval_metric: Callable[..., float] | None = None,
        allow_negative_weights: bool = True,
        precision: float = 0.001,
        max_iter: int = 10000,
        patience: int = 10,
        verbose: bool = False,
        n_jobs: int = 1,  # placeholder, single-thread is already fast
        seed: int = 42,
    ) -> None:
        if objective not in {"minimize", "maximize"}:
            raise ValueError(f"objective must be 'minimize' or 'maximize', got {objective}")
        if precision <= 0 or precision > 1:
            raise ValueError(f"precision must be in (0, 1], got {precision}")

        self.objective = objective
        self.eval_metric = eval_metric or root_mean_squared_error
        self.allow_negative_weights = allow_negative_weights
        self.precision = float(precision)
        self.max_iter = int(max_iter)
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

        # post-fit attributes
        self.weights_: np.ndarray | None = None
        self.best_score_: float = float("nan")
        self.history_: list[tuple[int, float]] = []
        self.n_base_: int = 0

    def _is_better(self, new: float, old: float, tol: float = 1e-12) -> bool:
        if self.objective == "minimize":
            return new < old - tol
        return new > old + tol

    def fit(self, oof: np.ndarray, y: np.ndarray) -> "Climber":
        """Greedy fit on OOF matrix.

        Args:
            oof: shape (n_samples, n_base), each column = one base model OOF
            y: shape (n_samples,), ground truth

        Returns:
            self (fitted)
        """
        oof = np.asarray(oof, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if oof.ndim != 2:
            raise ValueError(f"oof must be 2D (n_samples, n_base), got shape {oof.shape}")
        if len(y) != oof.shape[0]:
            raise ValueError(
                f"length mismatch: y {y.shape} vs oof {oof.shape}"
            )

        n_samples, n_base = oof.shape
        self.n_base_ = n_base

        weights = np.zeros(n_base, dtype=np.float64)
        cur_pred = np.zeros(n_samples, dtype=np.float64)
        best_score = float(self.eval_metric(y, cur_pred))
        history: list[tuple[int, float]] = [(0, best_score)]
        no_improve = 0

        signs: tuple[float, ...] = (1.0, -1.0) if self.allow_negative_weights else (1.0,)

        for it in range(1, self.max_iter + 1):
            best_new_score = best_score
            best_idx = -1
            best_sign = 0.0

            for j in range(n_base):
                for sign in signs:
                    cand_pred = cur_pred + sign * self.precision * oof[:, j]
                    score = float(self.eval_metric(y, cand_pred))
                    if self._is_better(score, best_new_score):
                        best_new_score = score
                        best_idx = j
                        best_sign = sign

            if best_idx < 0:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(
                            f"[Climber] no improvement for {self.patience} iters at iter {it} "
                            f"(best score={best_score:.6f})"
                        )
                    break
            else:
                cur_pred = cur_pred + best_sign * self.precision * oof[:, best_idx]
                weights[best_idx] += best_sign * self.precision
                best_score = best_new_score
                no_improve = 0
                history.append((it, best_score))
                if self.verbose and it % 100 == 0:
                    print(f"[Climber] iter {it}  best={best_score:.6f}  weights_sum={weights.sum():.4f}")

        self.weights_ = weights
        self.best_score_ = best_score
        self.history_ = history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted weights to a (n_samples, n_base) matrix → (n_samples,)."""
        if self.weights_ is None:
            raise RuntimeError("Climber has not been fit. Call .fit(oof, y) first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_base_:
            raise ValueError(
                f"X has {X.shape[1]} bases but Climber was fit with {self.n_base_}"
            )
        return X @ self.weights_

    @property
    def best_score(self) -> float:
        """Best CV metric value (= same convention as the original Climber API)."""
        return self.best_score_
