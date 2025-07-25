"""Block-UCB variant of portfolio UCB-RSSR for side-constraint settings.

A *block* groups ``block_size`` consecutive rounds.  At the start of each block
we:
1.  Compute optimistic (UCB) rewards for each base.
2.  Solve a linear programme that maximises expected optimistic reward subject
    to average-cost constraints *per resource*.
3.  Convert the optimal probability vector into an integer multiset of actions
    that will be executed sequentially during the block.

If the LP is infeasible we gracefully fall back to the per-round gating rule of
``UCBRSSRPortfolioBandit`` (possibly producing a skip).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.optimize import linprog

from .portfolio_ucb_rssr import UCBRSSRPortfolioBandit

__all__ = ["BlockUCBRSSRPortfolioBandit"]


class BlockUCBRSSRPortfolioBandit(UCBRSSRPortfolioBandit):
    """Extension that schedules arms in deterministic blocks via an LP."""

    def __init__(
        self,
        *args,
        block_size: int = 20,
        **kwargs,
    ) -> None:
        """Additional parameter
        ---------------------
        block_size : int, default=20
            Number of rounds per block.  Should be ≥ number of resources to
            ensure the rounding step can satisfy constraints in expectation.
        """
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)
        assert self.block_size >= 1, "block_size must be positive"
        self._schedule: List[Optional[int]] = []  # upcoming actions for current block

    # ------------------------------------------------------------------
    # Block construction helpers
    # ------------------------------------------------------------------
    def _solve_lp(self) -> Optional[np.ndarray]:
        """Return optimal probability vector p of length B, or None if infeasible."""
        # Objective: maximise s^T p  ⇒  minimise -s^T p
        scores = np.array([self._portfolio_score(b) for b in range(self.B)])

        # If any score is non-finite (occurs early when some arms still unseen
        # and receive +inf optimistic value) fall back to picking the single
        # best action instead of solving the LP, which cannot accept inf/nan.
        if not np.isfinite(scores).all():
            p = np.zeros(self.B)
            p[int(np.argmax(scores))] = 1.0
            return p

        c = -scores  # coefficients for linprog (minimisation)

        # Constraints: cost_matrix @ p <= budget
        if self.cost_budget is None:
            # Degenerates to picking argmax score
            p = np.zeros(self.B)
            p[int(np.argmax(scores))] = 1.0
            return p

        A_ub = self.costs
        b_ub = self.cost_budget

        # Probability simplex: p_i >=0, sum p_i =1 ⇒ sum p_i =1 as equality
        A_eq = np.ones((1, self.B))
        b_eq = np.array([1.0])

        bounds = [(0.0, 1.0) for _ in range(self.B)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            return None
        return res.x  # type: ignore[return-value]

    def _build_schedule(self) -> None:
        """Populate ``self._schedule`` with a new block of actions."""
        self._schedule.clear()
        p = self._solve_lp()
        if p is None:
            # Infeasible – fall back to per-round policy (schedule a single step)
            action = super().select_portfolio()
            self._schedule.append(action)
            return

        # Convert probabilities to integer counts summing to block_size
        fractional = p * self.block_size
        counts = np.floor(fractional).astype(int)
        remainder = self.block_size - counts.sum()
        if remainder > 0:
            # Allocate remaining draws to largest fractional parts
            frac_part = fractional - counts
            top_idx = np.argsort(-frac_part)[: remainder]
            counts[top_idx] += 1
        # Sanity
        assert counts.sum() == self.block_size

        # Expand into action list (deterministic order for simplicity)
        for idx, cnt in enumerate(counts):
            self._schedule.extend([idx] * cnt)
        # Shuffle to reduce correlation
        rng = np.random.default_rng()
        rng.shuffle(self._schedule)

    # ------------------------------------------------------------------
    # Public API override
    # ------------------------------------------------------------------
    def select_portfolio(self) -> Optional[int]:
        if not self._schedule:
            self._build_schedule()
        return self._schedule.pop(0) 