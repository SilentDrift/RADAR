from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np

__all__ = [
    "UCBRSSRBudgetBandit",
]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division that avoids ``ZeroDivisionError`` and returns *default* instead."""
    return num / den if den != 0 else default


# -----------------------------------------------------------------------------
# Arm statistics structure
# -----------------------------------------------------------------------------

@dataclass
class _ArmStats:
    """Sufficient statistics for one arm (stock)."""

    count: int = 0
    sum_returns: float = 0.0
    sum_squares: float = 0.0  # accumulator for squared returns (second raw moment)

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------
    def update(self, reward: float) -> None:
        """Accumulate one observed *reward* for the arm."""
        self.count += 1
        self.sum_returns += reward
        self.sum_squares += reward ** 2

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------
    @property
    def mean(self) -> float:
        """Empirical mean \bar X_i."""
        return _safe_div(self.sum_returns, self.count)

    @property
    def second_moment(self) -> float:
        """Empirical second raw moment \tilde X_i."""
        return _safe_div(self.sum_squares, self.count)

    @property
    def variance(self) -> float:
        """Empirical variance \bar V_i (biased)."""
        return max(1e-12, self.second_moment - self.mean ** 2)


# -----------------------------------------------------------------------------
# Main algorithm
# -----------------------------------------------------------------------------

class UCBRSSRBudgetBandit:
    """Single-stock UCB-RSSR algorithm with anytime average cost constraint.

    This class realises the baseline procedure described in the project proposal:

    1. Each arm is *initialised* by one forced trade (δ initialisation).
    2. Subsequent rounds choose the arm that maximises the UCB-RSSR score among
       those *budget-feasible* for the current round.
    3. If no arm satisfies the budget gate, the round is *skipped* (null action).

    Statistics only update on trades; merely observing market prices does **not**
    affect the algorithm – aligning with the trade-based learning requirement.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        K: int,
        costs: Sequence[float],
        L: float = 1.0,
        cost_budget: Optional[float] = None,
        under_utilisation: bool = True,
        omega: float = 0.1,
    ) -> None:
        """Parameters
        ----------
        K : int
            Number of arms (stocks).
        costs : Sequence[float]
            Deterministic cost *c_i* for each arm.
        L : float, default=1.0
            Regularisation constant in RSSR definition.
        cost_budget : float | None, default=None
            Target average cost budget *c*.  ``None`` disables budget gating.
        under_utilisation : bool, default=True
            Whether to subtract the SUAK under-utilisation term
            ``log t / (omega^2 t)`` from the budget.
        omega : float, default=0.1
            Lower bound on minimal cost gap for the under-utilisation term.
        """
        self.K = int(K)
        self.costs = np.asarray(costs, dtype=float)
        assert self.costs.size == self.K, "Length of costs must equal K."

        self.L = float(L)
        self.cost_budget = float(cost_budget) if cost_budget is not None else None
        self.under_ut = bool(under_utilisation)
        self.omega = float(omega)

        # Time index (round counter, including skips)
        self.t: int = 0
        # Cumulative realised cost C(t)
        self.C: float = 0.0
        # Per-arm statistics
        self._stats: list[_ArmStats] = [_ArmStats() for _ in range(self.K)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _epsilon(self, arm: int) -> float:
        """Confidence radius ε_i(t)."""
        n = self._stats[arm].count
        return math.inf if n == 0 else math.sqrt(2 * math.log(max(2, self.t)) / n)

    def _ucb_value(self, arm: int) -> float:
        """Compute the UCB-RSSR score B_i(t)."""
        stats = self._stats[arm]
        if stats.count == 0:
            return math.inf  # force at least one pull

        bar_x = stats.mean
        bar_v = stats.variance
        tilde_x = stats.second_moment
        eps = self._epsilon(arm)

        denom = (self.L + bar_v) * (self.L + bar_v - 3 * eps)
        if denom <= 0:
            return math.inf  # avoid division by small/negative values
        cb = ((bar_v + tilde_x + 2 * eps + self.L) * eps) / denom

        rssr_hat = (bar_x ** 2) / (self.L + bar_v)
        return rssr_hat + cb

    def _budget_feasible(self, arm: int) -> bool:
        if self.cost_budget is None:
            return True
        prospective_cost = self.C + self.costs[arm]
        horizon = self.t + 1  # include current round
        target_c = self.cost_budget
        if self.under_ut:
            target_c -= math.log(max(2, horizon)) / (self.omega ** 2 * horizon)
        return (prospective_cost / horizon) <= target_c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_arm(self) -> Optional[int]:
        """Choose the arm to trade this round or ``None`` to skip."""
        # Initial exploration: each arm once
        for i, s in enumerate(self._stats):
            if s.count == 0 and self._budget_feasible(i):
                return i

        # Scoring phase
        scores = np.array([
            self._ucb_value(i) if self._budget_feasible(i) else -math.inf
            for i in range(self.K)
        ])
        if np.all(np.isneginf(scores)):
            return None  # must skip to respect budget
        return int(np.argmax(scores))

    def update(self, arm: Optional[int], reward: float | None = None) -> None:
        """Record the outcome of the current *round*.

        Parameters
        ----------
        arm : int | None
            Traded arm index or ``None`` if the round was skipped.
        reward : float | None
            Realised return (required if a trade occurred).
        """
        # Advance global clock
        self.t += 1

        if arm is None:
            return  # skip – no stats or cost updates

        assert reward is not None, "Reward must be supplied when trading."
        self._stats[arm].update(reward)
        self.C += self.costs[arm]

    # ------------------------------------------------------------------
    # Diagnostics / public read-only properties
    # ------------------------------------------------------------------
    @property
    def average_cost(self) -> float:
        """Realised average cost \bar C(t) = C(t) / t.

        Returns zero before any rounds have been processed. This mirrors the
        budget constraint definition used internally by the algorithm.
        """
        return self.C / self.t if self.t > 0 else 0.0 