"""Portfolio (multi-stock) UCB-RSSR bandit under multi-resource constraints.

This module extends the single-arm implementation in ``ucb_rssr.py`` to the
setting described in the research note:

* Each *action* activates a *base* – a fixed weight vector over the K stocks
  (e.g. one-hot vectors reduce to the single-stock model).
* Cost / risk may comprise *m ≥ 1* resources.  An average-cost constraint is
  enforced component-wise.  Optional deterministic *replenishment* per round
  can be specified.
* When some costs can be negative (restorative actions), an optional
  primal-dual update (BwRK-style) is available; the dual variables enter the
  selection rule as Lagrange multipliers.

Only i.i.d. returns are assumed, and statistics *per stock* are updated on
activation (trade-based learning).  Confidence bonuses follow exactly the
UCB-RSSR expression from the original algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional, Iterable, List

import numpy as np

# -----------------------------------------------------------------------------
# Helpers copied from the single-stock module to avoid circular imports
# -----------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division that avoids ``ZeroDivisionError`` and returns *default* instead."""
    return num / den if den != 0 else default


@dataclass
class _ArmStats:
    """Sufficient statistics for one arm (stock)."""

    count: int = 0
    sum_returns: float = 0.0
    sum_squares: float = 0.0  # second raw moment accumulator

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------
    def update(self, reward: float) -> None:
        self.count += 1
        self.sum_returns += reward
        self.sum_squares += reward ** 2

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------
    @property
    def mean(self) -> float:
        return _safe_div(self.sum_returns, self.count)

    @property
    def second_moment(self) -> float:
        return _safe_div(self.sum_squares, self.count)

    @property
    def variance(self) -> float:
        return max(1e-12, self.second_moment - self.mean ** 2)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class UCBRSSRPortfolioBandit:
    """UCB-RSSR with weighted-portfolio actions and multi-resource constraints."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        weight_vectors: Sequence[Sequence[float]] | np.ndarray,
        costs: np.ndarray,
        resource_budgets: Sequence[float] | float | np.ndarray | None = None,
        L: float = 1.0,
        under_utilisation: bool = True,
        omega: float = 0.1,
        replenish_rates: Sequence[float] | None = None,
        primal_dual: bool = False,
        eta_dual: float = 0.01,
    ) -> None:
        """Parameters
        ----------
        weight_vectors : Sequence of length |B| each containing K floats
            Definition of the bases / portfolios that can be activated.
            Each vector ``w`` should satisfy ``w_i ≥ 0`` and ``sum w_i = 1``
            (the latter is not strictly required by the maths but typical).
        costs : (m, K) or (K,) array
            Deterministic per-resource per-arm cost matrix.  ``m`` denotes the
            number of resource dimensions.  If 1-D, a single resource is
            assumed.
        resource_budgets : array-like length m or scalar or None
            Target average cost budget for each resource.  ``None`` disables
            budget gating entirely (pure UCB-RSSR).
        L : float, default=1.0
            Regularisation constant in RSSR definition.
        under_utilisation : bool, default=True
            Whether to apply the SUAK under-utilisation correction.
        omega : float, default=0.1
            Lower bound on minimal cost gap for the under-utilisation term.
        replenish_rates : array-like length m or None, default=None
            Deterministic per-round replenishment of each resource (side
            constraints model).  If ``None``, no replenishment occurs.
        primal_dual : bool, default=False
            If *True*, activates a BwRK-style primal-dual rule that keeps track
            of Lagrange multipliers for each resource.  The selection rule then
            maximises "score – λ·cost" instead of the raw score, and the duals
            update via projected gradient ascent with step size ``eta_dual``.
        eta_dual : float, default=0.01
            Step size for the dual variable update when ``primal_dual`` is
            enabled.
        """
        # Normalise weight list to a (B, K) array
        self.W = np.asarray(weight_vectors, dtype=float)  # shape (B, K)
        if self.W.ndim == 1:
            self.W = self.W.reshape(1, -1)
        self.B, self.K = self.W.shape

        # Costs
        self.costs = np.asarray(costs, dtype=float)
        if self.costs.ndim == 1:
            self.costs = self.costs.reshape(1, -1)  # (1, K)
        self.m = self.costs.shape[0]  # number of resources
        assert self.costs.shape == (self.m, self.K)

        # Budgets and replenishment handling
        self.cost_budget: Optional[np.ndarray]
        if resource_budgets is None:
            self.cost_budget = None
        else:
            self.cost_budget = np.asarray(resource_budgets, dtype=float).reshape(-1)
            if self.cost_budget.size == 1 and self.m > 1:
                self.cost_budget = np.repeat(self.cost_budget, self.m)
            assert self.cost_budget.size == self.m, "Budget vector length mismatch"

        self.replenish = None if replenish_rates is None else np.asarray(replenish_rates, dtype=float).reshape(-1)
        if self.replenish is not None:
            if self.replenish.size == 1 and self.m > 1:
                self.replenish = np.repeat(self.replenish, self.m)
            assert self.replenish.size == self.m, "Replenish rate vector length mismatch"

        self.L = float(L)
        self.under_ut = bool(under_utilisation)
        self.omega = float(omega)

        # Primal-dual variables
        self.primal_dual = bool(primal_dual)
        self.eta_dual = float(eta_dual)
        self._lambda = np.zeros(self.m)  # initial dual multipliers

        # Time index
        self.t: int = 0
        # Cumulative realised cost per resource
        self.C = np.zeros(self.m)
        # Per-arm stats
        self._stats: List[_ArmStats] = [_ArmStats() for _ in range(self.K)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # Confidence radius per arm
    def _epsilon(self, arm: int) -> float:
        n = self._stats[arm].count
        return math.inf if n == 0 else math.sqrt(2 * math.log(max(2, self.t)) / n)

    # UCB-RSSR per arm
    def _ucb_value_arm(self, arm: int) -> float:
        stats = self._stats[arm]
        if stats.count == 0:
            return math.inf
        bar_x = stats.mean
        bar_v = stats.variance
        tilde_x = stats.second_moment
        eps = self._epsilon(arm)
        denom = (self.L + bar_v) * (self.L + bar_v - 3 * eps)
        if denom <= 0:
            return math.inf
        cb = ((bar_v + tilde_x + 2 * eps + self.L) * eps) / denom
        rssr_hat = (bar_x ** 2) / (self.L + bar_v)
        return rssr_hat + cb

    # Portfolio score
    def _portfolio_score(self, idx: int) -> float:
        w = self.W[idx]
        # Linear aggregation of means and variances according to weights
        means = np.array([s.mean for s in self._stats])
        vars_ = np.array([s.variance for s in self._stats])
        # If some arms unseen, treat their score as +inf to enforce exploration
        unseen = np.array([s.count == 0 for s in self._stats])
        if unseen.any() and (w[unseen] > 0).any():
            return math.inf
        num = (w @ means) ** 2
        den = self.L + (w ** 2 @ vars_)
        portfolio_rssr = num / den
        # Confidence bonus – conservative sum of weighted arm bonuses
        cb = float(np.sum(w * np.array([self._ucb_value_arm(i) - ((self._stats[i].mean ** 2) / (self.L + self._stats[i].variance)) for i in range(self.K)])))
        return portfolio_rssr + cb

    # Portfolio cost vector (m,)
    def _portfolio_cost(self, idx: int) -> np.ndarray:
        return self.costs @ self.W[idx]

    # Feasibility check (anytime constraint)
    def _budget_feasible(self, idx: int) -> bool:
        if self.cost_budget is None:
            return True
        prospective_C = self.C + self._portfolio_cost(idx)
        horizon = self.t + 1
        target = self.cost_budget.copy()
        if self.under_ut:
            target -= math.log(max(2, horizon)) / (self.omega ** 2 * horizon)
        return np.all((prospective_C / horizon) <= target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_portfolio(self) -> Optional[int]:
        """Return the index of the chosen *base* for this round or ``None`` to skip."""
        # Force initial exploration for any unseen arm contained in a base that is budget-feasible
        for b_idx in range(self.B):
            w = self.W[b_idx]
            for arm_idx, weight in enumerate(w):
                if weight > 0 and self._stats[arm_idx].count == 0 and self._budget_feasible(b_idx):
                    return b_idx
        # Compute objective for each base
        scores = np.full(self.B, -math.inf)
        for b in range(self.B):
            if not self._budget_feasible(b):
                continue
            score = self._portfolio_score(b)
            if self.primal_dual:
                score -= float(self._lambda @ self._portfolio_cost(b))
            scores[b] = score
        if np.all(np.isneginf(scores)):
            return None  # must skip
        return int(np.argmax(scores))

    def update(self, b_idx: Optional[int], rewards: Optional[Sequence[float]] = None) -> None:
        """Update statistics after the chosen action.

        Parameters
        ----------
        b_idx : int | None
            Index of the activated base or ``None`` if the round was skipped.
        rewards : Sequence[float] | None
            Observed per-arm returns for the *K* arms **in order**.  Only those
            entries corresponding to positive weights in the selected base are
            used.  Required when ``b_idx`` is not ``None``.
        """
        # Advance time and apply replenishment
        self.t += 1
        if self.replenish is not None:
            self.C = self.C - self.replenish
        # Skip handling
        if b_idx is None:
            # Dual update still occurs using zero cost diff if enabled
            if self.primal_dual:
                self._dual_update(np.zeros(self.m))
            return

        assert rewards is not None, "Per-arm rewards must be supplied for a trade."
        rewards = np.asarray(rewards, dtype=float)
        assert rewards.size == self.K, "rewards must contain K elements"

        # Update stats for each activated arm (weight > 0)
        active = self.W[b_idx] > 0
        for i in np.where(active)[0]:
            self._stats[i].update(float(rewards[i]))

        # Update cumulative cost
        cost_vec = self._portfolio_cost(b_idx)
        self.C += cost_vec

        # Dual variable update
        if self.primal_dual:
            self._dual_update(cost_vec)

    # ------------------------------------------------------------------
    # Dual (BwRK-style) helper
    # ------------------------------------------------------------------
    def _dual_update(self, cost_vec: np.ndarray) -> None:
        """Projected gradient ascent step on the dual variables."""
        if self.cost_budget is None:
            return  # no budget – nothing to do
        gradient = cost_vec - self.cost_budget
        self._lambda = np.maximum(0.0, self._lambda + self.eta_dual * gradient)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def average_cost(self) -> np.ndarray:
        return self.C / self.t if self.t > 0 else np.zeros(self.m)

    @property
    def dual_variables(self) -> np.ndarray:
        return self._lambda.copy() 