from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Dict, Any

import numpy as np

__all__ = ["UCBRSSRPortfolioManager"]


class UCBRSSRPortfolioManager:
    """User-supplied single-stock UCB-RSSR manager with average-cost skipping."""

    def __init__(
        self,
        costs: Sequence[float],
        budget_c: float,
        L: float = 1.0,
        init_trades: int = 1,
        seed: Optional[int] = None,
        alpha: float = 0.05,
    ):
        self.random = random.Random(seed)
        self.costs = np.asarray(costs, dtype=float)
        self.K = self.costs.size
        self.budget_c = float(budget_c)
        self.L = float(L)
        self.init_trades = int(init_trades)

        self.N = np.zeros(self.K, dtype=int)
        # Exponential-moving statistics (constant step size *alpha*)
        self.mean_est = np.zeros(self.K)        # \bar X_i – running mean
        self.sec_raw_est = np.zeros(self.K)     # \tilde X_i – running second raw moment
        self.alpha = float(alpha)

        self.total_cost = 0.0
        self.current_time = 0
        self.action_history: List[Optional[int]] = []
        self.skips = 0

    # ---------- stats helpers ---------- #
    def mean(self, i: int) -> float:
        return float(self.mean_est[i])

    def second_raw_moment(self, i: int) -> float:
        return float(self.sec_raw_est[i])

    def variance(self, i: int) -> float:
        if self.N[i] == 0:
            return 0.0
        m = self.mean(i)
        return max(self.second_raw_moment(i) - m * m, 0.0)

    def epsilon(self, i: int) -> float:
        if self.N[i] == 0 or self.current_time < 1:
            return float("inf")
        return math.sqrt(2.0 * math.log(self.current_time) / self.N[i])

    # The classic UCB-RSSR bonus is replaced by plain epsilon for exploration.
    def cb_term(self, i: int) -> float:
        return self.epsilon(i)

    def q_hat(self, i: int) -> float:
        if self.N[i] == 0:
            return 0.0
        var_i = self.variance(i)
        m = self.mean(i)
        return (m * m) / (self.L + var_i)

    def index(self, i: int) -> float:
        """Selection index = q_hat + epsilon (simple exploration)."""
        return self.q_hat(i) + self.epsilon(i)

    # ---------- budget feasibility ---------- #
    def _omega_est(self) -> float:
        positive = np.sort(np.unique(self.costs[self.costs > 0]))
        if positive.size < 2:
            return 1.0
        gaps = np.diff(positive)
        return max(float(np.min(gaps)), 1e-6)

    def can_trade(self, stock: int, under_utilisation: bool = False) -> bool:
        t = self.current_time + 1
        projected = (self.total_cost + self.costs[stock]) / t
        if not under_utilisation:
            return projected <= self.budget_c
        shrink = math.log(max(t, 2)) / (self._omega_est() ** 2 * t)
        eff_budget = max(self.budget_c - shrink, 0.0)
        return projected <= eff_budget

    # ---------- main decision ---------- #
    def select_action(self, under_utilisation: bool = False) -> Optional[int]:
        self.current_time += 1
        # initialisation – ensure at least init_trades per arm
        for i in range(self.K):
            if self.N[i] < self.init_trades and self.can_trade(i, under_utilisation):
                return i

        # Vectorised score computation
        q_hat = (self.mean_est ** 2) / (self.L + np.maximum(self.sec_raw_est - self.mean_est ** 2, 1e-12))
        eps = np.where(self.N == 0, np.inf, np.sqrt(2.0 * np.log(max(self.current_time, 2)) / self.N))
        scores = q_hat + eps

        # Feasibility filter
        feasible = np.array([self.can_trade(i, under_utilisation) for i in range(self.K)])
        if not feasible.any():
            self.skips += 1
            return None

        # Pick best feasible arm (ties broken randomly for stability)
        feas_scores = np.where(feasible, scores, -np.inf)
        best_idx = int(np.argmax(feas_scores))
        return best_idx

    def observe_trade(self, stock: int, reward: float):
        self.N[stock] += 1
        a = self.alpha
        self.mean_est[stock] = (1 - a) * self.mean_est[stock] + a * reward
        self.sec_raw_est[stock] = (1 - a) * self.sec_raw_est[stock] + a * (reward ** 2)
        self.total_cost += float(self.costs[stock])

    def step(self, returns_dict: Dict[int, float], under_utilisation: bool = False) -> Optional[int]:
        chosen = self.select_action(under_utilisation)
        if chosen is not None:
            self.observe_trade(chosen, returns_dict[chosen])
        self.action_history.append(chosen)
        return chosen

    # ---------- diagnostics ---------- #
    def average_cost(self) -> float:
        return float(self.total_cost / self.current_time) if self.current_time > 0 else 0.0 