"""Executable demo for the UCB-RSSR budgeted bandit.

Run with:
    $ python main.py

The script simulates K Gaussian-return stocks and lets the bandit select trades
for T rounds while respecting an average cost constraint.
"""

from __future__ import annotations

import argparse

import numpy as np

from bandit.environment import StockEnvironment
from bandit.ucb_rssr import UCBRSSRBudgetBandit
from bandit.portfolio_ucb_rssr import UCBRSSRPortfolioBandit


def run_simulation(
    K: int = 5,
    T: int = 2000,
    seed: int | None = 0,
    cost_budget: float | None = 0.6,
    mode: str = "single",
) -> None:
    """Simulate the bandit for *T* rounds and print summary stats."""

    rng = np.random.default_rng(seed)

    # True means and standard deviations for each stock
    mus = rng.uniform(0.0, 0.05, size=K)  # small positive drift
    sigmas = rng.uniform(0.01, 0.05, size=K)

    # Per-trade deterministic costs (e.g. 0.5 – 1.0)
    costs = rng.uniform(0.4, 0.8, size=K)

    env = StockEnvironment(mus, sigmas, costs, seed=seed)

    if mode == "single":
        bandit = UCBRSSRBudgetBandit(
            K=K,
            costs=costs,
            L=1.0,
            cost_budget=cost_budget,
            under_utilisation=True,
            omega=0.1,
        )

        cumulative_reward = 0.0
        skips = 0

        for _ in range(T):
            arm = bandit.select_arm()
            if arm is None:
                bandit.update(None)
                skips += 1
                continue

            reward = env.sample_return(arm)
            cumulative_reward += reward
            bandit.update(arm, reward)

    elif mode == "portfolio":
        # Construct simple weight vectors: each single arm + uniform portfolio
        weight_vectors = [np.eye(1, K, k).flatten() for k in range(K)]
        weight_vectors.append(np.ones(K) / K)  # uniform index

        # Example: two resource dimensions with independent costs
        cost_matrix = np.vstack([
            costs,                      # resource 1 (same as single cost)
            rng.uniform(0.2, 0.5, K),   # resource 2
        ])

        bandit = UCBRSSRPortfolioBandit(
            weight_vectors=weight_vectors,
            costs=cost_matrix,
            resource_budgets=[cost_budget, 0.45],
            L=1.0,
            under_utilisation=True,
            omega=0.1,
            replenish_rates=[0.0, 0.0],
            primal_dual=False,
        )

        cumulative_reward = 0.0
        skips = 0

        for _ in range(T):
            b_idx = bandit.select_portfolio()
            if b_idx is None:
                bandit.update(None)
                skips += 1
                continue

            # Sample returns for all K arms once (statistics only use active ones)
            rewards = env.sample_returns(range(K))
            # Weighted portfolio reward for logging
            portfolio_reward = float(np.dot(weight_vectors[b_idx], rewards))
            cumulative_reward += portfolio_reward
            bandit.update(b_idx, rewards)

    else:
        raise ValueError("mode must be 'single' or 'portfolio'")

    print("Simulation complete")
    print(f"Total rounds:      {T}")
    print(f"Skips:             {skips} ({skips / T:.2%})")
    print(f"Cumulative reward: {cumulative_reward:.4f}")
    print(f"Average reward:    {cumulative_reward / T:.6f}")
    print("Average costs per resource:")
    if mode == "single":
        print(f"  Resource 0: {bandit.average_cost:.4f} (budget={cost_budget})")
    else:
        for j, c in enumerate(bandit.average_cost):
            print(f"  Resource {j}: {c:.4f} (budget={bandit.cost_budget[j] if bandit.cost_budget is not None else 'N/A'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=5, help="Number of stocks (arms)")
    parser.add_argument("--T", type=int, default=2000, help="Horizon length (rounds)")
    parser.add_argument("--budget", type=float, default=0.6, help="Average cost budget")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--mode", choices=["single", "portfolio"], default="single", help="Simulation mode")
    args = parser.parse_args()

    run_simulation(K=args.K, T=args.T, seed=args.seed, cost_budget=args.budget, mode=args.mode) 