"""Run the UCB-RSSR budgeted bandit on real historical stock data.

Usage (in project root):
    python historical_run.py --period 2y --budget 0.6 --T 500

The script downloads *adjusted close* prices for a basket of large-cap US
stocks using *yfinance*, converts them to daily log returns, and feeds them to
:class:`bandit.ucb_rssr.UCBRSSRBudgetBandit`.  Only the return of the selected
stock is revealed to the learner each round, mirroring the trade-based
observability assumption.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from bandit.ucb_rssr import UCBRSSRBudgetBandit

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def download_returns(tickers: List[str], period: str = "2y") -> np.ndarray:
    """Download *Adj Close* and convert to log returns array (T, K)."""
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=False)[
        "Adj Close"
    ]
    # Forward-fill and drop initial NaNs
    data = data.ffill().dropna()
    # Compute log returns to stabilise heavy tails
    log_prices = np.log(data)
    log_returns = log_prices.diff().dropna()
    return log_returns.values.astype(float)


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------

def run_real_data_simulation(
    tickers: List[str],
    budget: float = 0.6,
    under_utilisation: bool = True,
    L: float = 1.0,
    omega: float = 0.1,
    period: str = "2y",
    max_steps: int | None = 10_000,
    seed: int | None = 0,
):
    returns = download_returns(tickers, period=period)
    T_full, K = returns.shape

    if max_steps is not None and max_steps < T_full:
        returns = returns[: max_steps]
        T_full = max_steps

    rng = np.random.default_rng(seed)

    # Constant deterministic per-trade costs (could be replaced by volatility-scaled costs)
    costs = rng.uniform(0.4, 0.8, size=K)

    bandit = UCBRSSRBudgetBandit(
        K=K,
        costs=costs,
        L=L,
        cost_budget=budget,
        under_utilisation=under_utilisation,
        omega=omega,
    )

    cumulative_reward = 0.0
    skips = 0

    for t in range(T_full):
        arm = bandit.select_arm()
        if arm is None:
            bandit.update(None)
            skips += 1
            continue

        reward = float(returns[t, arm])
        cumulative_reward += reward
        bandit.update(arm, reward)

    print("Simulation complete (real data)")
    print(f"Tickers:          {', '.join(tickers)}")
    print(f"Rounds simulated: {T_full}")
    print(f"Skips:            {skips} ({skips / T_full:.2%})")
    print(f"Cum. log-return:  {cumulative_reward:.4f}")
    print(f"Avg. log-return:  {cumulative_reward / T_full:.6f}")
    print(f"Final avg cost:   {bandit.average_cost:.4f} (budget={budget})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    DEFAULT_TICKERS = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "TSLA",
        "BRK-B",
        "JPM",
        "JNJ",
        "V",
        "PG",
        "UNH",
        "HD",
        "MA",
        "PFE",
        "XOM",
        "BAC",
        "KO",
        "DIS",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=0.6, help="Average cost budget")
    parser.add_argument("--period", type=str, default="2y", help="Lookback period for Yahoo Finance data (e.g., '2y', '1y')")
    parser.add_argument("--max_steps", type=int, default=10_000, help="Maximum number of trading days to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    run_real_data_simulation(
        tickers=DEFAULT_TICKERS,
        budget=args.budget,
        period=args.period,
        max_steps=args.max_steps,
        seed=args.seed,
    ) 