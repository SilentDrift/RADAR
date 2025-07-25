"""Run Block-UCB-RSSR portfolio bandit on a local CSV dataset.

Example:
    python run_csv_simulation.py \
        --csv data/yfinance_100_100000.csv \
        --budget 0.6 --block 20 --T 5000
"""

from __future__ import annotations

import argparse
import numpy as np
from tqdm.auto import tqdm

from bandit.block_ucb_rssr import BlockUCBRSSRPortfolioBandit
from bandit.csv_environment import CsvReturnEnvironment
from bandit.user_manager import UCBRSSRPortfolioManager

# -----------------------------------------------------------------------------
# Helper to build sector mapping if present in CSV
# -----------------------------------------------------------------------------


def load_tickers(csv_path: str):
    import pandas as pd
    df_head = pd.read_csv(csv_path, nrows=0)
    usecols = ["Ticker"]
    if "Sector" in df_head.columns:
        usecols.append("Sector")
    df_full = pd.read_csv(csv_path, usecols=usecols)
    tickers = sorted(df_full["Ticker"].unique())
    # Build sector map if available
    sector_map = {}
    sector_codes = {}
    if "Sector" in df_full.columns:
        sector_map = df_full.drop_duplicates("Ticker").set_index("Ticker")["Sector"].to_dict()
        unique_sectors = sorted(set(sector_map.values()))
        sector_codes = {s: i for i, s in enumerate(unique_sectors)}
    return tickers, sector_map, sector_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to tidy CSV with Date,Ticker,AdjClose")
    parser.add_argument("--budget", type=float, default=0.6, help="Average cost budget per resource")
    parser.add_argument("--T", type=int, default=5000, help="Number of rounds to simulate")
    parser.add_argument("--block", type=int, default=20, help="Block size for Block-UCB")
    parser.add_argument("--algo", choices=["block", "user"], default="block", help="Algorithm variant to run")
    parser.add_argument("--cycle", action="store_true", help="Cycle dataset to reach T even if shorter")
    parser.add_argument("--alpha", type=float, default=0.05, help="Constant step size for mean/variance updates (EMA)")
    parser.add_argument("--out_prefix", type=str, default="results", help="Prefix for output CSV files")
    args = parser.parse_args()

    tickers, sector_map, sector_codes = load_tickers(args.csv)
    env = CsvReturnEnvironment(args.csv, tickers)

    K = len(tickers)
    rng = np.random.default_rng(0)
    costs = rng.uniform(0.4, 0.8, size=K)

    # Simple bases: single-stock only (could add portfolios)
    weight_vectors = [np.eye(1, K, k).flatten() for k in range(K)]

    if args.algo == "block":
        bandit = BlockUCBRSSRPortfolioBandit(
            weight_vectors=weight_vectors,
            costs=costs,
            resource_budgets=[args.budget],
            block_size=args.block,
        )
        def bandit_step():
            b_idx = bandit.select_portfolio()
            if b_idx is None:
                bandit.update(None)
                return None
            arm = b_idx
            reward = env.current_returns_dict()[arm]
            rewards_vec = np.zeros(K)
            rewards_vec[arm] = reward
            bandit.update(b_idx, rewards_vec)
            return arm, reward
    else:
        user_mgr = UCBRSSRPortfolioManager(costs=costs, budget_c=args.budget, L=1.0, init_trades=1, seed=0, alpha=args.alpha)
        def bandit_step():
            returns_dict = env.current_returns_dict()
            chosen = user_mgr.step(returns_dict, under_utilisation=False)
            if chosen is None:
                return None
            return chosen, returns_dict[chosen]

    cumulative_reward = 0.0
    skips = 0

    actions_log = []  # (date, chosen_ticker)
    est_records = []  # dict per date with mean/var per stock

    for t in tqdm(range(args.T), desc="Simulating", unit="step"):
        if not env.has_next():
            if args.cycle:
                env.reset()
            else:
                break
        current_date = env.current_date()
        step_result = bandit_step()
        if step_result is None:
            skips += 1
            actions_log.append((current_date, None))
        else:
            arm, reward = step_result
            cumulative_reward += reward
            ticker = tickers[arm]
            actions_log.append((current_date, ticker))
        env.advance()

        # Log current estimates after update (wide format)
        record = {"Date": current_date}
        if args.algo == "block":
            for idx, stat in enumerate(bandit._stats):
                record[f"{tickers[idx]}_mean"] = stat.mean
                record[f"{tickers[idx]}_var"] = stat.variance
        else:
            for idx in range(K):
                record[f"{tickers[idx]}_mean"] = user_mgr.mean(idx)
                record[f"{tickers[idx]}_var"] = user_mgr.variance(idx)
        est_records.append(record)

    T_sim = user_mgr.current_time if args.algo == "user" else bandit.t
    print("CSV simulation complete")
    print(f"Rounds simulated: {T_sim}")
    print(f"Skips:           {skips} ({skips/T_sim:.2%})")
    print(f"Cum. log-return: {cumulative_reward:.4f}")
    print(f"Avg. log-return: {cumulative_reward / T_sim:.6f}")
    avg_cost_val = user_mgr.average_cost() if args.algo == "user" else bandit.average_cost[0]
    print(f"Avg. cost:       {avg_cost_val:.4f} (budget={args.budget})")

    # ------------------------------------------------------------------
    # Persist logs
    # ------------------------------------------------------------------
    import pandas as pd
    actions_df = pd.DataFrame(actions_log, columns=["Date", "ChosenTicker"])
    est_df = pd.DataFrame(est_records)

    actions_path = f"{args.out_prefix}_chosen_actions.csv"
    est_path = f"{args.out_prefix}_stock_estimates.csv"
    actions_df.to_csv(actions_path, index=False)
    est_df.to_csv(est_path, index=False)
    print(f"Action log written to {actions_path}")
    print(f"Estimates log written to {est_path}")


if __name__ == "__main__":
    main() 