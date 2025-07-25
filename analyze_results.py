import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_price_matrix(csv_path: Path, tickers: list[str]) -> pd.DataFrame:
    """Load *adj-close* prices and pivot to a Date × Ticker matrix."""
    df = pd.read_csv(csv_path, usecols=["Date", "Ticker", "AdjClose"])
    df["Date"] = pd.to_datetime(df["Date"])  # ensure dtype
    prices = (
        df[df["Ticker"].isin(tickers)]
        .pivot(index="Date", columns="Ticker", values="AdjClose")
        .sort_index()
    )
    # Ensure column order matches *tickers* exactly
    prices = prices[tickers]
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from price matrix (dropping the first row)."""
    return np.log(prices).diff().dropna()


# -----------------------------------------------------------------------------
# Main analysis routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse RADAR simulation results – build summary tables and graphs.")
    parser.add_argument("--actions", default="results/full100k2_chosen_actions.csv", help="CSV file produced by run_csv_simulation.py containing Date,ChosenTicker columns")
    parser.add_argument("--data", default="data/yfinance_100_100000.csv", help="Underlying tidy price dataset (Date,Ticker,AdjClose)")
    parser.add_argument("--out", default="analysis", help="Output directory for tables & figures")
    parser.add_argument("--top", type=int, default=20, help="Number of top tickers to plot in frequency bar chart")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load action log
    # ------------------------------------------------------------------
    actions_df = pd.read_csv(args.actions)
    actions_df["Date"] = pd.to_datetime(actions_df["Date"])

    # Remove skipped rounds (NaN ticker)
    actions_df = actions_df.dropna(subset=["ChosenTicker"]).reset_index(drop=True)

    tickers = sorted(actions_df["ChosenTicker"].unique())

    # ------------------------------------------------------------------
    # Price / return data
    # ------------------------------------------------------------------
    prices = load_price_matrix(Path(args.data), tickers)
    log_returns = compute_log_returns(prices)

    # Align actions with returns (there might be leading dates without returns)
    actions_df = actions_df[actions_df["Date"].isin(log_returns.index)]

    # Vectorised lookup of returns corresponding to chosen ticker / date
    returns_lookup = log_returns.stack()
    actions_df["Return"] = actions_df.apply(
        lambda row: returns_lookup.loc[row["Date"], row["ChosenTicker"]], axis=1
    )
    actions_df["CumLogReturn"] = actions_df["Return"].cumsum()

    # ------------------------------------------------------------------
    # TABLES
    # ------------------------------------------------------------------
    freq_tbl = actions_df["ChosenTicker"].value_counts().rename_axis("Ticker").to_frame("Count")
    freq_tbl["Percentage"] = freq_tbl["Count"] / freq_tbl["Count"].sum() * 100
    freq_tbl.to_csv(tbl_dir / "ticker_frequency.csv")

    summary_tbl = pd.DataFrame(
        {
            "Rounds": [len(actions_df)],
            "UniqueTickers": [freq_tbl.shape[0]],
            "CumLogReturn": [actions_df["CumLogReturn"].iloc[-1]],
            "AvgLogReturn": [actions_df["Return"].mean()],
        }
    )
    summary_tbl.to_csv(tbl_dir / "summary.csv", index=False)

    # ------------------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # 1. Cumulative log-return over time
    plt.figure(figsize=(10, 6))
    plt.plot(actions_df["Date"], actions_df["CumLogReturn"], label="Cumulative log-return", color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Cum. log-return")
    plt.title("Cumulative log-return of chosen actions")
    plt.tight_layout()
    plt.savefig(fig_dir / "cumulative_log_return.png", dpi=150)
    plt.close()

    # 2. Frequency bar chart (top N)
    top_n = freq_tbl.head(args.top).iloc[::-1]  # reverse for horizontal plot
    plt.figure(figsize=(8, max(4, 0.3 * args.top)))
    sns.barplot(x="Count", y=top_n.index, data=top_n, palette="viridis")
    plt.title(f"Top {args.top} most frequently chosen tickers")
    plt.xlabel("Selections")
    plt.ylabel("Ticker")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_ticker_frequency.png", dpi=150)
    plt.close()

    # 3. Distribution of daily returns (histogram)
    plt.figure(figsize=(8, 5))
    sns.histplot(actions_df["Return"], bins=50, kde=True, color="tab:orange")
    plt.title("Distribution of daily log-returns for chosen actions")
    plt.xlabel("Daily log-return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / "return_distribution.png", dpi=150)
    plt.close()

    # 4. Rolling (30-day) average log-return
    window = 30
    roll_mean = actions_df.set_index("Date")["Return"].rolling(window).mean()
    plt.figure(figsize=(10, 6))
    roll_mean.plot(color="tab:green")
    plt.title(f"{window}-day rolling average of log-returns")
    plt.xlabel("Date")
    plt.ylabel("Log-return (mean)")
    plt.tight_layout()
    plt.savefig(fig_dir / "rolling_mean_return.png", dpi=150)
    plt.close()

    # 5. Distribution of number of selections per ticker
    plt.figure(figsize=(8, 5))
    sns.histplot(freq_tbl["Count"], bins=20, color="tab:purple")
    plt.title("Distribution of selections per ticker")
    plt.xlabel("Selections per ticker")
    plt.ylabel("Number of tickers")
    plt.tight_layout()
    plt.savefig(fig_dir / "selection_count_distribution.png", dpi=150)
    plt.close()

    # 6. Bar chart of selections per ticker (all tickers)
    plt.figure(figsize=(10, max(4, 0.25 * len(freq_tbl))))
    sns.barplot(x="Count", y=freq_tbl.index, data=freq_tbl, color="skyblue")
    plt.title("Number of selections per ticker (sorted)")
    plt.xlabel("Selections")
    plt.ylabel("Ticker")
    # Annotate counts at end of bars
    for i, (count) in enumerate(freq_tbl["Count"]):
        plt.text(count + 5, i, f"{count}", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "ticker_selection_counts.png", dpi=150)
    plt.close()

    print("Analysis complete. Tables saved to", tbl_dir.resolve())
    print("Figures saved to", fig_dir.resolve())


if __name__ == "__main__":
    main() 