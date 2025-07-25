"""Download older historical Yahoo Finance data for 20 tickers and
store the first 50 000 (date,ticker) rows in a tidy CSV.

Output: ``data/yfinance_20_50000.csv``

Rationale: complements the 10 000-row file by providing a larger sample from an
earlier time window (non-overlapping with the 2-year slice).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
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

PERIOD = "10y"  # longer look-back to have ≥ 50 000 rows
INTERVAL = "1d"
ROWS = 50_000
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "yfinance_20_50000.csv"


def main() -> None:
    print(f"Fetching {PERIOD} of daily data for {len(TICKERS)} tickers…")
    raw = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by="column", auto_adjust=False)

    # Extract Adjusted Close (fallback to Close)
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique()
        attr = "Adj Close" if "Adj Close" in level0 else "Close"
        data = raw.xs(attr, level=0, axis=1)
    else:
        attr = "Adj Close" if "Adj Close" in raw.columns else "Close"
        data = raw[[attr]]

    data = data.ffill().dropna()

    df = (
        data.reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
        .sort_values(["Date", "Ticker"], ignore_index=True)
    )

    print(f"Total rows available: {len(df):,}")
    df = df.head(ROWS)
    print(
        f"Using first {ROWS:,} rows → {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV.resolve()} ({OUTPUT_CSV.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main() 