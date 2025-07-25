"""Download historical Yahoo Finance data for 20 large-cap stocks and
write a 10 000-row tidy CSV locally.

The output path is ``data/yfinance_20_10000.csv`` relative to the project root.
Each row contains ``Date, Ticker, AdjClose`` columns sorted chronologically and
lexicographically by ticker, matching the flat-table format in the user
specification.

Run once:
    python build_local_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
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
PERIOD = "2y"  # look-back horizon
INTERVAL = "1d"
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "yfinance_20_10000.csv"
ROWS = 10_000

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:
    print(f"Downloading {PERIOD} of daily data for {len(TICKERS)} tickers …")
    raw = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by="column", auto_adjust=False)

    # ``raw`` uses a MultiIndex (Attribute, Ticker) when multiple tickers are
    # requested; ``xs`` extracts the *Adj Close* slice.  In some yfinance
    # versions/flags the adjusted price may instead appear under *Close* if
    # ``auto_adjust=True``.  We guard for both situations.
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0).unique()
        attribute = "Adj Close" if "Adj Close" in first_level else "Close"
        data = raw.xs(attribute, level=0, axis=1)
    else:  # Single ticker → plain columns
        attribute = "Adj Close" if "Adj Close" in raw.columns else "Close"
        data = raw[[attribute]]

    # Forward-fill to handle non-synchronous trading holidays and drop rows that
    # are still NaN (e.g. very early dates before some IPOs).
    data = data.ffill().dropna()

    # Convert to tidy long-form table
    df = (
        data.reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
        .sort_values(["Date", "Ticker"], ignore_index=True)
    )

    print(f"Rows available after melt: {len(df):,}")
    df = df.head(ROWS)
    print(f"Keeping first {ROWS:,} rows → from {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV to {OUTPUT_CSV.resolve()} ({OUTPUT_CSV.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main() 