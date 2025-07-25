"""Generate a second, larger slice (100 000 rows) for the same 100-ticker
universe, taken from a *different* time region (latest end of the 10-year
history) so it does not overlap with ``yfinance_100_50000.csv``.

Output → ``data/yfinance_100_100000.csv``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from build_local_dataset_100_50k import SECTOR_MAP, fetch_adj_close, TICKERS  # reuse helpers

ROWS = 100_000
PERIOD = "10y"  # ensure enough data for >100k rows
INTERVAL = "1d"
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "yfinance_100_100000.csv"


def main() -> None:
    print(f"Downloading {PERIOD} daily data for {len(TICKERS)} tickers …")
    data = fetch_adj_close(TICKERS)

    df = (
        data.reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
        .sort_values(["Date", "Ticker"], ignore_index=True)
    )
    print(f"Rows available: {len(df):,}")

    if len(df) <= ROWS:
        raise RuntimeError("Not enough rows to create dataset – try a longer PERIOD")

    # Take the *last* ROWS rows to avoid overlap with earlier slice
    df = df.tail(ROWS).reset_index(drop=True)
    print(
        f"Using last {ROWS:,} rows → {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    # Sector annotations
    df["Sector"] = df["Ticker"].map(SECTOR_MAP).fillna("Others")
    sector_codes = {s: i for i, s in enumerate(sorted(df["Sector"].unique()))}
    df["SectorId"] = df["Sector"].map(sector_codes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset → {OUTPUT_CSV.resolve()} ({OUTPUT_CSV.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main() 