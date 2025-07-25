"""Build a 50 000-row tidy CSV for 100 US large-cap stocks, including sector info.

Output → ``data/yfinance_100_50000.csv``
Each row: ``Date, Ticker, AdjClose, Sector, SectorId``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "AMZN": "Retail", "GOOGL": "Communications", "META": "Communications",
    "NVDA": "Tech", "BRK-B": "Finance", "JPM": "Finance", "JNJ": "Non Durable", "V": "Finance",
    "PG": "Non Durable", "UNH": "Non Durable", "HD": "Retail", "MA": "Finance", "XOM": "Oil",
    "KO": "Non Durable", "PFE": "Non Durable", "BAC": "Finance", "DIS": "Communications", "CSCO": "Tech",
    "PEP": "Non Durable", "AVGO": "Tech", "ADBE": "Tech", "NFLX": "Communications", "MRK": "Non Durable",
    "TMO": "Others", "CRM": "Tech", "ABBV": "Non Durable", "ABT": "Non Durable", "WMT": "Retail",
    "ACN": "Tech", "LIN": "Others", "CVX": "Oil", "MCD": "Retail", "DHR": "Others",
    "COST": "Retail", "AMD": "Tech", "WFC": "Finance", "TXN": "Tech", "VZ": "Communications",
    "INTC": "Tech", "NEE": "Others", "HON": "Durable", "AMAT": "Tech", "PM": "Non Durable",
    "AMGN": "Non Durable", "INTU": "Tech", "LOW": "Retail", "UNP": "Durable", "QCOM": "Tech",
    "CAT": "Durable", "IBM": "Tech", "MS": "Finance", "GS": "Finance", "RTX": "Durable",
    "SBUX": "Retail", "NOW": "Tech", "SPGI": "Finance", "GE": "Durable", "BKNG": "Retail",
    "BLK": "Finance", "LMT": "Durable", "AMT": "Others", "ISRG": "Others", "MO": "Non Durable",
    "DE": "Durable", "MDT": "Others", "ADI": "Tech", "EL": "Non Durable", "SYK": "Others",
    "C": "Finance", "REGN": "Non Durable", "T": "Communications", "GILD": "Non Durable", "MMC": "Finance",
    "BDX": "Others", "PGR": "Finance", "ADP": "Tech", "CI": "Finance", "DUK": "Others",
    "CB": "Finance", "SO": "Others", "SHW": "Non Durable", "EQIX": "Others", "ICE": "Finance",
    "APD": "Others", "CL": "Non Durable", "MU": "Tech", "ZTS": "Non Durable", "PLD": "Others",
    "PNC": "Finance", "NSC": "Durable", "FDX": "Durable", "GM": "Durable", "TGT": "Retail",
    "USB": "Finance", "AON": "Finance", "FI": "Tech", "EOG": "Oil", "CME": "Finance",
    "VRTX": "Non Durable", "ALL": "Finance", "AIG": "Finance", "HUM": "Finance",
}

TICKERS = sorted(SECTOR_MAP.keys())  # 100 tickers
ROWS = 50_000
PERIOD = "5y"  # 5 years provides ~1250 trading days > 50k rows
INTERVAL = "1d"
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "yfinance_100_50000.csv"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def fetch_adj_close(tickers):
    raw = yf.download(tickers, period=PERIOD, interval=INTERVAL, group_by="column", auto_adjust=False)
    # Extract Adjusted Close or Close as fallback
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique()
        attr = "Adj Close" if "Adj Close" in level0 else "Close"
        data = raw.xs(attr, level=0, axis=1)
    else:
        attr = "Adj Close" if "Adj Close" in raw.columns else "Close"
        data = raw[[attr]]
    data = data.ffill()
    # Drop tickers that remain entirely NaN (failed download)
    data = data.dropna(axis=1, how="all")
    return data.dropna(how="any")


def main() -> None:
    print(f"Downloading {PERIOD} daily data for {len(TICKERS)} tickers…")
    data = fetch_adj_close(TICKERS)

    df = (
        data.reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
        .sort_values(["Date", "Ticker"], ignore_index=True)
    )
    print(f"Total rows available after melt: {len(df):,}")

    df = df.head(ROWS)
    print(
        f"Keeping first {ROWS:,} rows → {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    # Sector & codes
    df["Sector"] = df["Ticker"].map(SECTOR_MAP).fillna("Others")
    sector_codes = {s: i for i, s in enumerate(sorted(df["Sector"].unique()))}
    df["SectorId"] = df["Sector"].map(sector_codes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset → {OUTPUT_CSV.resolve()} ({OUTPUT_CSV.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main() 