"""Environment that feeds historical returns from a pre-built CSV.

The CSV must contain columns: Date, Ticker, AdjClose.  Optionally sector info
columns are ignored.
Returns are computed as log returns between consecutive trading days for each
ticker, then aligned into a (T, K) array ordered by ``tickers`` parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

__all__ = ["CsvReturnEnvironment"]


class CsvReturnEnvironment:
    """Finite-horizon environment that yields pre-computed returns per arm."""

    def __init__(self, csv_path: str | Path, tickers: List[str]):
        self.tickers = list(tickers)
        self.K = len(tickers)

        df = pd.read_csv(csv_path)
        # Filter desired tickers and pivot to Date index
        df = df[df["Ticker"].isin(self.tickers)].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        prices = df.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
        prices = prices[self.tickers]  # ensure column order
        log_prices = np.log(prices)
        log_returns = log_prices.diff().dropna()
        self.dates = log_returns.index.to_numpy()  # preserve dates for logging
        self.returns = log_returns.values.astype(float)  # shape (T, K)
        self.T = self.returns.shape[0]
        self._current = 0

    # ------------------------------------------------------------------
    def has_next(self) -> bool:
        return self._current < self.T

    def next_return(self, arm: int) -> float:
        assert self.has_next(), "Environment exhausted"
        r = float(self.returns[self._current, arm])
        return r

    def advance(self):
        """Move to next day."""
        self._current += 1

    def reset(self):
        """Reset pointer to first day (for cyclic access)."""
        self._current = 0

    # ------------------------------------------------------------------
    # Convenience: full-day returns access
    # ------------------------------------------------------------------
    def current_returns_dict(self) -> dict[int, float]:
        """Return a dictionary {arm_index: return} for the current day."""
        if not self.has_next():
            raise RuntimeError("Environment exhausted")
        vec = self.returns[self._current]
        return {i: float(vec[i]) for i in range(self.K)}

    # ------------------------------------------------------------------
    # Date helper for logging
    # ------------------------------------------------------------------
    def current_date(self):
        """Return the pandas Timestamp of the current step (before advance)."""
        return self.dates[self._current] if hasattr(self, "dates") else None 