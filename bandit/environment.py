from __future__ import annotations

import numpy as np
from typing import Sequence


class StockEnvironment:
    """IID Gaussian return generator for *K* stocks.

    Parameters
    ----------
    mus : Sequence[float]
        True mean return for each stock.
    sigmas : Sequence[float]
        True standard deviation of returns for each stock.
    costs : Sequence[float]
        Deterministic per-trade cost for each stock (e.g. transaction fee or risk capital usage).
    seed : int | None, optional
        RNG seed for reproducibility.
    """

    def __init__(self, mus: Sequence[float], sigmas: Sequence[float], costs: Sequence[float], seed: int | None = None):
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        assert self.mus.shape == self.sigmas.shape == self.costs.shape, "All parameter arrays must have the same length."
        self.K = self.mus.size
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def sample_return(self, arm: int) -> float:
        """Draw one return for the requested stock index."""
        return float(self._rng.normal(self.mus[arm], self.sigmas[arm]))

    def cost(self, arm: int) -> float:
        """Deterministic per-trade cost for *arm*."""
        return float(self.costs[arm])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def sample_returns(self, arms: Sequence[int]) -> list[float]:
        """Vectorised wrapper around :py:meth:`sample_return`.

        Samples and returns a list of realisations for each *arm* in the
        provided ``arms`` sequence. This is mainly used by portfolio-style
        algorithms that activate several stocks in a single round.
        """
        return [self.sample_return(a) for a in arms] 