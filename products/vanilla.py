"""
European call/put — compatible with MCPricer via duck typing.

The model simulates performances S(t)/S(0); the payoff converts back
to absolute price units so the result matches the Black-Scholes formula.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VanillaOption:
    """
    Parameters
    ----------
    spot : float
        Initial spot price S(0).
    strike : float
        Absolute strike level K.
    maturity : float
        Option expiry in years.
    rate : float
        Continuously compounded discount rate.
    is_call : bool
        True for call, False for put.
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    is_call: bool = True

    @property
    def observation_dates(self) -> np.ndarray:
        return np.array([self.maturity])

    def evaluate_payoff(self, performances: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        performances : np.ndarray, shape (n_paths, 1)
            S(T)/S(0) for each path.

        Returns
        -------
        pv : np.ndarray, shape (n_paths,)
            Discounted payoff in same currency units as spot.
        """
        perf = performances[:, 0]
        k = self.strike / self.spot
        df = np.exp(-self.rate * self.maturity)
        if self.is_call:
            return self.spot * df * np.maximum(perf - k, 0.0)
        else:
            return self.spot * df * np.maximum(k - perf, 0.0)
