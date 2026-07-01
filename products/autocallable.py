"""
Flexible autocallable note definition.

The product is specified entirely as data — no model assumptions here.
The MC engine evaluates the payoff on simulated paths.

Supported structures (controlled by fields below):
  - Standard autocall:   conditional_coupon=False, step_down=None
  - Phoenix autocall:    conditional_coupon=True  (coupon paid if underlying > coupon_barrier)
  - Snowball/Step-down:  pass step_down list to autocall_barriers

Worst-of basket support: implemented upstream in models/basket.py
(BasketLocalVolModel / BasketHestonModel), which simulate correlated
multi-asset paths and return the already-aggregated (min-across-assets)
performance array -- evaluate_payoff() below never needs to know whether
performances came from one asset or a worst-of basket. In basket mode,
`spot` on this class is an unused placeholder (conventionally 1.0): all
barriers here are performance fractions, and each asset's own spot lives
on its own ImpliedVolSurface inside the basket model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass
class AutocallableNote:
    """
    Parameters
    ----------
    notional : float
        Face value of the note.
    spot : float | list[float]
        Initial spot price(s). List for multi-asset (future).
    maturity : float
        Note maturity in years.
    observation_dates : sequence of float
        Autocall observation dates in years (must be <= maturity).
    autocall_barriers : float | sequence of float
        Autocall trigger level(s) as a fraction of initial spot.
        Single float → same barrier on all dates.
        Sequence → one per observation date (enables step-down / snowball).
    coupon_rate : float
        Coupon paid at each autocall date (as fraction of notional), e.g. 0.05 = 5%.
    conditional_coupon : bool
        If False (standard autocall): coupon only paid on autocall.
        If True (Phoenix): coupon paid at each date where performance > coupon_barrier,
        even if the autocall barrier is not reached.
    coupon_barrier : float
        Performance threshold for conditional coupon (only used when conditional_coupon=True).
        Expressed as fraction of initial spot.
    capital_barrier : float
        Down-and-in put barrier for capital protection at maturity.
        If final performance > capital_barrier → full notional returned.
        If final performance <= capital_barrier → notional * final_performance returned.
    capital_barrier_active : bool
        Set False for fully capital-protected notes (always return notional at maturity).
    discount_rate : float
        Flat discount rate for present value calculation.
        For full term-structure support, pass a discount curve to the MC engine instead.
    """

    notional: float = 1.0
    spot: float | list[float] = 100.0
    maturity: float = 3.0
    observation_dates: Sequence[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    autocall_barriers: float | Sequence[float] = 1.0
    coupon_rate: float | Sequence[float] = 0.05
    conditional_coupon: bool = False
    coupon_barrier: float = 0.80
    capital_barrier: float = 0.60
    capital_barrier_active: bool = True
    discount_rate: float = 0.03

    def __post_init__(self) -> None:
        self.observation_dates = np.array(self.observation_dates, dtype=float)
        n = len(self.observation_dates)

        if np.isscalar(self.autocall_barriers):
            self._autocall_barriers = np.full(n, float(self.autocall_barriers))
        else:
            barriers = np.array(self.autocall_barriers, dtype=float)
            assert len(barriers) == n, "autocall_barriers length must match observation_dates"
            self._autocall_barriers = barriers

        if np.isscalar(self.coupon_rate):
            self._coupon_schedule = np.full(n, float(self.coupon_rate))
        else:
            coupons = np.array(self.coupon_rate, dtype=float)
            assert len(coupons) == n, "coupon_rate length must match observation_dates"
            self._coupon_schedule = coupons

        # Ensure maturity observation is included
        if not np.isclose(self.observation_dates[-1], self.maturity):
            raise ValueError(
                f"Last observation date ({self.observation_dates[-1]}) must equal maturity ({self.maturity})."
            )

    @property
    def autocall_barrier_schedule(self) -> np.ndarray:
        return self._autocall_barriers

    @property
    def n_assets(self) -> int:
        if isinstance(self.spot, (list, np.ndarray)):
            return len(self.spot)
        return 1

    def evaluate_payoff(
        self,
        performances: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the discounted payoff for a batch of simulated paths.

        Parameters
        ----------
        performances : np.ndarray, shape (n_paths, n_obs)
            Performance of the underlying at each observation date,
            defined as S(t_i) / S(0).
            For worst-of basket (future): pass min performance across assets.

        Returns
        -------
        pv : np.ndarray, shape (n_paths,)
            Present value of the note per unit notional, for each path.
        """
        n_paths, n_obs = performances.shape
        assert n_obs == len(self.observation_dates), "performance cols must match observation dates"

        pv = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)  # paths not yet autocalled

        for i, (t, barrier) in enumerate(
            zip(self.observation_dates, self._autocall_barriers)
        ):
            df = np.exp(-self.discount_rate * t)
            perf = performances[:, i]
            is_last = i == n_obs - 1

            coupon_rate_i = self._coupon_schedule[i]

            # --- Conditional coupon (Phoenix) ---
            if self.conditional_coupon and not is_last:
                coupon_paid = alive & (perf >= self.coupon_barrier)
                pv[coupon_paid] += coupon_rate_i * self.notional * df

            # --- Autocall trigger ---
            autocalled = alive & (perf >= barrier)

            if autocalled.any():
                pv[autocalled] += (self.notional + coupon_rate_i * self.notional) * df
                alive[autocalled] = False

            # --- Maturity (last observation) ---
            if is_last and alive.any():
                final_perf = perf[alive]
                if self.capital_barrier_active:
                    protected = final_perf >= self.capital_barrier
                    # Full repayment + final coupon (Phoenix pays coupon regardless)
                    redeem = np.where(protected, self.notional, self.notional * final_perf)
                    if self.conditional_coupon:
                        coupon_at_mat = np.where(final_perf >= self.coupon_barrier, coupon_rate_i * self.notional, 0.0)
                        redeem += coupon_at_mat
                else:
                    redeem = np.full(alive.sum(), self.notional)
                    if self.conditional_coupon:
                        coupon_at_mat = np.where(final_perf >= self.coupon_barrier, coupon_rate_i * self.notional, 0.0)
                        redeem += coupon_at_mat

                pv[alive] += redeem * df

        return pv

    def evaluate_duration(self, performances: np.ndarray) -> float:
        """
        Average time to termination (autocall or maturity) across MC paths.

        Paths that autocall at observation t_i contribute t_i years.
        Paths that survive all observations contribute maturity years.
        The average is the risk-neutral expected duration of the note.

        Parameters
        ----------
        performances : np.ndarray, shape (n_paths, n_obs)
            Same array passed to evaluate_payoff — reuse without re-simulating.

        Returns
        -------
        float : mean time to termination in years.
        """
        n_paths, n_obs = performances.shape
        termination_times = np.full(n_paths, self.observation_dates[-1], dtype=float)
        alive = np.ones(n_paths, dtype=bool)

        for i, (t, barrier) in enumerate(
            zip(self.observation_dates, self._autocall_barriers)
        ):
            autocalled = alive & (performances[:, i] >= barrier)
            if autocalled.any():
                termination_times[autocalled] = t
                alive[autocalled] = False

        return float(np.mean(termination_times))
