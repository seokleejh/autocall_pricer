"""
Model-agnostic Monte Carlo pricing engine.

The engine connects a model simulator to an AutocallableNote and runs
the standard MC loop:  simulate paths → evaluate payoff → average.

The model interface contract:
    model.simulate(n_paths, observation_times, antithetic) -> np.ndarray (n_paths, n_obs)

Any object satisfying this signature works: LocalVolModel, HestonModel, SABRModel,
or a future worst-of basket simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from products.autocallable import AutocallableNote


@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval: tuple[float, float]
    n_paths: int
    model_name: str
    expected_duration: float = float("nan")   # risk-neutral avg time to termination (years)

    def __str__(self) -> str:
        lo, hi = self.conf_interval
        dur = f"  dur={self.expected_duration:.2f}y" if self.expected_duration == self.expected_duration else ""
        return (
            f"[{self.model_name}]  price = {self.price:.6f}"
            f"  ± {self.std_error:.6f}"
            f"  95% CI = [{lo:.6f}, {hi:.6f}]"
            f"  (N={self.n_paths:,})"
            f"{dur}"
        )


class MCPricer:
    """
    Parameters
    ----------
    model
        Any model with a .simulate(n_paths, observation_times, antithetic) method.
    model_name : str
        Label used in output.
    antithetic : bool
        Enable antithetic variates variance reduction.
    """

    def __init__(self, model, model_name: str = "Model", antithetic: bool = True) -> None:
        self.model = model
        self.model_name = model_name
        self.antithetic = antithetic

    def price(self, product: AutocallableNote, n_paths: int) -> PricingResult:
        """
        Price `product` using `n_paths` Monte Carlo paths.

        Returns
        -------
        PricingResult with price, std error, and 95% CI.
        """
        performances = self.model.simulate(
            n_paths=n_paths,
            observation_times=product.observation_dates,
            antithetic=self.antithetic,
        )
        payoffs = product.evaluate_payoff(performances)
        price = float(np.mean(payoffs))
        se = float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
        ci = (price - 1.96 * se, price + 1.96 * se)
        duration = product.evaluate_duration(performances)
        return PricingResult(
            price=price,
            std_error=se,
            conf_interval=ci,
            n_paths=n_paths,
            model_name=self.model_name,
            expected_duration=duration,
        )

    def price_batch(
        self,
        product: AutocallableNote,
        n_paths_list: list[int],
    ) -> list[PricingResult]:
        """Price at multiple n_paths for convergence analysis."""
        return [self.price(product, n) for n in n_paths_list]


def compare_models(
    pricers: list[MCPricer],
    product: AutocallableNote,
    n_paths: int,
) -> list[PricingResult]:
    """
    Price `product` under each model in `pricers` and return results.
    Results are printed to stdout.
    """
    results = []
    for pricer in pricers:
        result = pricer.price(product, n_paths)
        print(result)
        results.append(result)
    return results
