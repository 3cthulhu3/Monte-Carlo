"""Monte Carlo pricing of European options with error bars and variance reduction."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import NamedTuple

import numpy as np

from .analytic import black_scholes_price
from .simulation import simulate_terminal_prices
from .validation import (
    _check_positive_int,
    validate_confidence,
    validate_market,
    validate_option_type,
    validate_strike,
)

METHODS = ("plain", "antithetic", "control_variate")


class PricingResult(NamedTuple):
    """Monte Carlo estimate with its standard error and confidence interval.

    Being a ``NamedTuple`` it still unpacks as
    ``price, std_error, ci_lower, ci_upper = result`` like the original API.
    """

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_upper - self.ci_lower)

    def contains(self, value: float) -> bool:
        """True if ``value`` lies inside the confidence interval."""
        return self.ci_lower <= value <= self.ci_upper


def _payoff(ST: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def _summarise(samples: np.ndarray, confidence: float) -> PricingResult:
    n = samples.size
    mean = float(samples.mean())
    std_error = float(samples.std(ddof=1) / math.sqrt(n))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return PricingResult(mean, std_error, mean - z * std_error, mean + z * std_error)


def monte_carlo_price(
    S0,
    K,
    T,
    r,
    sigma,
    n_sims,
    option_type: str = "call",
    method: str = "plain",
    seed=None,
    Z=None,
    confidence: float = 0.95,
) -> PricingResult:
    """Price a European option by Monte Carlo.

    Parameters
    ----------
    n_sims:
        Number of payoff evaluations (>= 2). For ``method="antithetic"`` it must be
        even and is split into ``n_sims // 2`` antithetic pairs.
    method:
        ``"plain"`` – crude Monte Carlo;
        ``"antithetic"`` – pairs ``Z`` with ``-Z``;
        ``"control_variate"`` – uses the discounted terminal price (known mean ``S0``)
        as control with the variance-minimising coefficient estimated from the sample.
    Z:
        Optional pre-drawn standard normals, shape ``(n_sims,)``
        (``(n_sims // 2,)`` for antithetic). Overrides ``seed``.
    confidence:
        Two-sided confidence level of the returned interval.
    """
    validate_market(S0, T, r, sigma)
    validate_strike(K)
    option_type = validate_option_type(option_type)
    validate_confidence(confidence)
    _check_positive_int("n_sims", n_sims, minimum=2)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")

    discount = math.exp(-r * T)

    if method == "antithetic":
        if n_sims % 2:
            raise ValueError("n_sims must be even for the antithetic method.")
        n_pairs = n_sims // 2
        if n_pairs < 2:
            raise ValueError("antithetic method needs n_sims >= 4.")
        if Z is None:
            Z = np.random.default_rng(seed).standard_normal(n_pairs)
        ST_plus = simulate_terminal_prices(S0, T, r, sigma, n_pairs, Z=Z)
        ST_minus = simulate_terminal_prices(S0, T, r, sigma, n_pairs, Z=-np.asarray(Z, dtype=float))
        samples = 0.5 * discount * (_payoff(ST_plus, K, option_type) + _payoff(ST_minus, K, option_type))
        return _summarise(samples, confidence)

    ST = simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=seed, Z=Z)
    samples = discount * _payoff(ST, K, option_type)

    if method == "control_variate":
        control = discount * ST  # E[control] = S0 under the risk-neutral measure
        var_control = control.var(ddof=1)
        beta = 0.0 if var_control == 0.0 else np.cov(samples, control, ddof=1)[0, 1] / var_control
        samples = samples - beta * (control - S0)

    return _summarise(samples, confidence)


def monte_carlo_call_price(
    S0, K, T, r, sigma, n_sims, seed=None, Z=None, *, method: str = "plain", confidence: float = 0.95
) -> PricingResult:
    """European call by Monte Carlo (original signature, now returns a ``PricingResult``)."""
    return monte_carlo_price(
        S0, K, T, r, sigma, n_sims, option_type="call", method=method, seed=seed, Z=Z, confidence=confidence
    )


def convergence_study(
    S0,
    K,
    T,
    r,
    sigma,
    n_sims_list,
    option_type: str = "call",
    method: str = "plain",
    seed=42,
    confidence: float = 0.95,
) -> list[dict]:
    """Price with increasing sample sizes and compare against Black-Scholes.

    Each row reports the estimate, its standard error, the absolute error, the
    error measured in standard errors (``z_score``) and whether the confidence
    interval covers the analytical price. Standard errors should shrink like
    ``1 / sqrt(n_sims)`` – see :func:`estimate_convergence_rate`.
    """
    benchmark = black_scholes_price(S0, K, T, r, sigma, option_type)
    rows = []
    for i, n_sims in enumerate(n_sims_list):
        result = monte_carlo_price(
            S0, K, T, r, sigma, n_sims, option_type=option_type, method=method,
            seed=None if seed is None else seed + i, confidence=confidence,
        )
        error = result.price - benchmark
        rows.append({
            "n_sims": n_sims,
            "price": result.price,
            "std_error": result.std_error,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "benchmark": benchmark,
            "abs_error": abs(error),
            "z_score": error / result.std_error if result.std_error > 0 else 0.0,
            "ci_covers_benchmark": result.contains(benchmark),
        })
    return rows


def estimate_convergence_rate(rows: list[dict]) -> float:
    """Least-squares slope of ``log(std_error)`` vs ``log(n_sims)`` (theory: -0.5)."""
    n = np.array([row["n_sims"] for row in rows], dtype=float)
    se = np.array([row["std_error"] for row in rows], dtype=float)
    if n.size < 2 or np.any(se <= 0):
        raise ValueError("need at least two rows with positive standard errors.")
    slope, _ = np.polyfit(np.log(n), np.log(se), 1)
    return float(slope)


def variance_reduction_study(
    S0, K, T, r, sigma, n_sims, option_type: str = "call", seed=42
) -> list[dict]:
    """Compare the standard error of every method at the same number of payoff evaluations.

    ``variance_ratio`` is ``(SE_plain / SE_method)^2``: how many times more plain
    samples would be needed to reach the same accuracy.
    """
    benchmark = black_scholes_price(S0, K, T, r, sigma, option_type)
    n_sims = n_sims - (n_sims % 2)  # keep it even so antithetic is comparable
    results = {
        method: monte_carlo_price(S0, K, T, r, sigma, n_sims, option_type, method, seed=seed)
        for method in METHODS
    }
    plain_se = results["plain"].std_error
    return [
        {
            "method": method,
            "price": res.price,
            "std_error": res.std_error,
            "abs_error": abs(res.price - benchmark),
            "variance_ratio": (plain_se / res.std_error) ** 2 if res.std_error > 0 else math.inf,
        }
        for method, res in results.items()
    ]
