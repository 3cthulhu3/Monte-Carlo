"""Monte Carlo estimators of delta and gamma.

Two approaches are provided:

* :func:`monte_carlo_delta_gamma` – central finite differences with common random
  numbers (CRN). Simple, works for any payoff, but biased by ``O(h^2)`` and gives
  no error bar.
* :func:`monte_carlo_greeks_pathwise` – unbiased pathwise delta and a mixed
  pathwise/likelihood-ratio gamma, each with a standard error.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .pricing import monte_carlo_price
from .validation import (
    _check_finite,
    _check_positive_int,
    validate_market,
    validate_option_type,
    validate_strike,
)


class GreekEstimates(NamedTuple):
    delta: float
    delta_se: float
    gamma: float
    gamma_se: float


def monte_carlo_delta_gamma(S0, K, T, r, sigma, n_sims, h=1.0, seed=42, option_type: str = "call"):
    """Finite-difference delta and gamma with common random numbers.

    The same normals are reused for ``S0 - h``, ``S0`` and ``S0 + h`` so most of the
    Monte Carlo noise cancels in the differences. Returns ``(delta, gamma)``.
    """
    validate_market(S0, T, r, sigma)
    _check_finite("h", h)
    if not 0 < h < S0:
        raise ValueError(f"h must satisfy 0 < h < S0 (got h={h}, S0={S0}).")
    _check_positive_int("n_sims", n_sims, minimum=2)
    Z = np.random.default_rng(seed).standard_normal(n_sims)

    def price(spot):
        return monte_carlo_price(spot, K, T, r, sigma, n_sims, option_type=option_type, Z=Z).price

    up, mid, down = price(S0 + h), price(S0), price(S0 - h)
    delta = (up - down) / (2.0 * h)
    gamma = (up - 2.0 * mid + down) / h**2
    return float(delta), float(gamma)


def monte_carlo_greeks_pathwise(
    S0, K, T, r, sigma, n_sims, option_type: str = "call", seed=42
) -> GreekEstimates:
    """Pathwise delta and mixed pathwise/likelihood-ratio gamma with standard errors.

    With ``S_T = S0 exp((r - sigma^2/2)T + sigma sqrt(T) Z)``:

    * call delta  = E[e^{-rT} 1{S_T > K} S_T / S0]
    * call gamma  = E[e^{-rT} 1{S_T > K} (S_T / S0^2) (Z / (sigma sqrt T) - 1)]

    Put formulas use ``-1{S_T < K}`` instead of ``1{S_T > K}``. Requires ``sigma > 0``.
    """
    validate_market(S0, T, r, sigma)
    validate_strike(K)
    option_type = validate_option_type(option_type)
    _check_positive_int("n_sims", n_sims, minimum=2)
    if sigma == 0:
        raise ValueError("pathwise/likelihood-ratio Greeks require sigma > 0.")

    Z = np.random.default_rng(seed).standard_normal(n_sims)
    sqrt_t = math.sqrt(T)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_t * Z)
    discount = math.exp(-r * T)
    slope = (ST > K).astype(float) if option_type == "call" else -(ST < K).astype(float)

    delta_samples = discount * slope * ST / S0
    gamma_samples = discount * slope * (ST / S0**2) * (Z / (sigma * sqrt_t) - 1.0)

    root_n = math.sqrt(n_sims)
    return GreekEstimates(
        delta=float(delta_samples.mean()),
        delta_se=float(delta_samples.std(ddof=1) / root_n),
        gamma=float(gamma_samples.mean()),
        gamma_se=float(gamma_samples.std(ddof=1) / root_n),
    )
