"""Closed-form Black-Scholes benchmark (no dividends).

These functions are the ground truth the Monte Carlo estimators are checked
against. They accept ``sigma == 0`` and return the deterministic limit
(discounted intrinsic value against the forward) instead of dividing by zero.
"""
from __future__ import annotations

import math

import numpy as np

from .validation import validate_market, validate_option_type, validate_strike

try:  # SciPy is optional: it only makes the vectorised CDF faster.
    from scipy.special import ndtr as _ndtr
except ImportError:  # pragma: no cover - exercised only without SciPy
    _ndtr = None

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    """Standard normal CDF for a scalar."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def normal_pdf(x: float) -> float:
    """Standard normal PDF for a scalar."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


_erf_vec = np.vectorize(math.erf, otypes=[float])


def normal_cdf_array(x) -> np.ndarray:
    """Vectorised standard normal CDF (uses SciPy when installed)."""
    x = np.asarray(x, dtype=float)
    if _ndtr is not None:
        return _ndtr(x)
    return 0.5 * (1.0 + _erf_vec(x / _SQRT2))


def _d1_d2(S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(tau)
    numerator = math.log(S / K) + (r + 0.5 * sigma**2) * tau
    if vol_sqrt_t == 0.0:
        # Deterministic limit: d1 = d2 = +/-inf depending on moneyness vs the forward.
        d = math.copysign(math.inf, numerator) if numerator != 0.0 else 0.0
        return d, d
    d1 = numerator / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _check(S0, K, T, r, sigma, option_type):
    validate_market(S0, T, r, sigma)
    validate_strike(K)
    return validate_option_type(option_type)


def black_scholes_price(S0, K, T, r, sigma, option_type: str = "call") -> float:
    """Black-Scholes price of a European call or put."""
    option_type = _check(S0, K, T, r, sigma, option_type)
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    discount = math.exp(-r * T)
    if option_type == "call":
        return S0 * normal_cdf(d1) - K * discount * normal_cdf(d2)
    return K * discount * normal_cdf(-d2) - S0 * normal_cdf(-d1)


def black_scholes_delta(S0, K, T, r, sigma, option_type: str = "call") -> float:
    """Black-Scholes delta dV/dS."""
    option_type = _check(S0, K, T, r, sigma, option_type)
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    call_delta = normal_cdf(d1)
    return call_delta if option_type == "call" else call_delta - 1.0


def black_scholes_gamma(S0, K, T, r, sigma, option_type: str = "call") -> float:
    """Black-Scholes gamma d2V/dS2 (identical for calls and puts)."""
    _check(S0, K, T, r, sigma, option_type)
    if sigma == 0.0:
        return 0.0
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return normal_pdf(d1) / (S0 * sigma * math.sqrt(T))


def black_scholes_delta_tau(S, K, tau, r, sigma, option_type: str = "call") -> float:
    """Delta with time-to-maturity ``tau``; at ``tau <= 0`` returns the payoff slope."""
    option_type = validate_option_type(option_type)
    if tau <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    call_delta = normal_cdf(d1)
    return call_delta if option_type == "call" else call_delta - 1.0


def black_scholes_delta_array(S, K, tau, r, sigma, option_type: str = "call") -> np.ndarray:
    """Vectorised delta over an array of spots for a single ``tau > 0``."""
    S = np.asarray(S, dtype=float)
    vol_sqrt_t = sigma * math.sqrt(tau)
    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * tau
    if vol_sqrt_t == 0.0:
        call_delta = np.where(numerator > 0, 1.0, np.where(numerator < 0, 0.0, 0.5))
    else:
        call_delta = normal_cdf_array(numerator / vol_sqrt_t)
    return call_delta if option_type == "call" else call_delta - 1.0


# --- Backward-compatible call-only names -----------------------------------

def black_scholes_call_price(S0, K, T, r, sigma) -> float:
    return black_scholes_price(S0, K, T, r, sigma, "call")


def black_scholes_call_delta(S0, K, T, r, sigma) -> float:
    return black_scholes_delta(S0, K, T, r, sigma, "call")


def black_scholes_call_gamma(S0, K, T, r, sigma) -> float:
    return black_scholes_gamma(S0, K, T, r, sigma, "call")


def black_scholes_call_delta_tau(S, K, tau, r, sigma) -> float:
    return black_scholes_delta_tau(S, K, tau, r, sigma, "call")
