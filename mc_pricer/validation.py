"""Input validation shared by every public function.

All checks raise ``ValueError`` (or ``TypeError`` for wrong integer types) with
a message that names the offending parameter, so mistakes surface at the call
site instead of as ``ZeroDivisionError``/``NaN`` deep inside a formula.
"""
from __future__ import annotations

import math
import numbers

OPTION_TYPES = ("call", "put")


def _check_finite(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


def _check_positive_int(name: str, value, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def validate_market(S0, T, r, sigma) -> None:
    """Validate spot, maturity, rate and volatility."""
    for name, value in (("S0", S0), ("T", T), ("r", r), ("sigma", sigma)):
        _check_finite(name, value)
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")


def validate_strike(K) -> None:
    _check_finite("K", K)
    if K <= 0:
        raise ValueError("K must be positive.")


def validate_option_type(option_type: str) -> str:
    if not isinstance(option_type, str) or option_type.lower() not in OPTION_TYPES:
        raise ValueError(f"option_type must be one of {OPTION_TYPES}, got {option_type!r}.")
    return option_type.lower()


def validate_inputs(S0, T, r, sigma, n_sims, n_steps=None) -> None:
    """Validate simulation inputs (kept for backward compatibility).

    ``n_sims`` must be an integer >= 1 and ``n_steps`` (if given) an integer >= 1.
    """
    validate_market(S0, T, r, sigma)
    _check_positive_int("n_sims", n_sims)
    if n_steps is not None:
        _check_positive_int("n_steps", n_steps)


def validate_confidence(confidence) -> None:
    _check_finite("confidence", confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
