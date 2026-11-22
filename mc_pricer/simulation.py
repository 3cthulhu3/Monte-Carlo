"""Risk-neutral geometric Brownian motion (GBM) samplers.

Under the risk-neutral measure ``dS = r S dt + sigma S dW`` has the exact solution
``S_t = S0 * exp((r - sigma^2/2) t + sigma W_t)``, so paths are simulated exactly
on the time grid (no Euler discretisation error).
"""
from __future__ import annotations

import numpy as np

from .validation import validate_inputs


def _as_normals(Z, expected_shape: tuple[int, ...]) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if Z.shape != expected_shape:
        raise ValueError(f"Z must have shape {expected_shape}, got {Z.shape}.")
    return Z


def simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=None, Z=None) -> np.ndarray:
    """Sample ``n_sims`` terminal prices ``S_T``.

    Pass ``Z`` (shape ``(n_sims,)``) to reuse the same normals across calls,
    e.g. for common-random-number Greeks; otherwise they are drawn from ``seed``.
    """
    validate_inputs(S0, T, r, sigma, n_sims)
    if Z is None:
        Z = np.random.default_rng(seed).standard_normal(n_sims)
    else:
        Z = _as_normals(Z, (n_sims,))
    drift = (r - 0.5 * sigma**2) * T
    return S0 * np.exp(drift + sigma * np.sqrt(T) * Z)


def paths_from_normals(S0, T, r, sigma, Z) -> np.ndarray:
    """Turn a ``(n_paths, n_steps)`` array of standard normals into GBM paths.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first column is S0.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n_paths, n_steps = Z.shape
    validate_inputs(S0, T, r, sigma, n_paths, n_steps)
    dt = T / n_steps
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate((np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)), axis=1)
    return S0 * np.exp(log_paths)


def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, seed=None) -> np.ndarray:
    """Simulate ``n_paths`` GBM paths on ``n_steps`` equal steps, shape ``(n_paths, n_steps + 1)``."""
    validate_inputs(S0, T, r, sigma, n_paths, n_steps)
    Z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    return paths_from_normals(S0, T, r, sigma, Z)


def simulate_single_gbm_path(S0, T, r, sigma, n_steps, seed=None) -> np.ndarray:
    """Simulate one GBM path, shape ``(n_steps + 1,)`` (kept for backward compatibility)."""
    validate_inputs(S0, T, r, sigma, 1, n_steps)
    Z = np.random.default_rng(seed).standard_normal(n_steps)
    return paths_from_normals(S0, T, r, sigma, Z[None, :])[0]
