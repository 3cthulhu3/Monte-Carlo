"""Discrete-time delta hedging of a short European option, with proportional costs.

Set-up (seller's view): at t=0 we receive the Black-Scholes premium, buy ``delta``
shares and keep the rest in a cash account earning ``r``. At each of the
``n_steps - 1`` intermediate dates we rebalance to the new Black-Scholes delta,
paying ``transaction_cost_bps`` on the traded notional. At maturity

    hedging_error = (shares * S_T + cash) - payoff

With continuous, cost-free hedging the error would be zero; discrete hedging
leaves an error whose RMSE shrinks roughly like ``1 / sqrt(n_steps)`` while
transaction costs grow with the rebalancing frequency.

All paths are processed at once (vectorised over paths), and path ``i`` is still
generated from ``numpy.random.default_rng(base_seed + i)``, so results are
identical to the original path-by-path implementation.
"""
from __future__ import annotations

import math

import numpy as np

from .analytic import black_scholes_delta_array, black_scholes_delta_tau, black_scholes_price
from .simulation import paths_from_normals, simulate_single_gbm_path
from .validation import (
    _check_finite,
    _check_positive_int,
    validate_inputs,
    validate_option_type,
    validate_strike,
)


def _check_cost(transaction_cost_bps) -> float:
    _check_finite("transaction_cost_bps", transaction_cost_bps)
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")
    return transaction_cost_bps / 10000.0


def _delta_vector(S, K, tau, r, sigma, option_type):
    if tau <= 0:
        return np.array([black_scholes_delta_tau(s, K, 0.0, r, sigma, option_type) for s in S])
    return black_scholes_delta_array(S, K, tau, r, sigma, option_type)


def hedge_paths(
    paths,
    K,
    T,
    r,
    sigma,
    transaction_cost_bps=0.0,
    option_type: str = "call",
    charge_closing_cost: bool = False,
) -> dict:
    """Delta-hedge a short option along each row of ``paths`` (shape ``(n_paths, n_steps + 1)``).

    ``charge_closing_cost=True`` also charges the cost of liquidating the final
    stock position at maturity (cash-settled view). The default ``False`` keeps the
    original behaviour (position delivered/settled without a final trade).

    Returns a dict of per-path arrays: ``terminal_stock``, ``option_payoff``,
    ``terminal_portfolio``, ``hedging_error``, ``total_transaction_cost``.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    S0 = float(paths[0, 0])
    validate_inputs(S0, T, r, sigma, n_paths, n_steps)
    validate_strike(K)
    option_type = validate_option_type(option_type)
    if not np.allclose(paths[:, 0], S0):
        raise ValueError("all paths must start from the same S0.")
    k = _check_cost(transaction_cost_bps)

    dt = T / n_steps
    growth = math.exp(r * dt)

    option_price_0 = black_scholes_price(S0, K, T, r, sigma, option_type)
    delta_0 = black_scholes_delta_tau(S0, K, T, r, sigma, option_type)

    initial_cost = abs(delta_0) * S0 * k
    stock = np.full(n_paths, delta_0)
    cash = np.full(n_paths, option_price_0 - delta_0 * S0 - initial_cost)
    total_cost = np.full(n_paths, initial_cost)

    for i in range(1, n_steps + 1):
        S = paths[:, i]
        cash *= growth
        if i < n_steps:
            tau = max(T - i * dt, 0.0)
            new_delta = _delta_vector(S, K, tau, r, sigma, option_type)
            change = new_delta - stock
            trade_cost = np.abs(change) * S * k
            cash -= change * S
            cash -= trade_cost
            total_cost += trade_cost
            stock = new_delta

    terminal_stock = paths[:, -1]
    if charge_closing_cost:
        closing_cost = np.abs(stock) * terminal_stock * k
        cash -= closing_cost
        total_cost += closing_cost

    if option_type == "call":
        payoff = np.maximum(terminal_stock - K, 0.0)
    else:
        payoff = np.maximum(K - terminal_stock, 0.0)
    portfolio = stock * terminal_stock + cash

    return {
        "terminal_stock": terminal_stock,
        "option_payoff": payoff,
        "terminal_portfolio": portfolio,
        "hedging_error": portfolio - payoff,
        "total_transaction_cost": total_cost,
    }


def simulate_delta_hedge_one_path_with_costs(
    S0, K, T, r, sigma, n_steps, transaction_cost_bps=0.0, seed=None,
    option_type: str = "call", charge_closing_cost: bool = False,
) -> dict:
    """Hedge along a single simulated path; returns scalar results."""
    path = simulate_single_gbm_path(S0, T, r, sigma, n_steps, seed=seed)
    result = hedge_paths(path[None, :], K, T, r, sigma, transaction_cost_bps, option_type, charge_closing_cost)
    return {key: float(value[0]) for key, value in result.items()}


def _summary(errors: np.ndarray, costs: np.ndarray) -> dict:
    ddof = 1 if errors.size > 1 else 0
    return {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std(ddof=ddof)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=ddof)),
        "min_error": float(errors.min()),
        "max_error": float(errors.max()),
    }


def simulate_delta_hedge_many_paths_with_costs(
    S0, K, T, r, sigma, n_steps, n_paths, transaction_cost_bps=0.0, base_seed=42,
    option_type: str = "call", charge_closing_cost: bool = False,
) -> dict:
    """Hedge ``n_paths`` independent paths and summarise the hedging error distribution.

    ``std_error`` is the *standard deviation* of the hedging error across paths
    (name kept for backward compatibility), not a Monte Carlo standard error.
    """
    validate_inputs(S0, T, r, sigma, n_paths, n_steps)
    _check_positive_int("base_seed", base_seed, minimum=0)
    Z = np.stack([np.random.default_rng(base_seed + i).standard_normal(n_steps) for i in range(n_paths)])
    paths = paths_from_normals(S0, T, r, sigma, Z)
    result = hedge_paths(paths, K, T, r, sigma, transaction_cost_bps, option_type, charge_closing_cost)
    errors, costs = result["hedging_error"], result["total_transaction_cost"]
    return {"hedging_errors": errors, "transaction_costs": costs, **_summary(errors, costs)}


def run_hedging_frequency_study_with_costs(
    S0, K, T, r, sigma, hedge_steps_list, n_paths, transaction_cost_bps=0.0, base_seed=42,
    option_type: str = "call", charge_closing_cost: bool = False,
) -> list[dict]:
    """Run the many-path hedge for each rebalancing frequency in ``hedge_steps_list``."""
    if len(hedge_steps_list) == 0:
        raise ValueError("hedge_steps_list must not be empty.")
    rows = []
    for n_steps in hedge_steps_list:
        study = simulate_delta_hedge_many_paths_with_costs(
            S0, K, T, r, sigma, n_steps, n_paths, transaction_cost_bps, base_seed,
            option_type, charge_closing_cost,
        )
        rows.append({
            "n_steps": n_steps,
            **{key: study[key] for key in (
                "mean_error", "std_error", "rmse", "mean_abs_error", "mean_cost", "min_error", "max_error"
            )},
        })
    return rows
