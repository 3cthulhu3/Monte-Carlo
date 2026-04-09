import math
import numpy as np


def validate_inputs(S0, T, r, sigma, n_sims, n_steps=None):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if n_steps is not None and n_steps <= 0:
        raise ValueError("n_steps must be positive.")


def simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=None):
    validate_inputs(S0, T, r, sigma, n_sims)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S0 * np.exp(drift + diffusion)
    return ST


def simulate_gbm_paths(S0, T, r, sigma, n_sims, n_steps, seed=None):
    validate_inputs(S0, T, r, sigma, n_sims, n_steps)

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal((n_sims, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.column_stack([np.zeros(n_sims), log_paths])

    paths = S0 * np.exp(log_paths)
    return paths


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S0 * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return call_price


def monte_carlo_call_price(S0, K, T, r, sigma, n_sims, seed=None):
    ST = simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=seed)

    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    price_estimate = discounted_payoffs.mean()
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_sims)

    return price_estimate, std_error


if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    n_sims = 100000

    mc_price, mc_std_error = monte_carlo_call_price(S0, K, T, r, sigma, n_sims, seed=42)
    bs_price = black_scholes_call_price(S0, K, T, r, sigma)

    print("Monte Carlo call price:", mc_price)
    print("Monte Carlo standard error:", mc_std_error)
    print("Black-Scholes call price:", bs_price)
    print("Absolute pricing difference:", abs(mc_price - bs_price))
