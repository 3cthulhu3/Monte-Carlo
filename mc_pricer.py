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


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=None, Z=None):
    validate_inputs(S0, T, r, sigma, n_sims)

    if Z is None:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_sims)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S0 * np.exp(drift + diffusion)
    return ST


def black_scholes_call_price(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S0 * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def black_scholes_call_delta(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return normal_cdf(d1)


def black_scholes_call_gamma(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return normal_pdf(d1) / (S0 * sigma * math.sqrt(T))


def monte_carlo_call_price(S0, K, T, r, sigma, n_sims, seed=None, Z=None):
    ST = simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=seed, Z=Z)

    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    price_estimate = discounted_payoffs.mean()
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_sims)

    ci_lower = price_estimate - 1.96 * std_error
    ci_upper = price_estimate + 1.96 * std_error

    return price_estimate, std_error, ci_lower, ci_upper


def monte_carlo_delta_gamma(S0, K, T, r, sigma, n_sims, h=1.0, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)

    price_up, _, _, _ = monte_carlo_call_price(S0 + h, K, T, r, sigma, n_sims, Z=Z)
    price_mid, _, _, _ = monte_carlo_call_price(S0, K, T, r, sigma, n_sims, Z=Z)
    price_down, _, _, _ = monte_carlo_call_price(S0 - h, K, T, r, sigma, n_sims, Z=Z)

    delta_mc = (price_up - price_down) / (2.0 * h)
    gamma_mc = (price_up - 2.0 * price_mid + price_down) / (h ** 2)

    return delta_mc, gamma_mc


if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    n_sims = 100000
    h = 1.0

    mc_price, mc_std_error, ci_lower, ci_upper = monte_carlo_call_price(
        S0, K, T, r, sigma, n_sims, seed=42
    )
    bs_price = black_scholes_call_price(S0, K, T, r, sigma)

    bs_delta = black_scholes_call_delta(S0, K, T, r, sigma)
    bs_gamma = black_scholes_call_gamma(S0, K, T, r, sigma)

    mc_delta, mc_gamma = monte_carlo_delta_gamma(S0, K, T, r, sigma, n_sims, h=h, seed=42)

    print("Black-Scholes call price:", bs_price)
    print("Monte Carlo call price:", mc_price)
    print("Monte Carlo standard error:", mc_std_error)
    print("95% confidence interval:", (ci_lower, ci_upper))
    print("Absolute pricing difference:", abs(mc_price - bs_price))

    print("\nGreeks Comparison")
    print("-" * 60)
    print(f"{'Metric':<20}{'Black-Scholes':>18}{'Monte Carlo':>18}")
    print("-" * 60)
    print(f"{'Delta':<20}{bs_delta:>18.6f}{mc_delta:>18.6f}")
    print(f"{'Gamma':<20}{bs_gamma:>18.6f}{mc_gamma:>18.6f}")
