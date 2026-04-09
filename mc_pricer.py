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

    ci_lower = price_estimate - 1.96 * std_error
    ci_upper = price_estimate + 1.96 * std_error

    return price_estimate, std_error, ci_lower, ci_upper


def run_convergence_analysis(S0, K, T, r, sigma, sim_counts, base_seed=42):
    results = []

    for n_sims in sim_counts:
        mc_price, std_error, ci_lower, ci_upper = monte_carlo_call_price(
            S0, K, T, r, sigma, n_sims, seed=base_seed
        )

        results.append({
            "n_sims": n_sims,
            "mc_price": mc_price,
            "std_error": std_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

    return results


def print_convergence_table(results, bs_price):
    print("\nConvergence Analysis")
    print("-" * 95)
    print(
        f"{'Sims':>10} | {'MC Price':>12} | {'Std Error':>12} | "
        f"{'95% CI Lower':>12} | {'95% CI Upper':>12} | {'|MC-BS|':>12}"
    )
    print("-" * 95)

    for row in results:
        diff = abs(row["mc_price"] - bs_price)
        print(
            f"{row['n_sims']:10d} | "
            f"{row['mc_price']:12.6f} | "
            f"{row['std_error']:12.6f} | "
            f"{row['ci_lower']:12.6f} | "
            f"{row['ci_upper']:12.6f} | "
            f"{diff:12.6f}"
        )


if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20

    bs_price = black_scholes_call_price(S0, K, T, r, sigma)

    print("Black-Scholes call price:", bs_price)

    sim_counts = [1000, 5000, 10000, 50000, 100000]
    results = run_convergence_analysis(S0, K, T, r, sigma, sim_counts, base_seed=42)

    print_convergence_table(results, bs_price)
