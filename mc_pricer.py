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


def simulate_single_gbm_path(S0, T, r, sigma, n_steps, seed=None):
    validate_inputs(S0, T, r, sigma, 1, n_steps)

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal(n_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_path = np.concatenate(([0.0], np.cumsum(increments)))
    path = S0 * np.exp(log_path)
    return path


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


def black_scholes_call_delta_tau(S, K, tau, r, sigma):
    if tau <= 0:
        return 1.0 if S > K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    return normal_cdf(d1)


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


def simulate_delta_hedge_one_path(S0, K, T, r, sigma, n_steps, seed=None):
    path = simulate_single_gbm_path(S0, T, r, sigma, n_steps, seed=seed)
    dt = T / n_steps

    option_price_0 = black_scholes_call_price(S0, K, T, r, sigma)
    delta_0 = black_scholes_call_delta_tau(S0, K, T, r, sigma)

    stock_position = delta_0
    cash_account = option_price_0 - stock_position * S0

    hedge_history = [{
        "step": 0,
        "time": 0.0,
        "stock_price": path[0],
        "delta": stock_position,
        "cash": cash_account,
        "portfolio_value": stock_position * path[0] + cash_account
    }]

    for i in range(1, n_steps + 1):
        t = i * dt
        tau = max(T - t, 0.0)
        S = path[i]

        cash_account *= math.exp(r * dt)

        if i < n_steps:
            new_delta = black_scholes_call_delta_tau(S, K, tau, r, sigma)
            delta_change = new_delta - stock_position
            cash_account -= delta_change * S
            stock_position = new_delta

        portfolio_value = stock_position * S + cash_account

        hedge_history.append({
            "step": i,
            "time": t,
            "stock_price": S,
            "delta": stock_position,
            "cash": cash_account,
            "portfolio_value": portfolio_value
        })

    terminal_stock = path[-1]
    option_payoff = max(terminal_stock - K, 0.0)
    terminal_portfolio = stock_position * terminal_stock + cash_account
    hedging_error = terminal_portfolio - option_payoff

    return {
        "path": path,
        "hedge_history": hedge_history,
        "option_price_0": option_price_0,
        "terminal_stock": terminal_stock,
        "option_payoff": option_payoff,
        "terminal_portfolio": terminal_portfolio,
        "hedging_error": hedging_error
    }


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

    hedge_result = simulate_delta_hedge_one_path(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        n_steps=12,
        seed=42
    )

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

    print("\nOne-Path Delta Hedging Result")
    print("-" * 60)
    print("Initial option price:", hedge_result["option_price_0"])
    print("Terminal stock price:", hedge_result["terminal_stock"])
    print("Option payoff at maturity:", hedge_result["option_payoff"])
    print("Terminal hedge portfolio value:", hedge_result["terminal_portfolio"])
    print("Hedging error (portfolio - payoff):", hedge_result["hedging_error"])
