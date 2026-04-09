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


def simulate_delta_hedge_one_path_with_costs(
    S0, K, T, r, sigma, n_steps, transaction_cost_bps=0.0, seed=None
):
    path = simulate_single_gbm_path(S0, T, r, sigma, n_steps, seed=seed)
    dt = T / n_steps
    k = transaction_cost_bps / 10000.0

    option_price_0 = black_scholes_call_price(S0, K, T, r, sigma)
    delta_0 = black_scholes_call_delta_tau(S0, K, T, r, sigma)

    initial_trade_cost = abs(delta_0) * S0 * k
    stock_position = delta_0
    cash_account = option_price_0 - stock_position * S0 - initial_trade_cost
    total_transaction_cost = initial_trade_cost

    for i in range(1, n_steps + 1):
        t = i * dt
        tau = max(T - t, 0.0)
        S = path[i]

        cash_account *= math.exp(r * dt)

        if i < n_steps:
            new_delta = black_scholes_call_delta_tau(S, K, tau, r, sigma)
            delta_change = new_delta - stock_position
            trade_cost = abs(delta_change) * S * k

            cash_account -= delta_change * S
            cash_account -= trade_cost

            total_transaction_cost += trade_cost
            stock_position = new_delta

    terminal_stock = path[-1]
    option_payoff = max(terminal_stock - K, 0.0)
    terminal_portfolio = stock_position * terminal_stock + cash_account
    hedging_error = terminal_portfolio - option_payoff

    return {
        "terminal_stock": terminal_stock,
        "option_payoff": option_payoff,
        "terminal_portfolio": terminal_portfolio,
        "hedging_error": hedging_error,
        "total_transaction_cost": total_transaction_cost
    }


def simulate_delta_hedge_many_paths_with_costs(
    S0, K, T, r, sigma, n_steps, n_paths, transaction_cost_bps=0.0, base_seed=42
):
    errors = []
    costs = []

    for i in range(n_paths):
        result = simulate_delta_hedge_one_path_with_costs(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_steps=n_steps,
            transaction_cost_bps=transaction_cost_bps,
            seed=base_seed + i
        )

        errors.append(result["hedging_error"])
        costs.append(result["total_transaction_cost"])

    errors = np.array(errors)
    costs = np.array(costs)

    return {
        "hedging_errors": errors,
        "transaction_costs": costs,
        "mean_error": errors.mean(),
        "std_error": errors.std(ddof=1),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_abs_error": np.mean(np.abs(errors)),
        "mean_cost": costs.mean(),
        "std_cost": costs.std(ddof=1),
        "min_error": errors.min(),
        "max_error": errors.max()
    }


def run_hedging_frequency_study_with_costs(
    S0, K, T, r, sigma, hedge_steps_list, n_paths, transaction_cost_bps=0.0, base_seed=42
):
    results = []

    for n_steps in hedge_steps_list:
        study = simulate_delta_hedge_many_paths_with_costs(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_steps=n_steps,
            n_paths=n_paths,
            transaction_cost_bps=transaction_cost_bps,
            base_seed=base_seed
        )

        results.append({
            "n_steps": n_steps,
            "mean_error": study["mean_error"],
            "std_error": study["std_error"],
            "rmse": study["rmse"],
            "mean_abs_error": study["mean_abs_error"],
            "mean_cost": study["mean_cost"],
            "min_error": study["min_error"],
            "max_error": study["max_error"]
        })

    return results


def print_hedging_study_table_with_costs(results, transaction_cost_bps):
    print(f"\nMany-Path Delta Hedging Study with Transaction Cost = {transaction_cost_bps:.1f} bps")
    print("-" * 110)
    print(
        f"{'Steps':>8} | {'Mean Error':>12} | {'Std Error':>12} | {'RMSE':>10} | "
        f"{'Mean |Error|':>12} | {'Mean Cost':>10} | {'Min Error':>12} | {'Max Error':>12}"
    )
    print("-" * 110)

    for row in results:
        print(
            f"{row['n_steps']:8d} | "
            f"{row['mean_error']:12.6f} | "
            f"{row['std_error']:12.6f} | "
            f"{row['rmse']:10.6f} | "
            f"{row['mean_abs_error']:12.6f} | "
            f"{row['mean_cost']:10.6f} | "
            f"{row['min_error']:12.6f} | "
            f"{row['max_error']:12.6f}"
        )


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

    transaction_cost_bps = 5.0
    hedge_steps_list = [12, 26, 52, 252]

    results = run_hedging_frequency_study_with_costs(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        hedge_steps_list=hedge_steps_list,
        n_paths=1000,
        transaction_cost_bps=transaction_cost_bps,
        base_seed=42
    )

    print_hedging_study_table_with_costs(results, transaction_cost_bps)
