"""Minimal library usage: python examples/quickstart.py"""
from mc_pricer import (
    black_scholes_price,
    monte_carlo_greeks_pathwise,
    monte_carlo_price,
    simulate_delta_hedge_many_paths_with_costs,
)

params = dict(S0=100.0, K=105.0, T=0.5, r=0.03, sigma=0.25)

# 1) Price a put three ways and check each against the closed form.
bs = black_scholes_price(**params, option_type="put")
for method in ("plain", "antithetic", "control_variate"):
    res = monte_carlo_price(**params, n_sims=200_000, option_type="put", method=method, seed=0)
    print(f"{method:>16}: {res.price:.4f} ± {res.std_error:.4f}   BS = {bs:.4f}   in CI: {res.contains(bs)}")

# 2) Greeks with error bars.
g = monte_carlo_greeks_pathwise(**params, n_sims=200_000, option_type="put", seed=0)
print(f"delta = {g.delta:.4f} ± {g.delta_se:.4f}, gamma = {g.gamma:.5f} ± {g.gamma_se:.5f}")

# 3) Weekly delta hedging with 10 bps costs.
study = simulate_delta_hedge_many_paths_with_costs(**params, n_steps=26, n_paths=5_000,
                                                   transaction_cost_bps=10.0, option_type="put")
print(f"hedging RMSE = {study['rmse']:.4f}, mean cost = {study['mean_cost']:.4f}")
