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


if __name__ == "__main__":
    S0 = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    n_sims = 10000
    n_steps = 252

    ST = simulate_terminal_prices(S0, T, r, sigma, n_sims, seed=42)
    paths = simulate_gbm_paths(S0, T, r, sigma, n_sims, n_steps, seed=42)

    print("Terminal prices shape:", ST.shape)
    print("Paths shape:", paths.shape)
    print("First path first 5 values:", paths[0, :5])
    print("Mean terminal price:", ST.mean())
    print("Risk-neutral expected terminal price:", S0 * np.exp(r * T))
    print("All path values positive:", np.all(paths > 0))
    print("Mean path terminal price:", paths[:, -1].mean())
    
