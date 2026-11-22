import math

import numpy as np
import pytest

from mc_pricer import (
    paths_from_normals,
    simulate_gbm_paths,
    simulate_single_gbm_path,
    simulate_terminal_prices,
)


def test_terminal_prices_reproducible_and_positive(base):
    a = simulate_terminal_prices(base["S0"], base["T"], base["r"], base["sigma"], 1000, seed=1)
    b = simulate_terminal_prices(base["S0"], base["T"], base["r"], base["sigma"], 1000, seed=1)
    np.testing.assert_array_equal(a, b)
    assert (a > 0).all() and a.shape == (1000,)


def test_terminal_prices_are_martingale_after_discounting(base):
    n = 400_000
    st = simulate_terminal_prices(base["S0"], base["T"], base["r"], base["sigma"], n, seed=3)
    discounted = math.exp(-base["r"] * base["T"]) * st
    se = discounted.std(ddof=1) / math.sqrt(n)
    assert abs(discounted.mean() - base["S0"]) < 4 * se


def test_log_returns_have_gbm_moments(base):
    st = simulate_terminal_prices(base["S0"], base["T"], base["r"], base["sigma"], 400_000, seed=4)
    log_ret = np.log(st / base["S0"])
    assert log_ret.mean() == pytest.approx((base["r"] - 0.5 * base["sigma"] ** 2) * base["T"], abs=2e-3)
    assert log_ret.std() == pytest.approx(base["sigma"] * math.sqrt(base["T"]), rel=5e-3)


def test_explicit_normals_must_match_n_sims(base):
    with pytest.raises(ValueError, match="shape"):
        simulate_terminal_prices(base["S0"], base["T"], base["r"], base["sigma"], 100, Z=np.zeros(10))


def test_path_shapes_and_start(base):
    paths = simulate_gbm_paths(base["S0"], base["T"], base["r"], base["sigma"], n_steps=12, n_paths=50, seed=0)
    assert paths.shape == (50, 13)
    assert np.all(paths[:, 0] == base["S0"])
    single = simulate_single_gbm_path(base["S0"], base["T"], base["r"], base["sigma"], 12, seed=0)
    assert single.shape == (13,)


def test_path_terminal_distribution_matches_one_step_sampler(base):
    # Exact GBM: the endpoint of a multi-step path has the same law as a one-step draw.
    paths = simulate_gbm_paths(base["S0"], base["T"], base["r"], base["sigma"], n_steps=10, n_paths=200_000, seed=5)
    log_end = np.log(paths[:, -1] / base["S0"])
    assert log_end.std() == pytest.approx(base["sigma"] * math.sqrt(base["T"]), rel=5e-3)


def test_zero_volatility_path_is_deterministic(base):
    path = paths_from_normals(base["S0"], base["T"], base["r"], 0.0, np.random.default_rng(0).standard_normal((1, 4)))
    expected = base["S0"] * np.exp(base["r"] * np.linspace(0, base["T"], 5))
    np.testing.assert_allclose(path[0], expected)
