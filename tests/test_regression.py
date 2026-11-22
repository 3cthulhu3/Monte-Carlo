"""Pin the numbers produced by the original single-file implementation (commit 325f20e).

The refactor (package split, vectorised hedging) must not change any result for
the same inputs and seeds. Only the confidence interval moved by ~2e-6 because
the hard-coded 1.96 was replaced by the exact normal quantile 1.959964.
"""
import pytest

import mc_pricer as m

ORIGINAL_HEDGE_TABLE = [
    # n_steps, mean_error, std_error, rmse, mean_cost
    (12, -0.08905426553154386, 1.8314937372504498, 1.8327426449142814, 0.08170829350),
    (26, -0.055595, 1.301468, 1.302005, 0.107844),
    (52, -0.093639, 0.945735, 0.949889, 0.140779),
    (252, -0.282114, 0.457732, 0.537491, 0.269481),
]


def test_original_mc_price_and_standard_error():
    res = m.monte_carlo_call_price(100, 100, 1.0, 0.05, 0.20, 100_000, seed=42)
    assert res.price == pytest.approx(10.420541193153113, abs=1e-12)
    assert res.std_error == pytest.approx(0.046769902520818175, abs=1e-12)
    assert res.ci_lower == pytest.approx(10.32887218421231, abs=1e-5)
    assert res.ci_upper == pytest.approx(10.512210202093916, abs=1e-5)


def test_original_finite_difference_greeks():
    delta, gamma = m.monte_carlo_delta_gamma(100, 100, 1.0, 0.05, 0.20, 100_000, h=1.0, seed=42)
    assert delta == pytest.approx(0.634554735404282, abs=1e-12)
    assert gamma == pytest.approx(0.01864715091545044, abs=1e-12)


def test_original_hedging_table():
    rows = m.run_hedging_frequency_study_with_costs(100, 100, 1.0, 0.05, 0.20, [12, 26, 52, 252], 1000, 5.0, 42)
    for row, (n, mean_err, std_err, rmse, cost) in zip(rows, ORIGINAL_HEDGE_TABLE):
        assert row["n_steps"] == n
        assert row["mean_error"] == pytest.approx(mean_err, abs=1e-6)
        assert row["std_error"] == pytest.approx(std_err, abs=1e-6)
        assert row["rmse"] == pytest.approx(rmse, abs=1e-6)
        assert row["mean_cost"] == pytest.approx(cost, abs=1e-6)


def test_every_original_public_name_still_importable():
    original_names = [
        "validate_inputs", "normal_cdf", "normal_pdf", "simulate_terminal_prices",
        "simulate_single_gbm_path", "black_scholes_call_price", "black_scholes_call_delta",
        "black_scholes_call_gamma", "black_scholes_call_delta_tau", "monte_carlo_call_price",
        "monte_carlo_delta_gamma", "simulate_delta_hedge_one_path_with_costs",
        "simulate_delta_hedge_many_paths_with_costs", "run_hedging_frequency_study_with_costs",
        "print_hedging_study_table_with_costs",
    ]
    for name in original_names:
        assert callable(getattr(m, name)), name
