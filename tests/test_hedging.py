import numpy as np
import pytest

from mc_pricer import (
    hedge_paths,
    run_hedging_frequency_study_with_costs,
    simulate_delta_hedge_many_paths_with_costs,
    simulate_delta_hedge_one_path_with_costs,
    simulate_gbm_paths,
)


def test_vectorised_many_paths_equals_path_by_path_loop(base):
    many = simulate_delta_hedge_many_paths_with_costs(**base, n_steps=26, n_paths=50, transaction_cost_bps=5.0, base_seed=3)
    loop = [
        simulate_delta_hedge_one_path_with_costs(**base, n_steps=26, transaction_cost_bps=5.0, seed=3 + i)
        for i in range(50)
    ]
    np.testing.assert_allclose(many["hedging_errors"], [r["hedging_error"] for r in loop], atol=1e-12)
    np.testing.assert_allclose(many["transaction_costs"], [r["total_transaction_cost"] for r in loop], atol=1e-12)


def test_one_path_accounting_identity(base):
    r = simulate_delta_hedge_one_path_with_costs(**base, n_steps=52, transaction_cost_bps=10.0, seed=1)
    assert r["hedging_error"] == pytest.approx(r["terminal_portfolio"] - r["option_payoff"])
    assert r["option_payoff"] == pytest.approx(max(r["terminal_stock"] - base["K"], 0.0))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rmse_shrinks_like_inverse_sqrt_of_rebalancing_frequency(base, option_type):
    steps = [13, 52, 208]
    rows = run_hedging_frequency_study_with_costs(**base, hedge_steps_list=steps, n_paths=2_000, option_type=option_type)
    rmse = np.array([r["rmse"] for r in rows])
    assert np.all(np.diff(rmse) < 0)
    slope = np.polyfit(np.log(steps), np.log(rmse), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.1)
    # Without costs, discrete hedging is unbiased on average.
    for r in rows:
        assert abs(r["mean_error"]) < 4 * r["std_error"] / np.sqrt(2_000)


def test_transaction_costs_grow_with_frequency_and_bias_error_down(base):
    rows = run_hedging_frequency_study_with_costs(**base, hedge_steps_list=[12, 52, 252], n_paths=500, transaction_cost_bps=10.0)
    costs = [r["mean_cost"] for r in rows]
    assert costs == sorted(costs) and costs[0] > 0
    free = run_hedging_frequency_study_with_costs(**base, hedge_steps_list=[12, 52, 252], n_paths=500)
    for with_cost, without in zip(rows, free):
        assert without["mean_cost"] == 0.0
        # Same paths: the mean error drops by roughly the mean cost (plus interest on it).
        assert with_cost["mean_error"] < without["mean_error"]
        assert without["mean_error"] - with_cost["mean_error"] == pytest.approx(with_cost["mean_cost"], rel=0.06)


def test_closing_cost_option_charges_unwind(base):
    paths = simulate_gbm_paths(base["S0"], base["T"], base["r"], base["sigma"], 52, 100, seed=0)
    kw = dict(K=base["K"], T=base["T"], r=base["r"], sigma=base["sigma"], transaction_cost_bps=5.0)
    open_ = hedge_paths(paths, **kw)
    closed = hedge_paths(paths, **kw, charge_closing_cost=True)
    assert np.all(closed["total_transaction_cost"] >= open_["total_transaction_cost"])
    assert np.all(closed["hedging_error"] <= open_["hedging_error"])


def test_zero_volatility_hedge_is_exact(base):
    base["sigma"] = 0.0
    r = simulate_delta_hedge_one_path_with_costs(**base, n_steps=12, seed=0)
    assert r["hedging_error"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("kwargs", [dict(transaction_cost_bps=-1.0), dict(option_type="digital")])
def test_invalid_hedging_arguments(base, kwargs):
    with pytest.raises(ValueError):
        simulate_delta_hedge_one_path_with_costs(**base, n_steps=12, seed=0, **kwargs)


def test_empty_frequency_list_rejected(base):
    with pytest.raises(ValueError, match="empty"):
        run_hedging_frequency_study_with_costs(**base, hedge_steps_list=[], n_paths=10)
