import pytest

from mc_pricer import (
    black_scholes_delta,
    black_scholes_gamma,
    monte_carlo_delta_gamma,
    monte_carlo_greeks_pathwise,
)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
def test_pathwise_greeks_within_error_bars(base, option_type, K):
    base["K"] = K
    g = monte_carlo_greeks_pathwise(**base, n_sims=200_000, option_type=option_type, seed=21)
    assert abs(g.delta - black_scholes_delta(**base, option_type=option_type)) < 4 * g.delta_se
    assert abs(g.gamma - black_scholes_gamma(**base)) < 4 * g.gamma_se


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_finite_difference_greeks_close_to_black_scholes(base, option_type):
    delta, gamma = monte_carlo_delta_gamma(**base, n_sims=200_000, h=1.0, seed=42, option_type=option_type)
    assert delta == pytest.approx(black_scholes_delta(**base, option_type=option_type), abs=0.01)
    assert gamma == pytest.approx(black_scholes_gamma(**base), rel=0.05)


def test_bump_size_is_validated(base):
    with pytest.raises(ValueError, match="h must satisfy"):
        monte_carlo_delta_gamma(**base, n_sims=1_000, h=100.0)
    with pytest.raises(ValueError, match="h must satisfy"):
        monte_carlo_delta_gamma(**base, n_sims=1_000, h=0.0)


def test_pathwise_requires_positive_volatility(base):
    base["sigma"] = 0.0
    with pytest.raises(ValueError, match="sigma > 0"):
        monte_carlo_greeks_pathwise(**base, n_sims=1_000)
