from mc_pricer import (
    black_scholes_call_price,
    black_scholes_call_delta,
    black_scholes_call_gamma,
    monte_carlo_call_price,
    monte_carlo_delta_gamma,
    simulate_terminal_prices,
)


def test_black_scholes_reference_price():
    price = black_scholes_call_price(100, 100, 1.0, 0.05, 0.20)
    assert abs(price - 10.4506) < 0.02


def test_black_scholes_delta_range():
    delta = black_scholes_call_delta(100, 100, 1.0, 0.05, 0.20)
    assert 0.0 < delta < 1.0


def test_black_scholes_gamma_positive():
    gamma = black_scholes_call_gamma(100, 100, 1.0, 0.05, 0.20)
    assert gamma > 0.0


def test_terminal_prices_positive():
    st = simulate_terminal_prices(100, 1.0, 0.05, 0.20, 10000, seed=42)
    assert (st > 0).all()


def test_monte_carlo_close_to_black_scholes():
    mc_price, _, _, _ = monte_carlo_call_price(100, 100, 1.0, 0.05, 0.20, 200000, seed=42)
    bs_price = black_scholes_call_price(100, 100, 1.0, 0.05, 0.20)
    assert abs(mc_price - bs_price) < 0.15


def test_monte_carlo_delta_gamma_reasonable():
    mc_delta, mc_gamma = monte_carlo_delta_gamma(100, 100, 1.0, 0.05, 0.20, 200000, h=1.0, seed=42)
    assert 0.0 < mc_delta < 1.0
    assert mc_gamma > 0.0