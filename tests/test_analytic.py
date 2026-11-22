import math

import numpy as np
import pytest

from mc_pricer import (
    black_scholes_delta,
    black_scholes_delta_array,
    black_scholes_delta_tau,
    black_scholes_gamma,
    black_scholes_price,
    normal_cdf,
    normal_cdf_array,
)


def test_reference_prices(base):
    # Published values for S=K=100, T=1, r=5%, sigma=20%.
    assert black_scholes_price(**base) == pytest.approx(10.450583572185565, abs=1e-12)
    assert black_scholes_price(**base, option_type="put") == pytest.approx(5.573526022256971, abs=1e-12)


@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 115.0, 180.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6])
def test_put_call_parity(base, K, sigma):
    base.update(K=K, sigma=sigma)
    call = black_scholes_price(**base)
    put = black_scholes_price(**base, option_type="put")
    assert call - put == pytest.approx(base["S0"] - K * math.exp(-base["r"] * base["T"]), abs=1e-10)


def test_delta_and_gamma_match_finite_differences_of_price(base):
    h = 1e-3
    up = dict(base, S0=base["S0"] + h)
    down = dict(base, S0=base["S0"] - h)
    for kind in ("call", "put"):
        fd_delta = (black_scholes_price(**up, option_type=kind) - black_scholes_price(**down, option_type=kind)) / (2 * h)
        assert black_scholes_delta(**base, option_type=kind) == pytest.approx(fd_delta, abs=1e-7)
    fd_gamma = (black_scholes_price(**up) - 2 * black_scholes_price(**base) + black_scholes_price(**down)) / h**2
    assert black_scholes_gamma(**base) == pytest.approx(fd_gamma, abs=1e-4)


def test_call_and_put_delta_ranges(base):
    assert 0.0 < black_scholes_delta(**base) < 1.0
    assert -1.0 < black_scholes_delta(**base, option_type="put") < 0.0
    assert black_scholes_delta(**base) - black_scholes_delta(**base, option_type="put") == pytest.approx(1.0)


def test_zero_volatility_limit_is_discounted_forward_intrinsic(base):
    base["sigma"] = 0.0
    forward_k = base["K"] * math.exp(-base["r"] * base["T"])
    assert black_scholes_price(**base) == pytest.approx(base["S0"] - forward_k)
    assert black_scholes_price(**base, option_type="put") == pytest.approx(0.0)
    assert black_scholes_delta(**base) == 1.0
    assert black_scholes_gamma(**base) == 0.0


def test_tiny_volatility_converges_to_zero_vol_limit(base):
    near = black_scholes_price(**dict(base, sigma=1e-8))
    limit = black_scholes_price(**dict(base, sigma=0.0))
    assert near == pytest.approx(limit, abs=1e-8)


def test_delta_tau_at_expiry_is_payoff_slope():
    assert black_scholes_delta_tau(105, 100, 0.0, 0.05, 0.2) == 1.0
    assert black_scholes_delta_tau(95, 100, 0.0, 0.05, 0.2) == 0.0
    assert black_scholes_delta_tau(95, 100, 0.0, 0.05, 0.2, "put") == -1.0
    assert black_scholes_delta_tau(105, 100, -1.0, 0.05, 0.2, "put") == 0.0


def test_vectorised_helpers_match_scalar_versions():
    xs = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(normal_cdf_array(xs), [normal_cdf(x) for x in xs], atol=1e-15)
    spots = np.array([70.0, 95.0, 100.0, 130.0])
    for kind in ("call", "put"):
        expected = [black_scholes_delta_tau(s, 100, 0.5, 0.05, 0.2, kind) for s in spots]
        np.testing.assert_allclose(black_scholes_delta_array(spots, 100, 0.5, 0.05, 0.2, kind), expected, atol=1e-15)
