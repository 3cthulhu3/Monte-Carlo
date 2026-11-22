import numpy as np
import pytest

from mc_pricer import (
    METHODS,
    PricingResult,
    black_scholes_price,
    convergence_study,
    estimate_convergence_rate,
    monte_carlo_call_price,
    monte_carlo_price,
    variance_reduction_study,
)

# Seeds are fixed, so these "statistical" assertions are deterministic; the
# 4-standard-error tolerance means they would also pass for ~99.99% of seeds.
TOL_SE = 4.0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_mc_agrees_with_black_scholes_within_error_bars(base, method, option_type, K):
    base["K"] = K
    res = monte_carlo_price(**base, n_sims=100_000, option_type=option_type, method=method, seed=11)
    bs = black_scholes_price(**base, option_type=option_type)
    assert abs(res.price - bs) < TOL_SE * res.std_error


def test_result_is_backward_compatible_tuple(base):
    res = monte_carlo_call_price(base["S0"], base["K"], base["T"], base["r"], base["sigma"], 10_000, seed=1)
    assert isinstance(res, PricingResult) and isinstance(res.price, float)
    price, se, lo, hi = res  # original 4-tuple unpacking still works
    assert lo < price < hi and se > 0
    assert res.half_width == pytest.approx(1.959963984540054 * se)


def test_confidence_level_changes_interval_width(base):
    r90 = monte_carlo_price(**base, n_sims=10_000, seed=1, confidence=0.90)
    r99 = monte_carlo_price(**base, n_sims=10_000, seed=1, confidence=0.99)
    assert r90.price == r99.price
    assert r90.half_width < r99.half_width


def test_ci_coverage_is_close_to_nominal(base):
    bs = black_scholes_price(**base)
    hits = sum(monte_carlo_price(**base, n_sims=2_000, seed=s).contains(bs) for s in range(400))
    # Binomial(400, 0.95): mean 380, sd ~4.4
    assert 362 <= hits <= 396


def test_variance_reduction_methods_reduce_standard_error(base):
    rows = {r["method"]: r for r in variance_reduction_study(**base, n_sims=50_000, seed=2)}
    assert rows["plain"]["variance_ratio"] == pytest.approx(1.0)
    assert rows["antithetic"]["std_error"] < rows["plain"]["std_error"]
    assert rows["control_variate"]["std_error"] < rows["antithetic"]["std_error"]


def test_standard_error_scales_as_inverse_sqrt_n(base):
    rows = convergence_study(**base, n_sims_list=[1_000, 4_000, 16_000, 64_000, 256_000], seed=7)
    assert estimate_convergence_rate(rows) == pytest.approx(-0.5, abs=0.05)
    assert all(abs(r["z_score"]) < TOL_SE for r in rows)


def test_zero_volatility_gives_exact_price_and_zero_error(base):
    base["sigma"] = 0.0
    res = monte_carlo_price(**base, n_sims=100, seed=0)
    assert res.price == pytest.approx(black_scholes_price(**base))
    assert res.std_error == pytest.approx(0.0, abs=1e-12)


def test_common_random_numbers_via_explicit_z(base):
    Z = np.random.default_rng(9).standard_normal(5_000)
    a = monte_carlo_price(**base, n_sims=5_000, Z=Z)
    b = monte_carlo_price(**base, n_sims=5_000, Z=Z, seed=123)  # Z overrides seed
    assert a == b


def test_z_length_mismatch_is_rejected(base):
    # The original code silently used n_sims for the standard error -> wrong CI.
    with pytest.raises(ValueError, match="shape"):
        monte_carlo_call_price(**base, n_sims=100, Z=np.zeros(10_000))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_sims=1), "n_sims"),
        (dict(n_sims=5, method="antithetic"), "even"),
        (dict(n_sims=100, method="nope"), "method"),
        (dict(n_sims=100, option_type="straddle"), "option_type"),
        (dict(n_sims=100, confidence=1.0), "confidence"),
    ],
)
def test_invalid_pricing_arguments(base, kwargs, match):
    with pytest.raises(ValueError, match=match):
        monte_carlo_price(**base, **kwargs)
