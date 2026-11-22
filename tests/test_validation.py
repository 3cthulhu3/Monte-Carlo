import math

import pytest

from mc_pricer import black_scholes_price, monte_carlo_price, validate_inputs


@pytest.mark.parametrize(
    "override, exc, match",
    [
        (dict(S0=0.0), ValueError, "S0"),
        (dict(S0=-1.0), ValueError, "S0"),
        (dict(K=0.0), ValueError, "K"),
        (dict(K=-5.0), ValueError, "K"),  # original MC code silently priced a negative strike
        (dict(T=0.0), ValueError, "T"),
        (dict(sigma=-0.1), ValueError, "sigma"),
        (dict(r=math.nan), ValueError, "r"),
        (dict(S0=math.inf), ValueError, "S0"),
        (dict(S0="100"), TypeError, "S0"),
    ],
)
def test_bad_market_inputs_are_rejected_everywhere(base, override, exc, match):
    base.update(override)
    with pytest.raises(exc, match=match):
        black_scholes_price(**base)
    with pytest.raises(exc, match=match):
        monte_carlo_price(**base, n_sims=100, seed=0)


@pytest.mark.parametrize("n_sims, exc", [(0, ValueError), (2.5, TypeError), (True, TypeError)])
def test_integer_counts(n_sims, exc):
    with pytest.raises(exc, match="n_sims"):
        validate_inputs(100, 1, 0.05, 0.2, n_sims)


def test_numpy_scalars_are_accepted():
    import numpy as np

    validate_inputs(np.float64(100), np.float32(1.0), 0.05, 0.2, np.int64(10), np.int32(5))
