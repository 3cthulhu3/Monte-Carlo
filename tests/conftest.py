import pytest

# Hull's textbook example: ATM 1-year call, r = 5%, sigma = 20%.
BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def base():
    return dict(BASE)
