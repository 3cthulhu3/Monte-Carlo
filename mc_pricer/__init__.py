"""Monte Carlo option pricer.

European option pricing by Monte Carlo under Black-Scholes (GBM) dynamics, checked
against the closed-form benchmark, with Greeks, variance reduction and a study of
discrete delta hedging with transaction costs.

Every name that the original single-file ``mc_pricer.py`` exposed is still
importable from here, e.g. ``from mc_pricer import monte_carlo_call_price``.
"""
from .analytic import (
    black_scholes_call_delta,
    black_scholes_call_delta_tau,
    black_scholes_call_gamma,
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_delta_array,
    black_scholes_delta_tau,
    black_scholes_gamma,
    black_scholes_price,
    normal_cdf,
    normal_cdf_array,
    normal_pdf,
)
from .greeks import GreekEstimates, monte_carlo_delta_gamma, monte_carlo_greeks_pathwise
from .hedging import (
    hedge_paths,
    run_hedging_frequency_study_with_costs,
    simulate_delta_hedge_many_paths_with_costs,
    simulate_delta_hedge_one_path_with_costs,
)
from .pricing import (
    METHODS,
    PricingResult,
    convergence_study,
    estimate_convergence_rate,
    monte_carlo_call_price,
    monte_carlo_price,
    variance_reduction_study,
)
from .report import print_hedging_study_table_with_costs
from .simulation import (
    paths_from_normals,
    simulate_gbm_paths,
    simulate_single_gbm_path,
    simulate_terminal_prices,
)
from .validation import validate_inputs

__version__ = "0.2.0"

__all__ = [
    "GreekEstimates",
    "METHODS",
    "PricingResult",
    "black_scholes_call_delta",
    "black_scholes_call_delta_tau",
    "black_scholes_call_gamma",
    "black_scholes_call_price",
    "black_scholes_delta",
    "black_scholes_delta_array",
    "black_scholes_delta_tau",
    "black_scholes_gamma",
    "black_scholes_price",
    "convergence_study",
    "estimate_convergence_rate",
    "hedge_paths",
    "monte_carlo_call_price",
    "monte_carlo_delta_gamma",
    "monte_carlo_greeks_pathwise",
    "monte_carlo_price",
    "normal_cdf",
    "normal_cdf_array",
    "normal_pdf",
    "paths_from_normals",
    "print_hedging_study_table_with_costs",
    "run_hedging_frequency_study_with_costs",
    "simulate_delta_hedge_many_paths_with_costs",
    "simulate_delta_hedge_one_path_with_costs",
    "simulate_gbm_paths",
    "simulate_single_gbm_path",
    "simulate_terminal_prices",
    "validate_inputs",
    "variance_reduction_study",
]
