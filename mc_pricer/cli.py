"""Command-line demo: ``python -m mc_pricer --help``."""
from __future__ import annotations

import argparse
import sys

from .analytic import black_scholes_delta, black_scholes_gamma, black_scholes_price
from .greeks import monte_carlo_delta_gamma, monte_carlo_greeks_pathwise
from .hedging import run_hedging_frequency_study_with_costs
from .pricing import (
    METHODS,
    convergence_study,
    estimate_convergence_rate,
    monte_carlo_price,
    variance_reduction_study,
)
from .report import (
    format_convergence_table,
    format_greeks_table,
    format_hedging_table,
    format_pricing_summary,
    format_variance_reduction_table,
)

SECTIONS = ("pricing", "greeks", "convergence", "variance", "hedging")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mc_pricer",
        description="Monte Carlo pricing of European options vs Black-Scholes, Greeks, "
                    "variance reduction and discrete delta hedging with transaction costs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m = p.add_argument_group("market / contract")
    m.add_argument("--S0", type=float, default=100.0, help="spot price")
    m.add_argument("--K", type=float, default=100.0, help="strike")
    m.add_argument("--T", type=float, default=1.0, help="maturity in years")
    m.add_argument("--r", type=float, default=0.05, help="continuously compounded risk-free rate")
    m.add_argument("--sigma", type=float, default=0.20, help="volatility")
    m.add_argument("--option-type", choices=("call", "put"), default="call")

    s = p.add_argument_group("simulation")
    s.add_argument("--n-sims", type=int, default=100_000, help="Monte Carlo samples for pricing/Greeks")
    s.add_argument("--method", choices=METHODS, default="plain", help="pricing estimator")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--h", type=float, default=1.0, help="finite-difference bump for Greeks")

    h = p.add_argument_group("hedging")
    h.add_argument("--cost-bps", type=float, default=5.0, help="proportional transaction cost in bps")
    h.add_argument("--hedge-steps", type=int, nargs="+", default=[12, 26, 52, 252],
                   help="rebalancing frequencies to compare")
    h.add_argument("--n-paths", type=int, default=1000, help="paths per hedging frequency")
    h.add_argument("--charge-closing-cost", action="store_true",
                   help="also pay transaction costs to unwind the hedge at maturity")

    p.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                   help="which parts of the report to run")
    p.add_argument("--quick", action="store_true", help="small sample sizes for a fast smoke run")
    return p


def run(args: argparse.Namespace) -> str:
    if args.quick:
        args.n_sims = min(args.n_sims, 20_000)
        args.n_paths = min(args.n_paths, 200)
    S0, K, T, r, sigma, kind = args.S0, args.K, args.T, args.r, args.sigma, args.option_type
    blocks = []

    if "pricing" in args.sections:
        bs = black_scholes_price(S0, K, T, r, sigma, kind)
        mc = monte_carlo_price(S0, K, T, r, sigma, args.n_sims, kind, args.method, seed=args.seed)
        blocks.append(format_pricing_summary(bs, mc, f"{kind} ({args.method} MC)"))

    if "greeks" in args.sections:
        bs = (black_scholes_delta(S0, K, T, r, sigma, kind), black_scholes_gamma(S0, K, T, r, sigma, kind))
        fd = monte_carlo_delta_gamma(S0, K, T, r, sigma, args.n_sims, h=args.h, seed=args.seed, option_type=kind)
        pw = monte_carlo_greeks_pathwise(S0, K, T, r, sigma, args.n_sims, kind, seed=args.seed)
        blocks.append(format_greeks_table(bs, fd, pw))

    if "convergence" in args.sections:
        sizes = [n for n in (1_000, 4_000, 16_000, 64_000, 256_000) if n <= max(args.n_sims, 1_000) * 4]
        rows = convergence_study(S0, K, T, r, sigma, sizes, kind, args.method, seed=args.seed)
        rate = estimate_convergence_rate(rows) if len(rows) >= 2 else None
        blocks.append(format_convergence_table(rows, rate))

    if "variance" in args.sections:
        rows = variance_reduction_study(S0, K, T, r, sigma, args.n_sims, kind, seed=args.seed)
        blocks.append(format_variance_reduction_table(rows))

    if "hedging" in args.sections:
        rows = run_hedging_frequency_study_with_costs(
            S0, K, T, r, sigma, args.hedge_steps, args.n_paths, args.cost_bps, args.seed,
            option_type=kind, charge_closing_cost=args.charge_closing_cost,
        )
        blocks.append(format_hedging_table(rows, args.cost_bps))

    return "\n\n".join(blocks)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(run(args))
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
