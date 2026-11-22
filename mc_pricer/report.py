"""Plain-text tables for the demo / CLI. Each ``format_*`` returns a string (testable);
``print_*`` wrappers write it to stdout."""
from __future__ import annotations


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "-" * (sum(widths) + 3 * (len(widths) - 1))
    fmt = lambda cells: " | ".join(c.rjust(w) for c, w in zip(cells, widths))  # noqa: E731
    return "\n".join([title, line, fmt(headers), line, *(fmt(r) for r in rows)])


def format_pricing_summary(bs_price, result, option_type: str = "call", confidence: float = 0.95) -> str:
    diff = abs(result.price - bs_price)
    diff_text = f"{diff:.6f}"
    if result.std_error > 0:
        diff_text += f"  ({diff / result.std_error:.2f} std. errors)"
    lines = [
        ("Black-Scholes price", f"{bs_price:.6f}"),
        ("Monte Carlo price", f"{result.price:.6f}"),
        ("Monte Carlo std. error", f"{result.std_error:.6f}"),
        (f"{confidence:.0%} confidence interval", f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}]"),
        ("|MC - BS|", diff_text),
        ("CI covers Black-Scholes", "yes" if result.contains(bs_price) else "NO"),
    ]
    width = max(len(label) for label, _ in lines)
    return "\n".join([f"European {option_type} pricing", *(f"  {lab.ljust(width)} : {val}" for lab, val in lines)])


def format_greeks_table(bs, fd, pathwise) -> str:
    rows = [
        ["Delta", f"{bs[0]:.6f}", f"{fd[0]:.6f}", f"{pathwise.delta:.6f} ± {pathwise.delta_se:.6f}"],
        ["Gamma", f"{bs[1]:.6f}", f"{fd[1]:.6f}", f"{pathwise.gamma:.6f} ± {pathwise.gamma_se:.6f}"],
    ]
    return _table("Greeks", ["Greek", "Black-Scholes", "MC finite diff.", "MC pathwise/LR (± SE)"], rows)


def format_convergence_table(rows, rate=None) -> str:
    body = [
        [f"{r['n_sims']:,}", f"{r['price']:.6f}", f"{r['std_error']:.6f}", f"{r['abs_error']:.6f}",
         f"{r['z_score']:+.2f}", "yes" if r["ci_covers_benchmark"] else "NO"]
        for r in rows
    ]
    text = _table(
        "Convergence (benchmark = Black-Scholes)",
        ["N", "MC price", "Std. error", "|Error|", "Error/SE", "CI covers"],
        body,
    )
    if rate is not None:
        text += f"\nFitted std-error slope vs N (log-log): {rate:.3f}  (theory: -0.500)"
    return text


def format_variance_reduction_table(rows) -> str:
    body = [
        [r["method"], f"{r['price']:.6f}", f"{r['std_error']:.6f}", f"{r['abs_error']:.6f}",
         f"{r['variance_ratio']:.2f}x"]
        for r in rows
    ]
    return _table(
        "Variance reduction (same number of payoff evaluations)",
        ["Method", "MC price", "Std. error", "|Error|", "Efficiency vs plain"],
        body,
    )


def format_hedging_table(results, transaction_cost_bps) -> str:
    body = [
        [f"{r['n_steps']:d}", f"{r['mean_error']:.6f}", f"{r['std_error']:.6f}", f"{r['rmse']:.6f}",
         f"{r['mean_abs_error']:.6f}", f"{r['mean_cost']:.6f}", f"{r['min_error']:.6f}", f"{r['max_error']:.6f}"]
        for r in results
    ]
    return _table(
        f"Many-path delta hedging study, transaction cost = {transaction_cost_bps:.1f} bps",
        ["Steps", "Mean error", "Std dev", "RMSE", "Mean |error|", "Mean cost", "Min error", "Max error"],
        body,
    )


def print_hedging_study_table_with_costs(results, transaction_cost_bps) -> None:
    """Backward-compatible printer for :func:`run_hedging_frequency_study_with_costs` output."""
    print("\n" + format_hedging_table(results, transaction_cost_bps))
