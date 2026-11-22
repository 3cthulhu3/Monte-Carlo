"""Plot the two headline studies and save them as a PNG.

    pip install -e ".[examples]"
    python examples/plot_studies.py            # writes docs/studies.png

Left: standard error vs sample size for each estimator (log-log). All lines have
slope -1/2; variance reduction shifts the line down, i.e. the same accuracy for
fewer samples. Right: the hedging trade-off - replication error (RMSE) falls as
you rebalance more often while transaction costs rise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mc_pricer import METHODS, monte_carlo_price, run_hedging_frequency_study_with_costs  # noqa: E402

PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
COLORS = {"plain": "#2a78d6", "antithetic": "#eb6834", "control_variate": "#1baf7a"}
MARKERS = {"plain": "o", "antithetic": "s", "control_variate": "^"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main(out_path: str = "docs/studies.png") -> Path:
    sizes = [1_000, 4_000, 16_000, 64_000, 256_000]
    steps = [6, 12, 26, 52, 126, 252]
    hedge = run_hedging_frequency_study_with_costs(**PARAMS, hedge_steps_list=steps, n_paths=2_000,
                                                   transaction_cost_bps=5.0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for method in METHODS:
        se = [monte_carlo_price(**PARAMS, n_sims=n, method=method, seed=1).std_error for n in sizes]
        left.loglog(sizes, se, color=COLORS[method], marker=MARKERS[method], markersize=6, linewidth=2,
                    label=method.replace("_", " "))
        left.annotate(method.replace("_", " "), (sizes[-1], se[-1]), xytext=(6, 0),
                      textcoords="offset points", va="center", color=INK, fontsize=9)
    _style(left, "MC standard error vs samples (slope −1/2)", "payoff evaluations N", "standard error")
    left.set_xlim(right=sizes[-1] * 4)
    left.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left")

    rmse = [row["rmse"] for row in hedge]
    cost = [row["mean_cost"] for row in hedge]
    right.plot(steps, rmse, color=COLORS["plain"], marker="o", markersize=6, linewidth=2, label="hedging RMSE")
    right.plot(steps, cost, color=COLORS["antithetic"], marker="s", markersize=6, linewidth=2,
               label="mean transaction cost (5 bps)")
    right.annotate("hedging RMSE", (steps[0], rmse[0]), xytext=(8, 0), textcoords="offset points",
                   va="center", color=INK, fontsize=9)
    right.annotate("mean cost", (steps[-1], cost[-1]), xytext=(0, 10), textcoords="offset points",
                   ha="right", color=INK, fontsize=9)
    right.set_xscale("log")
    right.set_xticks(steps, [str(s) for s in steps])
    _style(right, "Delta hedging: accuracy vs cost", "rebalancing steps per year", "currency units")
    right.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
