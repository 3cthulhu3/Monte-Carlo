# Changelog

## 0.2.0

Goal of this release: make the project trustworthy as a demonstration of Monte Carlo
pricing (every number checkable against theory) and usable beyond one hard-coded run.
Results for the original inputs and seeds are unchanged (pinned in `tests/test_regression.py`).

### Fixed
- **Wrong standard error when `Z` was passed with a length different from `n_sims`.**
  The SE divided by `sqrt(n_sims)` instead of the real sample size (e.g. SE 1.46 instead of
  0.146). Now a `ValueError` is raised.
- **Negative or zero strike was silently priced** by the Monte Carlo functions. Now rejected.
- **`sigma = 0` crashed Black-Scholes and hedging with `ZeroDivisionError`** although the MC
  code accepted it. All functions now return the deterministic limit.
- `n_sims = 1` returned `NaN` standard errors with runtime warnings → now rejected (`n_sims >= 2`).
- `h >= S0` in finite-difference Greeks failed with the misleading message "S0 must be positive".
- Non-integer counts (`n_sims=2.5`) raised an obscure NumPy error; NaN/inf inputs were accepted.
- Printed confidence interval showed `np.float64(...)`; results are now plain floats.
- `notebooks` was a one-byte file, not a folder. Removed; runnable scripts are in `examples/`.
- README advertised a "convergence analysis" that did not exist in the code. Implemented.

### Changed
- Single 300-line `mc_pricer.py` split into the `mc_pricer/` package (analytic, simulation,
  pricing, greeks, hedging, report, validation, cli). Every original name is re-exported, so
  `from mc_pricer import ...` keeps working.
- **Run command:** `py mc_pricer.py` → `python -m mc_pricer` (a file and a package cannot
  share the name).
- `monte_carlo_call_price` returns a `PricingResult` named tuple; 4-tuple unpacking still works.
- 95 % CI uses the exact normal quantile (1.959964) instead of 1.96 (CI moves by ~2e-6).
- Many-path hedging is vectorised across paths (same seeds per path → identical numbers).
- `numpy==2.4.4` pin relaxed to `numpy>=1.24`; packaging via `pyproject.toml`.

### Added
- Put options everywhere (`option_type="put"`), put-call parity tests.
- Variance reduction: antithetic variates and control variate; `variance_reduction_study`.
- `convergence_study` + `estimate_convergence_rate` (log-log slope, theory −0.5).
- Pathwise delta and mixed pathwise/likelihood-ratio gamma with standard errors.
- `confidence` parameter, `charge_closing_cost` option for hedging, `hedge_paths` for custom paths.
- Command-line interface with all parameters, `--quick` and `--sections`.
- 107 tests (was 6), GitHub Actions on Linux + Windows, Python 3.10–3.13, with and without SciPy.
- `examples/quickstart.py`, `examples/plot_studies.py`, `docs/studies.png`.
