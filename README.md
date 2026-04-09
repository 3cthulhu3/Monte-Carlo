# Monte Carlo Option Pricer

A Python project for pricing European call options with Monte Carlo simulation, comparing results to the Black-Scholes benchmark, estimating Greeks, and analyzing discrete delta hedging with transaction costs.

## Features

- Monte Carlo pricing for a European call option
- Black-Scholes analytical benchmark
- 95% confidence interval and convergence analysis
- Delta and Gamma estimation
- One-path and many-path discrete delta hedging
- Transaction-cost hedging frequency study

## Project structure

```text
monte-carlo-option-pricer/
├── mc_pricer.py
├── requirements.txt
├── README.md
└── tests/
    └── test_mc_pricer.py
```

## Installation

1. Clone the repository.
2. Open the project folder in VS Code.
3. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Run the project

To run the main script:

```bash
py mc_pricer.py
```

## Run tests

To run the automated test suite:

```bash
py -m pytest
```

## What the script outputs

The script prints:
- Black-Scholes call price
- Monte Carlo call price
- Monte Carlo standard error
- 95% confidence interval
- Delta and Gamma comparison
- Many-path delta hedging study with transaction costs

## Notes

This project assumes geometric Brownian motion and Black-Scholes dynamics under risk-neutral pricing. The hedging section illustrates discrete-time replication error and the tradeoff between hedge accuracy and transaction costs.
