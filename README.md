# hestonmodel
A Python Implementation of Heston Model Calibration via Gradient-Based Optimization using the analytic formula as developed in [Full and fast calibration of the Heston stochastic volatility model](https://arxiv.org/abs/1511.08718.)

heston.py includes all of the formulas necessary to obtain an analytic gradient. 

MarketData.py is a class wrapper for dealing with the yfinance API. Included are various ways of excluding data, both from economic heuristics (no-arbitrage, monotonicity, reasonable implied volatility) as well as basic outlier detection.

In the notebook, we investigate various ways of mitigating model instability for call options via regularization, data cleaning, dimension reduction, and setting good beginning parameters.
