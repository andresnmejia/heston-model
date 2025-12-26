# hestonmodel
A Python Implementation of Heston Model Calibration via Gradient-Based Optimization using the analytic formula as developed in [Full and fast calibration of the Heston stochastic volatility model](https://arxiv.org/abs/1511.08718.)

hestoncalibration.py includes different regularization parameters to fit market data, including the Fisher information metric, ordinary ridge regression, and a Feller condition penalty. Details can be found in the notebook example.

heston.py includes all of the formulas necessary to obtain an analytic gradient. 

MarketData.py is a class wrapper for dealing with the yfinance API. Included are various ways of excluding data, both from economic heuristics (no-arbitrage, monotonicity, reasonable implied volatility) as well as basic outlier detection.

In the notebook, we investigate various ways of mitigating model instability for call options via regularization, data cleaning, dimension reduction, and setting good beginning parameters. The main novelty is using Fisher information to avoid situations with high reversion to mean volatility and bad v_0 fits or low reversion to mean and bad long term volatility fits.
