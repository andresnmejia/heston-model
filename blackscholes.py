import numpy as np

from scipy.stats import norm


def __d1(S0, K, sigma, T, r):
    return (np.log(S0 / K) + (r + (0.5) * sigma**2) * T) / (sigma * np.sqrt(T))


def __d1d2(S0, K, sigma, T, r):
    d1 = (np.log(S0 / K) + (r + (0.5) * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (d1, d2)


def bs_call(S0, K, sigma, T, r):
    d1, d2 = __d1d2(S0, K, sigma, T, r)

    call_value = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_value


# quick delta
def delta(S0, K, T, r, sigma):
    return norm.cdf(__d1)


# quick vega
def vega(S0, K, T, r, sigma):
    d1_val = __d1(S0, K, sigma, T, r)
    return S0 * norm.pdf(d1_val) * np.sqrt(T)


def __jackel_initial_guess(market_price, S0, K, T, r):
    """
    This is a method found in ____ that sets an approprite initial guess for IV for faster convergence.
    """

    F = S0 * np.exp(r * T)
    x = np.log(F / K)
    c = market_price * np.exp(r * T)

    sigma = np.zeros(market_price.shape, dtype=float)

    # if ATM, use heuristic
    atm_mask = np.abs(x) < 0.01
    if np.any(atm_mask):
        sigma[atm_mask] = np.sqrt(2 * np.pi / T[atm_mask]) * c[atm_mask] / F[atm_mask]

    not_deep_itm_mask = (c < F * 0.5) & ~atm_mask

    if np.any(not_deep_itm_mask):
        T_subset = T[not_deep_itm_mask]
        x_subset = x[not_deep_itm_mask]
        c_subset = c[not_deep_itm_mask]
        F_subset = F[not_deep_itm_mask]

        beta = x_subset / 2
        gamma = 1 / np.sqrt(T_subset)
        alpha = 2 * c_subset / F_subset

        sigma_sqrt_T = gamma * alpha / (1 + np.sqrt(1 - alpha + alpha**2 / (1 + beta**2)))
        sigma[not_deep_itm_mask] = sigma_sqrt_T / np.sqrt(T_subset)

    # if can't get good estimate quickly, default to something reasonable.
    deep_itm_mask = ~atm_mask & ~not_deep_itm_mask
    sigma[deep_itm_mask] = 0.3

    return sigma


# We had to do Newton's method by hand here because scipy hates vectors, but it's relatively straightforward.
def __implied_volatility(market_price, S0, K, T, r, tolerance=1e-6, max_iter=100):
    market_price = np.atleast_1d(market_price)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)

    shape = np.broadcast_shapes(market_price.shape, K.shape, T.shape)
    market_price = np.broadcast_to(market_price, shape)
    K = np.broadcast_to(K, shape)
    T = np.broadcast_to(T, shape)

    # Initial guess
    sigma = __jackel_initial_guess(market_price, S0, K, T, r)
    sigma = np.clip(sigma, 0.01, 2.0)

    converged = np.zeros(shape, dtype=bool)

    for i in range(max_iter):
        # Compute prices and differences
        price = bs_call(S0, K, sigma, T, r)
        diff = price - market_price

        # Check convergence
        newly_converged = np.abs(diff) < tolerance
        converged |= newly_converged

        if np.all(converged):
            break

        # Compute vega
        v = vega(S0, K, T, r, sigma)

        # Avoid division by zero
        safe_v = np.where(v > 1e-10, v, 1.0)

        # Newton step
        sigma_new = sigma - diff / safe_v
        sigma_new = np.clip(sigma_new, 0.01, 2.0)

        # Check sigma change
        sigma_converged = np.abs(sigma_new - sigma) < tolerance
        converged |= sigma_converged

        sigma = sigma_new

    # Set non-converged to NaN
    sigma[~converged] = np.nan

    return sigma





#Useful functions

def GBM_paths(S0, sigma, t, r, mu, n_sims, n_steps):
    
    
    dt = t/n_steps
    noise = np.random.normal(loc = 0, scale = 1, size = (n_sims, n_steps))
    log_returns = (mu+r-sigma**2*(0.5))*dt + sigma*np.sqrt(dt)*noise
    exponent = np.cumsum(log_returns, axis = 1)
    paths = S0*np.exp(exponent)
    paths_with_start = np.insert(paths, 0, S0, axis = 1)

    return paths_with_start
