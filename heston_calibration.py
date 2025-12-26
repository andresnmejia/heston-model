

from typing import Optional, Literal, Callable, Union, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
import warnings

# Assume these imports from your existing files
from heston import HestonModel
from marketdata import MarketData


@dataclass
class RegularizationConfig:
    """
    Configuration for different regularization strategies.
    
    Attributes
    ----------
    ridge_weight : float
        Weight for Ridge/Tikhonov regularization (L2 penalty on parameter deviation)
    ridge_prior : np.ndarray, optional
        Prior parameters for Ridge regularization. If None, uses initial guess.
    ridge_param_weights : np.ndarray, optional
        Per-parameter weights for Ridge regularization. Default: [1, 1, 0.5, 0.5, 1]
    feller_weight : float
        Weight for Feller condition soft penalty
    fisher_weight : float
        Weight for Fisher information metric regularization
    fisher_prior : np.ndarray, optional
        Prior parameters for Fisher regularization. If None, uses initial guess.
    regularization_type : {'none', 'ridge', 'fisher', 'combined'}
        Which regularization strategy to use
    """
    # Ridge/Tikhonov regularization
    ridge_weight: float = 0.0
    ridge_prior: Optional[np.ndarray] = None
    ridge_param_weights: Optional[np.ndarray] = None  # Per-parameter weights
    
    # Feller condition penalty
    feller_weight: float = 0.0
    
    # Fisher information regularization
    fisher_weight: float = 0.0
    fisher_prior: Optional[np.ndarray] = None
    
    # Which regularization to use
    regularization_type: Literal['none', 'ridge', 'fisher', 'combined'] = 'ridge'


class HestonFisherMetric:
    """
    Fisher Information Matrix for Heston model parameters.
    
    For option pricing context, we use the empirical Fisher information:
    G_ij(θ) = ∑_k (∂C_k/∂θ_i)(∂C_k/∂θ_j) / σ²_k
    
    where C_k is the price of option k and σ²_k is the pricing error variance.
    """
    
    def __init__(self, 
                 S0: np.ndarray,  # Now accepts array
                 r: float, 
                 strikes: np.ndarray, 
                 maturities: np.ndarray,
                 price_variance: float = 1.0):
        """
        Parameters
        ----------
        S0 : np.ndarray
            Spot prices (one per option)
        r : float
            Risk-free rate
        strikes, maturities : np.ndarray
            Grid of options for computing Fisher information
        price_variance : float
            Variance of pricing errors (for weighting).
            Use 1.0 for unweighted, or estimate from bid-ask spreads.
        """
        self.S0 = S0
        self.r = r
        self.strikes = strikes
        self.maturities = maturities
        self.price_variance = price_variance
        
        # Cache for expensive computations
        self._fisher_cache = {}
    
    def metric_tensor(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information matrix G(θ).
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters [v0, v̄, ρ, κ, σ]
        
        Returns
        -------
        G : np.ndarray, shape (5, 5)
            Fisher information matrix at parameters θ
        """
        cache_key = tuple(np.round(theta, decimals=6))
        if cache_key in self._fisher_cache:
            return self._fisher_cache[cache_key]
        
        model = HestonModel.from_array(theta)
        
        # Jacobian matrix: ∂C/∂θ for each option
        # Shape should be (5, n_options) from your _gradient method
        jacobian = model._gradient(
            self.S0, self.r, 
            self.strikes, self.maturities
        )  # Shape: (5, n_options)
        
        # Transpose to (n_options, 5) for Fisher computation
        if jacobian.shape[0] == 5:
            jacobian = jacobian.T  # Now (n_options, 5)
        
        # Fisher information: G = J^T W J / n
        # where W is diagonal weight matrix (1/σ² for each option)
        weights = 1.0 / self.price_variance
        if np.isscalar(weights):
            G = jacobian.T @ jacobian * weights / len(self.strikes)
        else:
            W = np.diag(weights)
            G = jacobian.T @ W @ jacobian / len(self.strikes)
        
        # Add small regularization for numerical stability
        epsilon = 1e-8
        G += epsilon * np.eye(5)
        
        # Cache it
        if len(self._fisher_cache) < 100:
            self._fisher_cache[cache_key] = G
        
        return G


class HestonCalibrator:
    """
    Flexible Heston model calibrator with multiple regularization options.
    
    Supports:
    - No regularization (standard least squares)
    - Ridge/Tikhonov regularization (Euclidean penalty)
    - Fisher information regularization (geometric penalty)
    - Combined regularization strategies
    - Soft Feller condition penalty
   """
    
    def __init__(self, 
                 market_data: MarketData,
                 bounds: Optional[list[tuple[float, float]]] = None):
        """
        Parameters
        ----------
        market_data : MarketData
            Market option prices and parameters (spot_price is now an array)
        bounds : list of tuples, optional
            Parameter bounds [(v0_min, v0_max), (v_bar_min, v_bar_max), ...]
            Default: [(0.001, 1.0), (0.001, 1.0), (-0.999, 0.999), (0.01, 10.0), (0.01, 2.0)]
        """
        self.market_data = market_data
        
        # Default bounds: [v0, v_bar, rho, kappa, sigma]
        if bounds is None:
            self.bounds = [
                (0.001, 1.0),    # v0
                (0.001, 1.0),    # v_bar
                (-0.999, 0.999), # rho
                (0.01, 10.0),    # kappa
                (0.01, 2.0)      # sigma
            ]
        else:
            self.bounds = bounds
        
        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])
        
        # Cache for Fisher information (computed lazily)
        self._fisher_metric = None
    
    def _get_fisher_metric(self) -> HestonFisherMetric:
        """Lazy initialization of Fisher metric"""
        if self._fisher_metric is None:
            self._fisher_metric = HestonFisherMetric(
                self.market_data.spot_price,  # Now an array
                self.market_data.risk_free_rate,
                self.market_data.strikes,
                self.market_data.maturities
            )
        return self._fisher_metric
    
    def calibrate(self,
                  initial_guess: Optional[HestonModel] = None,
                  reg_config: Optional[RegularizationConfig] = None,
                  verbose: int = 0,
                  max_nfev: int = 500) -> HestonModel:
        """
        Calibrate Heston model with specified regularization.
        
        Parameters
        ----------
        initial_guess : HestonModel, optional
            Initial parameter guess. If None, uses ATM short-term IV.
        reg_config : RegularizationConfig, optional
            Regularization settings. If None, uses no regularization.
        verbose : int
            Verbosity level for optimizer (0=silent, 1=summary, 2=detailed)
        max_nfev : int
            Maximum number of function evaluations
        
        Returns
        -------
        HestonModel
            Calibrated model
        """
        if reg_config is None:
            reg_config = RegularizationConfig(regularization_type='none')
        
        # Initial guess
        x0 = self._get_initial_guess(initial_guess)
        
        # Set up priors for regularization
        ridge_prior = reg_config.ridge_prior if reg_config.ridge_prior is not None else x0.copy()
        fisher_prior = reg_config.fisher_prior if reg_config.fisher_prior is not None else x0.copy()
        
        # Build residual and Jacobian functions
        residuals_fn: Callable[[np.ndarray], np.ndarray]
        jacobian_fn: Union[Callable[[np.ndarray], np.ndarray], str]
        
        if reg_config.regularization_type == 'none':
            residuals_fn = self._residuals_no_reg
            jacobian_fn = self._jacobian_no_reg
        elif reg_config.regularization_type == 'ridge':
            residuals_fn = lambda x: self._residuals_ridge(x, ridge_prior, reg_config)
            jacobian_fn = lambda x: self._jacobian_ridge(x, reg_config)
        elif reg_config.regularization_type == 'fisher':
            residuals_fn = lambda x: self._residuals_fisher(x, fisher_prior, reg_config)
            jacobian_fn = lambda x: self._jacobian_fisher(x, fisher_prior, reg_config)
        elif reg_config.regularization_type == 'combined':
            residuals_fn = lambda x: self._residuals_combined(x, ridge_prior, fisher_prior, reg_config)
            jacobian_fn = lambda x: self._jacobian_combined(x, ridge_prior, fisher_prior, reg_config)
        else:
            raise ValueError(f"Unknown regularization type: {reg_config.regularization_type}")
        
        # Optimize
        result: OptimizeResult = least_squares(
            residuals_fn,
            x0=x0,
            jac=jacobian_fn,  # type: ignore
            bounds=(self.lower_bounds, self.upper_bounds),
            verbose=verbose,
            max_nfev=max_nfev,
        )
        
        # Check convergence and Feller condition
        if verbose > 0:
            self._report_calibration(result, reg_config)
        
        return HestonModel.from_array(result.x)
    
    def _get_initial_guess(self, initial_guess: Optional[HestonModel]) -> np.ndarray:
        """Generate initial parameter guess"""
        if initial_guess is not None:
            x0 = initial_guess.to_array()
        else:
            # Use ATM short-term IV for v0 (squared for variance)
            try:
                iv = float(self.market_data.initial_volatility())
                v0 = iv ** 2  # Convert volatility to variance
                v_bar = v0    # Use same for long-term variance
            except:
                v0 = 0.04  # Fallback variance
                v_bar = 0.04
            
            # Initial guess: [v0, v_bar, rho, kappa, sigma]
            x0 = np.array([v0, v_bar, -0.2, 1.0, 0.2])
        return x0
    
    # ============ No Regularization ============
    
    def _residuals_no_reg(self, x: np.ndarray) -> np.ndarray:
        """Just market fit residuals"""
        model = HestonModel.from_array(x)
        model_prices = model._heston_price_call(
            self.market_data.spot_price,  # Array
            self.market_data.risk_free_rate,
            self.market_data.strikes,
            self.market_data.maturities
        )
        return model_prices - self.market_data.market_prices
    
    def _jacobian_no_reg(self, x: np.ndarray) -> np.ndarray:
        """Just market fit Jacobian"""
        model = HestonModel.from_array(x)
        jac = model._gradient(
            self.market_data.spot_price,  # Array
            self.market_data.risk_free_rate,
            self.market_data.strikes,
            self.market_data.maturities
        )
        # Ensure shape is (n_obs, 5)
        if jac.shape[0] == 5:
            jac = jac.T
        return jac
    
    # ============ Ridge Regularization ============
    
    def _residuals_ridge(self, x: np.ndarray, prior: np.ndarray, 
                        config: RegularizationConfig) -> np.ndarray:
        """Residuals with Ridge/Tikhonov regularization"""
        # Market fit
        market_residuals = self._residuals_no_reg(x)
        
        # Ridge regularization: ||W(x - x0)||²
        if config.ridge_param_weights is None:
            weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])  # Default weights
        else:
            weights = config.ridge_param_weights
        
        ridge_residuals = np.sqrt(config.ridge_weight) * weights * (x - prior)
        
        # Feller penalty
        feller_residual = self._feller_residual(x, config.feller_weight)
        
        return np.concatenate([market_residuals, ridge_residuals, [feller_residual]])
    
    def _jacobian_ridge(self, x: np.ndarray, 
                       config: RegularizationConfig) -> np.ndarray:
        """Jacobian with Ridge regularization"""
        # Market fit Jacobian
        market_jac = self._jacobian_no_reg(x)
        
        # Ridge Jacobian: diagonal matrix
        if config.ridge_param_weights is None:
            weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
        else:
            weights = config.ridge_param_weights
        
        ridge_jac = np.sqrt(config.ridge_weight) * np.diag(weights)
        
        # Feller Jacobian
        feller_jac = self._feller_jacobian(x, config.feller_weight)
        
        return np.vstack([market_jac, ridge_jac, feller_jac.reshape(1, -1)])
    
    # ============ Fisher Regularization ============
    
    def _residuals_fisher(self, x: np.ndarray, prior: np.ndarray,
                         config: RegularizationConfig) -> np.ndarray:
        """Residuals with Fisher information regularization"""
        # Market fit
        market_residuals = self._residuals_no_reg(x)
        
        # Fisher-weighted distance: ||x - x0||_G
        fisher_metric = self._get_fisher_metric()
        G = fisher_metric.metric_tensor(x)
        
        diff = x - prior
        # Cholesky decomposition: G = L L^T, so ||diff||_G = ||L diff||_2
        try:
            L = np.linalg.cholesky(G)
            fisher_residuals = np.sqrt(config.fisher_weight) * (L @ diff)
        except np.linalg.LinAlgError:
            # If not positive definite, use sqrt of eigenvalues
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals = np.maximum(eigvals, 1e-8)  # Ensure positivity
            sqrt_G = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            fisher_residuals = np.sqrt(config.fisher_weight) * (sqrt_G @ diff)
        
        # Feller penalty
        feller_residual = self._feller_residual(x, config.feller_weight)
        
        return np.concatenate([market_residuals, fisher_residuals, [feller_residual]])
    
    def _jacobian_fisher(self, x: np.ndarray, prior: np.ndarray,
                        config: RegularizationConfig) -> np.ndarray:
        """Jacobian with Fisher regularization"""
        # Market fit
        market_jac = self._jacobian_no_reg(x)
        
        # Fisher regularization Jacobian
        # For simplicity, use first-order approximation: ∇||x-x0||²_G ≈ G(x-x0)
        fisher_metric = self._get_fisher_metric()
        G = fisher_metric.metric_tensor(x)
        
        try:
            L = np.linalg.cholesky(G)
            fisher_jac = np.sqrt(config.fisher_weight) * L
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals = np.maximum(eigvals, 1e-8)
            sqrt_G = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            fisher_jac = np.sqrt(config.fisher_weight) * sqrt_G
        
        # Feller Jacobian
        feller_jac = self._feller_jacobian(x, config.feller_weight)
        
        return np.vstack([market_jac, fisher_jac, feller_jac.reshape(1, -1)])
    
    # ============ Combined Regularization ============
    
    def _residuals_combined(self, x: np.ndarray, ridge_prior: np.ndarray,
                           fisher_prior: np.ndarray, 
                           config: RegularizationConfig) -> np.ndarray:
        """Residuals with both Ridge and Fisher regularization"""
        # Market fit
        market_residuals = self._residuals_no_reg(x)
        
        # Ridge component
        if config.ridge_param_weights is None:
            weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
        else:
            weights = config.ridge_param_weights
        ridge_residuals = np.sqrt(config.ridge_weight) * weights * (x - ridge_prior)
        
        # Fisher component
        fisher_metric = self._get_fisher_metric()
        G = fisher_metric.metric_tensor(x)
        diff = x - fisher_prior
        
        try:
            L = np.linalg.cholesky(G)
            fisher_residuals = np.sqrt(config.fisher_weight) * (L @ diff)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals = np.maximum(eigvals, 1e-8)
            sqrt_G = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            fisher_residuals = np.sqrt(config.fisher_weight) * (sqrt_G @ diff)
        
        # Feller penalty
        feller_residual = self._feller_residual(x, config.feller_weight)
        
        return np.concatenate([
            market_residuals, 
            ridge_residuals, 
            fisher_residuals, 
            [feller_residual]
        ])
    
    def _jacobian_combined(self, x: np.ndarray, ridge_prior: np.ndarray,
                          fisher_prior: np.ndarray,
                          config: RegularizationConfig) -> np.ndarray:
        """Jacobian with combined regularization"""
        market_jac = self._jacobian_no_reg(x)
        
        # Ridge Jacobian
        if config.ridge_param_weights is None:
            weights = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
        else:
            weights = config.ridge_param_weights
        ridge_jac = np.sqrt(config.ridge_weight) * np.diag(weights)
        
        # Fisher Jacobian
        fisher_metric = self._get_fisher_metric()
        G = fisher_metric.metric_tensor(x)
        
        try:
            L = np.linalg.cholesky(G)
            fisher_jac = np.sqrt(config.fisher_weight) * L
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals = np.maximum(eigvals, 1e-8)
            sqrt_G = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            fisher_jac = np.sqrt(config.fisher_weight) * sqrt_G
        
        # Feller Jacobian
        feller_jac = self._feller_jacobian(x, config.feller_weight)
        
        return np.vstack([market_jac, ridge_jac, fisher_jac, feller_jac.reshape(1, -1)])
    
    # ============ Feller Penalty Helpers ============
    
    def _feller_residual(self, x: np.ndarray, weight: float) -> float:
        """Feller condition penalty residual"""
        if weight == 0:
            return 0.0
        
        v_bar, kappa, sigma = x[1], x[3], x[4]
        feller_violation = max(0, sigma**2 - 2 * kappa * v_bar)
        n_obs = len(self.market_data.market_prices)
        
        return np.sqrt(weight * n_obs) * feller_violation
    
    def _feller_jacobian(self, x: np.ndarray, weight: float) -> np.ndarray:
        """Feller condition penalty Jacobian"""
        if weight == 0:
            return np.zeros(5)
        
        v_bar, kappa, sigma = x[1], x[3], x[4]
        feller_violation = sigma**2 - 2 * kappa * v_bar
        n_obs = len(self.market_data.market_prices)
        
        if feller_violation > 0:
            return np.sqrt(weight * n_obs) * np.array([
                0,           # ∂/∂v0
                -2 * kappa,  # ∂/∂v_bar
                0,           # ∂/∂rho
                -2 * v_bar,  # ∂/∂kappa
                2 * sigma,   # ∂/∂sigma
            ])
        else:
            return np.zeros(5)
    
    # ============ Reporting ============
    
    def _report_calibration(self, result: OptimizeResult, config: RegularizationConfig) -> None:
        """Report calibration results"""
        params = result.x
        print(f"\n{'='*60}")
        print("Calibration Results")
        print(f"{'='*60}")
        print(f"Optimization status: {'Success' if result.success else 'Failed'}")
        print(f"Final cost: {result.cost:.6f}")
        print(f"Optimality: {np.linalg.norm(result.grad):.2e}")
        
        print(f"\nParameters:")
        param_names = ['v0', 'v̄', 'ρ', 'κ', 'σ']
        for name, val in zip(param_names, params):
            print(f"  {name:4s} = {val:.6f}")
        
        # Feller condition check
        v_bar, kappa, sigma = params[1], params[3], params[4]
        feller_lhs = 2 * kappa * v_bar
        feller_rhs = sigma**2
        feller_satisfied = feller_lhs > feller_rhs
        
        print(f"\nFeller Condition: 2κv̄ > σ²")
        print(f"  LHS = {feller_lhs:.6f}")
        print(f"  RHS = {feller_rhs:.6f}")
        print(f"  Status: {'✓ Satisfied' if feller_satisfied else '✗ Violated'}")
        
        # Regularization info
        if config.regularization_type != 'none':
            print(f"\nRegularization: {config.regularization_type}")
            if config.ridge_weight > 0:
                print(f"  Ridge weight: {config.ridge_weight}")
            if config.fisher_weight > 0:
                print(f"  Fisher weight: {config.fisher_weight}")
            if config.feller_weight > 0:
                print(f"  Feller weight: {config.feller_weight}")
        
        print(f"{'='*60}\n")

