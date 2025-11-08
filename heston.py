import numpy as np
from scipy.integrate import quad_vec
from pydantic import BaseModel,  ConfigDict, model_validator
import warnings




#This class contains any computations required without any data
class HestonModel(BaseModel):
    v0: float  # Initial variance
    v_bar: float  # Long-term variance 
    rho: float  # Correlation between S0 and variance
    kappa: float  # Mean reversion speed 
    sigma: float  # Volatility of volatility 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.v_bar, self.rho, self.kappa, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HestonModel":
        """Create from array: [v0, v̄, ρ, κ, σ]"""
        return cls(v0=arr[0], v_bar=arr[1], rho=arr[2], kappa=arr[3], sigma=arr[4])
    
    @model_validator(mode='after')
    def Feller(self) ->'HestonModel':
        feller:bool = 2 * self.kappa * self.v_bar > self.sigma**2
        
        if   not (feller
            and self.v0 > 0
            and self.kappa > 0
            and self.v_bar > 0
            and self.sigma > 0
            and -1 <= self.rho <= 1
        ):
            warnings.warn('Feller condition not satisfied')
        return self
            

    # Helper functions following paper notation exactly
    def __xi(self, u: complex) -> complex:
        return self.kappa - self.sigma * self.rho * 1j * u

    def __d(self, u: complex) -> complex:
        """d := sqrt(ξ² + σ²(u² + iu)) (Equation 11b)"""
        xi_val = self.__xi(u)
        return np.sqrt(xi_val**2 + self.sigma**2 * (u**2 + 1j * u))

    def __A1(self, u: complex, t: np.ndarray) -> np.ndarray:
        """A₁ := (u² + iu)sinh(dt/2) (Equation 15b)"""
        return (u**2 + 1j * u) * np.sinh(self.__d(u) * t / 2)

    def __A2(self, u: complex, t: np.ndarray) -> np.ndarray:
        """A₂ := (d/v₀)cosh(dt/2) + (ξ/v₀)sinh(dt/2) (Equation 15c)"""
        d_val = self.__d(u)
        xi_val = self.__xi(u)
        return (d_val) * np.cosh(d_val * t / 2) + (xi_val) * np.sinh(d_val * t / 2)

    def __A(self, u: complex, t: np.ndarray):
        return self.__A1(u, t) / self.__A2(u, t)

    def __B(self, u: complex, t: np.ndarray):
        """B := de^(κt/2)/(v₀A₂) (Equation 15d)"""
        d_val = self.__d(u)
        A2_val = self.__A2(u, t)
        return d_val * np.exp(self.kappa * t / 2) / (A2_val)

    def __D(self, u: complex, t: np.ndarray):
        """D := dt/2 + log((d+ξ)/(2v₀) + (d-ξ)/(2v₀)e^(-dt)) (Equation 17b)"""
        d_val = self.__d(u)
        xi_val = self.__xi(u)
        return (
            np.log(d_val)
            + (self.kappa - d_val) * t / 2
            - np.log((d_val + xi_val) / 2 + (d_val - xi_val) / 2 * np.exp(-d_val * t))
        )

    def __h_vector(self, u: complex, t: np.ndarray):
        i = 1j
        xi_val = self.__xi(u)
        d_val = self.__d(u)
        A_val = self.__A(u, t)
        A1_val = self.__A1(u, t)
        A2_val = self.__A2(u, t)
        B_val = self.__B(u, t)
        D_val = self.__D(u, t)
        # partial derivatives wrt rho
        d_drho = -xi_val * self.sigma * i * u / d_val
        A1_drho = (
            -i
            * u
            * (u**2 + i * u)
            * t
            * xi_val
            * self.sigma
            / (2 * d_val)
            * np.cosh(d_val * t / 2)
        )
        A2_drho = -(
            self.sigma
            * i
            * u
            * (2 + t * xi_val)
            / (2 * d_val)
            * (xi_val * np.cosh(d_val * t / 2) + d_val * np.sinh(d_val * t / 2))
        )
        A_drho = 1 / A2_val * A1_drho - A_val / A2_val * A2_drho
        B_drho = np.exp(self.kappa * t / 2) * (
            1 / A2_val * d_drho - d_val / (A2_val**2) * A2_drho
        )

        # partial derivatives wrt kappa
        B_dkappa = i / (self.sigma * u) * B_drho + t * B_val / 2

        # partial derivatives wrt sigma
        d_dsigma = (
            self.rho / self.sigma - 1 / xi_val
        ) * d_drho + self.sigma * u**2 / d_val

        A1_dsigma = (u**2 + i * u) * t / 2 * d_dsigma * np.cosh(d_val * t / 2)

        A2_dsigma = (
            self.rho / self.sigma * A2_drho
            - (2 + t * xi_val) / (i * u * t * xi_val) * A1_drho
            + self.sigma * t * A1_val / 2
        )
        A_dsigma = A1_dsigma / A2_val - A_val * A2_dsigma / A2_val

        # vector components
        h1 = -A_val
        h2 = (
            2 * self.kappa / self.sigma**2 * D_val
            - t * self.kappa * self.rho * i * u / self.sigma
        )
        h3 = (
            -self.v0 * A_drho
            + 2
            * self.kappa
            * self.v_bar
            / (self.sigma**2 * d_val)
            * (d_drho - d_val / A2_val * A2_drho)
            - t * self.kappa * self.v_bar * i * u / self.sigma
        )
        h4 = (
            self.v0 / (self.sigma * i * u) * A_drho
            + 2 * self.v_bar / (self.sigma**2) * (D_val)
            + 2 * self.kappa * self.v_bar / (self.sigma**2 * B_val) * B_dkappa
            - t * self.v_bar * self.rho * i * u / self.sigma
        )
        h5 = (
            -self.v0*(A_dsigma)
            - 4
            * self.kappa
            * self.v_bar
            / (self.sigma**3) * D_val+2*self.kappa*self.v_bar/(self.sigma**2 * d_val)*(d_dsigma-d_val/A2_val*A2_dsigma)+t*self.kappa*self.v_bar*self.rho*i*u/(self.sigma**2))
        return np.vstack([h1, h2, h3, h4, h5])
        




    def __characteristic_function(self,
        S0: float, r: float, u: complex, t: np.ndarray
    ):
        
        i = 1j

        return np.exp(
            i * u * np.log(S0)
            + i * u * r * t
            - t * self.kappa * self.v_bar * self.rho * i * u / self.sigma
            - self.v0 * (self.__A(u, t))
            + 2 * self.kappa * self.v_bar / self.sigma**2 * self.__D(u, t)
        )


    def _heston_price_call(self,
        S0: float, r: float, K: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
    
        i = 1j

        def integrand1(u: complex):
            return np.real(
                self.__characteristic_function(S0, r, u - i, T)
                * np.exp(-i * u * np.log(K))
                /(i* u)
            )

        def integrand2(u: complex):
            return np.real(
                self.__characteristic_function(S0, r, u, T)
                * np.exp(-i * u * np.log(K))
                / (i* u)
            )

        integral1 = quad_vec(integrand1, 0, 100, limit=100)[0]
        integral2 = quad_vec(integrand2, 0, 100, limit=100)[0]

        return( .5*(S0-np.exp(-r*T)*K)
            + np.exp(-r * T) / np.pi * (integral1
            - K * integral2))
        



    def _gradient(self,
        S0: float, r: float, K: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        i = 1j

        def integrand1(u: complex):
            return np.real(
                self.__characteristic_function(S0, r, u - i, T)
                * self.__h_vector(u-i, T)
                * np.exp(np.log(K) * -i * u)
                / (i * u)
            )

        def integrand2(u: complex):
            return np.real(
                self.__characteristic_function(S0, r, u, T)
                * self.__h_vector(u, T)
                * np.exp(np.log(K) * -i * u)
                / (i * u)
            )

        integral1 = quad_vec(integrand1, 0, 100, limit=100)[0]
        integral2 = quad_vec(integrand2, 0, 100, limit=100)[0]
        return (
            np.exp(-r * T) / np.pi * (integral1- K * integral2)
        )
    def _square_error(self,
        S0: float, r: float, strikes: np.ndarray, maturities: np.ndarray, market_prices: np.ndarray):
        predictions = self._heston_price_call(S0, r, strikes, maturities)
        residuals = predictions - market_prices
        return 0.5 * np.sum(residuals**2)
    
    def _grad_loss(self,
        S0: float, r: float, strikes: np.ndarray, maturities: np.ndarray, market_prices: np.ndarray
    ):
        predictions = self._heston_price_call(S0, r, strikes, maturities)
        residuals = predictions - market_prices
        jacobian = self._gradient(S0, r, strikes, maturities)

        return jacobian @ residuals



 
    



