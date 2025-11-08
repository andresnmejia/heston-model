
import numpy as np
from dataclasses import dataclass
from enum import Enum
from scipy.integrate import quad_vec
from pydantic import BaseModel, model_validator
from marketdata import MarketData

class HestonParameters(BaseModel):
    v0: float  # Initial variance
    v_bar: float  # Long-term variance (v̄ in paper)
    rho: float  # Correlation between asset and variance
    kappa: float  # Mean reversion speed (κ in paper)
    sigma: float  # Volatility of volatility (σ in paper)

    def to_array(self) -> np.ndarray:
        """Convert to array: [v0, v̄, ρ, κ, σ]"""
        return np.array([self.v0, self.v_bar, self.rho, self.kappa, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HestonParameters":
        """Create from array: [v0, v̄, ρ, κ, σ]"""
        return cls(v0=arr[0], v_bar=arr[1], rho=arr[2], kappa=arr[3], sigma=arr[4])
    
    @model_validator(mode='before')
    @classmethod
    def Feller(self,data):
        """Check Feller condition and parameter constraints"""
        feller:bool = 2 * self.kappa * self.v_bar > self.sigma**2
        
        if   (feller
            and self.v0 > 0
            and self.kappa > 0
            and self.v_bar > 0
            and self.sigma > 0
            and -1 <= self.rho <= 1
        ):
            return data

    # Helper functions following paper notation exactly
    def xi(self, u: complex) -> complex:
        """ξ := κ - σρiu (Equation 11a)"""
        return self.kappa - self.sigma * self.rho * 1j * u

    def d(self, u: complex) -> complex:
        """d := sqrt(ξ² + σ²(u² + iu)) (Equation 11b)"""
        xi_val = self.xi(u)
        return np.sqrt(xi_val**2 + self.sigma**2 * (u**2 + 1j * u))

    def A1(self, u: complex, t: np.ndarray) -> np.ndarray:
        """A₁ := (u² + iu)sinh(dt/2) (Equation 15b)"""
        return (u**2 + 1j * u) * np.sinh(self.d(u) * t / 2)

    def A2(self, u: complex, t: np.ndarray) -> np.ndarray:
        """A₂ := (d/v₀)cosh(dt/2) + (ξ/v₀)sinh(dt/2) (Equation 15c)"""
        d_val = self.d(u)
        xi_val = self.xi(u)
        return (d_val) * np.cosh(d_val * t / 2) + (xi_val) * np.sinh(d_val * t / 2)

    def A(self, u: complex, t: np.ndarray):
        return self.A1(u, t) / self.A2(u, t)

    def B(self, u: complex, t: np.ndarray):
        """B := de^(κt/2)/(v₀A₂) (Equation 15d)"""
        d_val = self.d(u)
        A2_val = self.A2(u, t)
        return d_val * np.exp(self.kappa * t / 2) / (A2_val)

    def D(self, u: complex, t: np.ndarray):
        """D := dt/2 + log((d+ξ)/(2v₀) + (d-ξ)/(2v₀)e^(-dt)) (Equation 17b)"""
        d_val = self.d(u)
        xi_val = self.xi(u)
        return (
            np.log(d_val)
            + (self.kappa - d_val) * t / 2
            - np.log((d_val + xi_val) / 2 + (d_val - xi_val) / 2 * np.exp(-d_val * t))
        )

    def h_vector(self, u: complex, t: np.ndarray):
        i = 1j
        xi_val = self.xi(u)
        d_val = self.d(u)
        A_val = self.A(u, t)
        A1_val = self.A1(u, t)
        A2_val = self.A2(u, t)
        B_val = self.B(u, t)
        D_val = self.D(u, t)
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

IG= HestonParameters(v0=0.04, v_bar=0.04, rho=-0.5, kappa=2.0, sigma=0.3)
IG.h_vector(1,np.array([2,1]))


def characteristic_function(
    S0: float, r: float, u: complex, t: np.ndarray, theta: HestonParameters, md:MarketData=None
):
    i = 1j

    return np.exp(
        i * u * np.log(S0)
        + i * u * r * t
        - t * theta.kappa * theta.v_bar * theta.rho * i * u / theta.sigma
        - theta.v0 * (theta.A(u, t))
        + 2 * theta.kappa * theta.v_bar / theta.sigma**2 * theta.D(u, t)
    )


def heston_price_call(
    S0: float, r: float, K: np.ndarray, T: np.ndarray, theta: HestonParameters
) -> np.ndarray:
 
    i = 1j

    def integrand1(u: complex):
        return np.real(
            characteristic_function(S0, r, u - i, T, theta)
            * np.exp(-i * u * np.log(K))
            /(i* u)
        )

    def integrand2(u: complex):
        return np.real(
            characteristic_function(S0, r, u, T, theta)
            * np.exp(-i * u * np.log(K))
            / (i* u)
        )

    integral1 = quad_vec(integrand1, 0, 100, limit=100)[0]
    integral2 = quad_vec(integrand2, 0, 100, limit=100)[0]

    return( .5*(S0-np.exp(-r*T)*K)
        + np.exp(-r * T) / np.pi * (integral1
        - K * integral2))
    


def heston_gradient(
    S0: float, r: float, K: np.ndarray, T: np.ndarray, theta: HestonParameters
) -> np.ndarray:
    i = 1j

    def integrand1(u: complex):
        return np.real(
            characteristic_function(S0, r, u - i, T, theta)
            * theta.h_vector(u-i, T)
            * np.exp(np.log(K) * -i * u)
            / (i * u)
        )

    def integrand2(u: complex):
        return np.real(
            characteristic_function(S0, r, u, T, theta)
            * theta.h_vector(u, T)
            * np.exp(np.log(K) * -i * u)
            / (i * u)
        )

    integral1 = quad_vec(integrand1, 0, 100, limit=100)[0]
    integral2 = quad_vec(integrand2, 0, 100, limit=100)[0]
    return (
        np.exp(-r * T) / np.pi * (integral1- K * integral2)
    )

