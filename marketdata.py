
from pydantic import BaseModel, model_validator, ConfigDict
import numpy as np
import yfinance as yf
from blackscholes import __implied_volatility as bs_iv


class MarketData(BaseModel):
    spot_price: np.ndarray  # S0 - now an array instead of float
    risk_free_rate: float  # r
    strikes: np.ndarray  # K
    maturities: np.ndarray  # T
    market_prices: np.ndarray  # C^*
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """Market data for calibration"""
    
    @model_validator(mode='after')
    def __post_init__(self):
        """Validate arrays have consistent shapes"""
        if not (len(self.spot_price)==len(self.strikes) == len(self.maturities) == len(self.market_prices)):
            raise ValueError("Spot prices, strikes, maturities, and prices must have same length")
        return self

    @classmethod
    def from_yahoo_finance(
        cls,
        ticker: str,
        risk_free_rate: float = 0.04,
        option_type: str = "call",
        min_volume: int = 10,
    ):
        stock: yf.Ticker = yf.Ticker(ticker)
        spot_price = stock.fast_info["lastPrice"]

        # Get all available expiration dates
        expirations = stock.options

        strikes_list = []
        maturities_list = []
        prices_list = []
        spot_prices_list = []

        # Current date for time-to-maturity calculation
        from datetime import datetime

        current_date = datetime.now()

        # Iterate through expiration dates
        for exp_date in expirations:
            # Get option chain for this expiration
            opt_chain = stock.option_chain(exp_date)

            # Select calls or puts
            if option_type.lower() == "call":
                options = opt_chain.calls
            else:
                options = opt_chain.puts

            # Filter by volume and price
            options = options[options["volume"] >= min_volume]
            options = options[options["lastPrice"] > 0]
            
            # Drop rows with any missing values
            options = options.dropna()

            if len(options) == 0:
                continue

            # Calculate time to maturity in years
            exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
            days_to_expiry = (exp_datetime - current_date).days
            time_to_maturity = days_to_expiry / 365.0

            if time_to_maturity <= 0.05:
                continue

            # Extract strikes and prices
            for row in options.itertuples(index=False):
                strikes_list.append(row.strike)
                maturities_list.append(time_to_maturity)
                # Use mid price (average of bid and ask)
                mid_price = (row.bid + row.ask) / 2
                prices_list.append(mid_price)
                spot_prices_list.append(spot_price)

        # Convert to numpy arrays
        strikes = np.array(strikes_list)
        maturities = np.array(maturities_list)
        market_prices = np.array(prices_list)
        spot_prices = np.array(spot_prices_list)

        return cls(
            spot_price=spot_prices,
            risk_free_rate=risk_free_rate,
            strikes=strikes,
            maturities=maturities,
            market_prices=market_prices,
        )
        
    def testing(self, train_ratio: float = 0.75):
        # Stack arrays to shuffle together
        num_options = len(self.strikes)
        
        # Generate shuffled indices
        shuffled_indices = np.random.permutation(num_options)
        
        # Split indices
        split = int(num_options * train_ratio)
        train_indices = shuffled_indices[:split]
        test_indices = shuffled_indices[split:]
        
        # Create training data
        training_data = MarketData(
            spot_price=self.spot_price[train_indices],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[train_indices],
            maturities=self.maturities[train_indices],
            market_prices=self.market_prices[train_indices]
        )
        
        # Create testing data
        testing_data = MarketData(
            spot_price=self.spot_price[test_indices],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[test_indices],
            maturities=self.maturities[test_indices],
            market_prices=self.market_prices[test_indices]
        )
        
        return training_data, testing_data
    
    def atm(self, tolerance=.05, max_iter=100):
        # Generate shuffled indices
        S0 = self.spot_price
        r = self.risk_free_rate
        forward_prices = S0 * np.exp(r * self.maturities)
    
        # Compute moneyness: |F - K|
        moneyness = np.abs(forward_prices - self.strikes)
        
        # Use mean spot price for tolerance calculation
        mean_spot = np.mean(S0)
        
        # Create mask for ATM options
        atm_mask = moneyness < tolerance * mean_spot
        while not atm_mask.any() and max_iter:
            tolerance += .01
            atm_mask = moneyness < tolerance * mean_spot
            max_iter -= 1
    
        # Stack back into shape (3, n)
        atm_options = MarketData(
            spot_price=self.spot_price[atm_mask],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[atm_mask],
            maturities=self.maturities[atm_mask],
            market_prices=self.market_prices[atm_mask]
        )
    
        return atm_options
    
    def short_term(self, tolerance=.01, max_iter=100):
        short_mask = self.maturities < tolerance
        # make sure there is at least one
        while not short_mask.any() and max_iter:
            tolerance += .01
            short_mask = self.maturities < tolerance
            max_iter -= 1
    
        atm_options = MarketData(
            spot_price=self.spot_price[short_mask],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[short_mask],
            maturities=self.maturities[short_mask],
            market_prices=self.market_prices[short_mask]
        )
    
        return atm_options
    
    def remove_outliers(self, strike_tol: float = 0.05, maturity_tol: float = 0.02, price_threshold: float = 3.0):
        keep_mask = np.ones(len(self.strikes), dtype=bool)
        
        for i in range(len(self.strikes)):
            strike = self.strikes[i]
            maturity = self.maturities[i]
            price = self.market_prices[i]
            
            strike_range = strike * strike_tol
            nearby_mask = (
                (np.abs(self.strikes - strike) <= strike_range) &
                (np.abs(self.maturities - maturity) <= maturity_tol)
            )
            
            nearby_prices = self.market_prices[nearby_mask]
            
            if len(nearby_prices) < 3:
                continue
            
            # Check if this price is a severe outlier compared to neighbors
            median = np.median(nearby_prices)
            mad = np.median(np.abs(nearby_prices - median))
            
            if mad > 0:
                modified_z_score = 0.6745 * np.abs(price - median) / mad
                if modified_z_score > price_threshold:
                    keep_mask[i] = False
        
        # Filter out zero or near-zero prices
        non_zero_price = self.market_prices > 0.01
        
        # Filter out zero or near-zero maturities
        non_zero_maturity = self.maturities > 0.0
        
        # Combine all filters
        keep_mask = keep_mask & non_zero_price & non_zero_maturity
        
        # Return new MarketData with filtered data
        return MarketData(
            spot_price=self.spot_price[keep_mask],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[keep_mask],
            maturities=self.maturities[keep_mask],
            market_prices=self.market_prices[keep_mask]
        )
        
    def filter_deep_itm(self, threshold: float = 0.5):
        # Calculate forward prices for each maturity
        forward_prices = self.spot_price * np.exp(self.risk_free_rate * self.maturities)
        
        # Filter out deep ITM options
        not_deep_itm_mask = self.market_prices < (forward_prices * threshold)
        
        return MarketData(
            spot_price=self.spot_price[not_deep_itm_mask],
            risk_free_rate=self.risk_free_rate,
            strikes=self.strikes[not_deep_itm_mask],
            maturities=self.maturities[not_deep_itm_mask],
            market_prices=self.market_prices[not_deep_itm_mask]
        )
    

    def initial_volatility(self):
        temp = self.atm().short_term()
        volatilities = bs_iv(
            temp.market_prices, temp.spot_price, temp.strikes, temp.maturities, temp.risk_free_rate
        )
        return np.mean(volatilities)


    def implied_volatility(self):
        return bs_iv(
            self.market_prices, self.spot_price, self.strikes, self.maturities, self.risk_free_rate
        )
    
    def validate_option_data(
        self,
        tolerance=0.01,
        check_monotonicity=False,
        check_iv=False,
        iv_multiplier=3.0,
        return_outliers=False,
    ):
        S0 = self.spot_price
        K = self.strikes
        T = self.maturities
        r = self.risk_free_rate
        C = self.market_prices

        valid_mask = np.ones(len(C), dtype=bool)

        arbitrage_lower = 0
        arbitrage_upper = 0
        monotonicity_violations = 0
        iv_violations = 0

        # 1. No-arbitrage lower bound: C >= max(S0 - K*exp(-rT), 0)
        lower_bound = np.maximum(S0 - K * np.exp(-r * T), 0)
        lower_violations = C < lower_bound - tolerance
        valid_mask &= ~lower_violations
        arbitrage_lower = lower_violations.sum()

        # 2. No-arbitrage upper bound: C <= S0
        upper_violations = C > S0 + tolerance
        valid_mask &= ~upper_violations
        arbitrage_upper = upper_violations.sum()

        # 3. Monotonicity in strike: calls should decrease with strike (for same maturity)
        if check_monotonicity:
            for t in np.unique(T):
                t_mask = T == t
                t_indices = np.where(t_mask)[0]
                t_strikes = K[t_mask]
                t_prices = C[t_mask]

                # Sort by strike
                sorted_idx = np.argsort(t_strikes)
                sorted_prices = t_prices[sorted_idx]

                # Check monotonicity: price should decrease (or stay flat within tolerance)
                for i in range(len(sorted_prices) - 1):
                    if sorted_prices[i + 1] > sorted_prices[i] + tolerance:  # Violation
                        # Flag only the option with higher price (likely the error)
                        actual_idx_next = t_indices[sorted_idx[i + 1]]
                        if valid_mask[actual_idx_next]:  # Only count if not already flagged
                            monotonicity_violations += 1
                        valid_mask[actual_idx_next] = False

        # Reasonable IV check
        if check_iv:
            ivs = bs_iv(C, S0, K, T, r)

            for t in np.unique(T):
                t_mask = (T == t) & valid_mask
                if not np.any(t_mask):
                    continue

                t_strikes = K[t_mask]
                t_ivs = ivs[t_mask]
                t_s0 = S0[t_mask]

                # Find ATM - use mean spot price for this maturity
                mean_s0 = np.mean(t_s0)
                F = mean_s0 * np.exp(r * t)
                atm_idx = np.argmin(np.abs(t_strikes - F))
                atm_iv = t_ivs[atm_idx]

                if np.isnan(atm_iv):
                    continue

                # get extreme IV relative to ATM
                t_indices = np.where(t_mask)[0]
                iv_too_high = t_ivs > iv_multiplier * atm_iv
                iv_too_low = t_ivs < atm_iv / iv_multiplier
                iv_is_nan = np.isnan(t_ivs)

                extreme_iv_mask = iv_too_high | iv_too_low | iv_is_nan

                # Count violations (only those currently valid)
                iv_violations += np.sum(extreme_iv_mask & valid_mask[t_indices])

                # Update valid_mask
                valid_mask[t_indices] = valid_mask[t_indices] & ~extreme_iv_mask

        total_removed = (~valid_mask).sum()
        total = len(C)
        print(f"Removed {total_removed} / {total} options ({100 * total_removed / total:.1f}%)")
        print(f"  - Lower bound violations: {arbitrage_lower}")
        print(f"  - Upper bound violations: {arbitrage_upper}")
        if check_monotonicity:
            print(f"  - Monotonicity violations: {monotonicity_violations}")
        if check_iv:
            print(f"  - IV outliers/unsolvable: {iv_violations}")
            
        if not return_outliers:
            return MarketData(
                spot_price=S0[valid_mask],
                risk_free_rate=r,
                strikes=K[valid_mask],
                maturities=T[valid_mask],
                market_prices=C[valid_mask],
            )
        else:
            removed = ~valid_mask
            return MarketData(
                spot_price=S0[valid_mask],
                risk_free_rate=r,
                strikes=K[valid_mask],
                maturities=T[valid_mask],
                market_prices=C[valid_mask],
            ), MarketData(
                spot_price=S0[removed],
                risk_free_rate=r,
                strikes=K[removed],
                maturities=T[removed],
                market_prices=C[removed],
            )