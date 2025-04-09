"""
Specific EVE and EVS regression models for different portfolio segments.
This module implements concrete regression models with different sensitivities
to interest rate movements for various portfolio types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from .base_model import EVEModel, EVSModel


class RetailMortgageEVEModel(EVEModel):
    """EVE model for retail mortgage portfolio."""
    
    def __init__(self, name: str = "retail_mortgage_eve", 
                description: str = "EVE model for retail mortgage portfolio"):
        """Initialize the retail mortgage EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients based on typical retail mortgage behavior
        # These would normally be fitted from data, but we're using synthetic values
        self.intercept = 1000000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -50000.0,      # Strong negative sensitivity to overnight rate
            'treasury_1y': -75000.0,    # Strong negative sensitivity to 1Y rate
            'treasury_2y': -60000.0,    # Moderate negative sensitivity to 2Y rate
            'treasury_3y': -40000.0,    # Moderate negative sensitivity to 3Y rate
            'treasury_5y': -30000.0,    # Moderate negative sensitivity to 5Y rate
            'treasury_10y': -20000.0,   # Weaker negative sensitivity to 10Y rate
            'prepayment_speed': -5000.0, # Negative sensitivity to prepayment speed
            'credit_spread': -10000.0    # Negative sensitivity to credit spreads
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -50000.0,
            'treasury_1y': -75000.0,
            'treasury_2y': -60000.0,
            'treasury_3y': -40000.0,
            'treasury_5y': -30000.0,
            'treasury_10y': -20000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.85  # Synthetic R-squared value


class CommercialLoanEVEModel(EVEModel):
    """EVE model for commercial loan portfolio."""
    
    def __init__(self, name: str = "commercial_loan_eve", 
                description: str = "EVE model for commercial loan portfolio"):
        """Initialize the commercial loan EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for commercial loans
        self.intercept = 2500000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -100000.0,     # Very strong negative sensitivity to overnight rate
            'treasury_1y': -120000.0,   # Very strong negative sensitivity to 1Y rate
            'treasury_2y': -90000.0,    # Strong negative sensitivity to 2Y rate
            'treasury_3y': -70000.0,    # Strong negative sensitivity to 3Y rate
            'treasury_5y': -50000.0,    # Moderate negative sensitivity to 5Y rate
            'treasury_10y': -30000.0,   # Moderate negative sensitivity to 10Y rate
            'credit_spread': -150000.0,  # Very strong negative sensitivity to credit spreads
            'default_rate': -200000.0    # Very strong negative sensitivity to default rates
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -100000.0,
            'treasury_1y': -120000.0,
            'treasury_2y': -90000.0,
            'treasury_3y': -70000.0,
            'treasury_5y': -50000.0,
            'treasury_10y': -30000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.88  # Synthetic R-squared value


class FixedIncomeEVEModel(EVEModel):
    """EVE model for fixed income securities portfolio."""
    
    def __init__(self, name: str = "fixed_income_eve", 
                description: str = "EVE model for fixed income securities portfolio"):
        """Initialize the fixed income EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for fixed income securities
        self.intercept = 5000000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -50000.0,      # Moderate negative sensitivity to overnight rate
            'treasury_1y': -150000.0,   # Strong negative sensitivity to 1Y rate
            'treasury_2y': -200000.0,   # Very strong negative sensitivity to 2Y rate
            'treasury_3y': -250000.0,   # Very strong negative sensitivity to 3Y rate
            'treasury_5y': -300000.0,   # Extremely strong negative sensitivity to 5Y rate
            'treasury_10y': -350000.0,  # Extremely strong negative sensitivity to 10Y rate
            'credit_spread': -200000.0,  # Strong negative sensitivity to credit spreads
            'liquidity_premium': -100000.0  # Strong negative sensitivity to liquidity premium
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -50000.0,
            'treasury_1y': -150000.0,
            'treasury_2y': -200000.0,
            'treasury_3y': -250000.0,
            'treasury_5y': -300000.0,
            'treasury_10y': -350000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.92  # Synthetic R-squared value


class DepositEVEModel(EVEModel):
    """EVE model for deposit portfolio."""
    
    def __init__(self, name: str = "deposit_eve", 
                description: str = "EVE model for deposit portfolio"):
        """Initialize the deposit EVE model."""
        super().__init__(name, description, model_type="log-linear")
        
        # Pre-defined coefficients for deposits
        self.intercept = np.log(3000000.0)  # Log of base portfolio value
        self.coefficients = {
            'sofr_rate': 0.5,           # Positive sensitivity to overnight rate (deposits become more valuable)
            'treasury_1y': 0.4,         # Positive sensitivity to 1Y rate
            'treasury_2y': 0.3,         # Positive sensitivity to 2Y rate
            'treasury_3y': 0.2,         # Positive sensitivity to 3Y rate
            'treasury_5y': 0.1,         # Weak positive sensitivity to 5Y rate
            'treasury_10y': 0.05,       # Very weak positive sensitivity to 10Y rate
            'deposit_beta': -0.8,       # Strong negative sensitivity to deposit beta
            'deposit_runoff': -1.2      # Very strong negative sensitivity to deposit runoff
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': 0.5,
            'treasury_1y': 0.4,
            'treasury_2y': 0.3,
            'treasury_3y': 0.2,
            'treasury_5y': 0.1,
            'treasury_10y': 0.05
        }
        
        self.is_fitted = True
        self.r_squared = 0.80  # Synthetic R-squared value


class CreditCardEVEModel(EVEModel):
    """EVE model for credit card portfolio."""
    
    def __init__(self, name: str = "credit_card_eve", 
                description: str = "EVE model for credit card portfolio"):
        """Initialize the credit card EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for credit cards
        self.intercept = 1500000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -20000.0,      # Weak negative sensitivity to overnight rate
            'treasury_1y': -25000.0,    # Weak negative sensitivity to 1Y rate
            'treasury_2y': -15000.0,    # Very weak negative sensitivity to 2Y rate
            'treasury_3y': -10000.0,    # Very weak negative sensitivity to 3Y rate
            'treasury_5y': -5000.0,     # Minimal negative sensitivity to 5Y rate
            'treasury_10y': -2000.0,    # Minimal negative sensitivity to 10Y rate
            'credit_spread': -50000.0,   # Moderate negative sensitivity to credit spreads
            'default_rate': -200000.0,   # Very strong negative sensitivity to default rates
            'utilization_rate': 100000.0 # Strong positive sensitivity to utilization rate
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -20000.0,
            'treasury_1y': -25000.0,
            'treasury_2y': -15000.0,
            'treasury_3y': -10000.0,
            'treasury_5y': -5000.0,
            'treasury_10y': -2000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.83  # Synthetic R-squared value


class AutoLoanEVEModel(EVEModel):
    """EVE model for auto loan portfolio."""
    
    def __init__(self, name: str = "auto_loan_eve", 
                description: str = "EVE model for auto loan portfolio"):
        """Initialize the auto loan EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for auto loans
        self.intercept = 800000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -30000.0,      # Moderate negative sensitivity to overnight rate
            'treasury_1y': -40000.0,    # Moderate negative sensitivity to 1Y rate
            'treasury_2y': -35000.0,    # Moderate negative sensitivity to 2Y rate
            'treasury_3y': -25000.0,    # Weak negative sensitivity to 3Y rate
            'treasury_5y': -15000.0,    # Weak negative sensitivity to 5Y rate
            'treasury_10y': -5000.0,    # Very weak negative sensitivity to 10Y rate
            'credit_spread': -30000.0,   # Moderate negative sensitivity to credit spreads
            'default_rate': -100000.0,   # Strong negative sensitivity to default rates
            'prepayment_speed': -20000.0 # Weak negative sensitivity to prepayment speed
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -30000.0,
            'treasury_1y': -40000.0,
            'treasury_2y': -35000.0,
            'treasury_3y': -25000.0,
            'treasury_5y': -15000.0,
            'treasury_10y': -5000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.81  # Synthetic R-squared value


class StudentLoanEVEModel(EVEModel):
    """EVE model for student loan portfolio."""
    
    def __init__(self, name: str = "student_loan_eve", 
                description: str = "EVE model for student loan portfolio"):
        """Initialize the student loan EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for student loans
        self.intercept = 1200000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -10000.0,      # Weak negative sensitivity to overnight rate
            'treasury_1y': -15000.0,    # Weak negative sensitivity to 1Y rate
            'treasury_2y': -20000.0,    # Weak negative sensitivity to 2Y rate
            'treasury_3y': -25000.0,    # Moderate negative sensitivity to 3Y rate
            'treasury_5y': -30000.0,    # Moderate negative sensitivity to 5Y rate
            'treasury_10y': -35000.0,   # Moderate negative sensitivity to 10Y rate
            'credit_spread': -20000.0,   # Weak negative sensitivity to credit spreads
            'default_rate': -150000.0,   # Strong negative sensitivity to default rates
            'unemployment_rate': -80000.0 # Strong negative sensitivity to unemployment rate
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -10000.0,
            'treasury_1y': -15000.0,
            'treasury_2y': -20000.0,
            'treasury_3y': -25000.0,
            'treasury_5y': -30000.0,
            'treasury_10y': -35000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.79  # Synthetic R-squared value


class CommercialRealEstateEVEModel(EVEModel):
    """EVE model for commercial real estate portfolio."""
    
    def __init__(self, name: str = "commercial_real_estate_eve", 
                description: str = "EVE model for commercial real estate portfolio"):
        """Initialize the commercial real estate EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for commercial real estate
        self.intercept = 4000000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -50000.0,      # Moderate negative sensitivity to overnight rate
            'treasury_1y': -75000.0,    # Strong negative sensitivity to 1Y rate
            'treasury_2y': -100000.0,   # Strong negative sensitivity to 2Y rate
            'treasury_3y': -125000.0,   # Very strong negative sensitivity to 3Y rate
            'treasury_5y': -150000.0,   # Very strong negative sensitivity to 5Y rate
            'treasury_10y': -175000.0,  # Very strong negative sensitivity to 10Y rate
            'credit_spread': -100000.0,  # Strong negative sensitivity to credit spreads
            'vacancy_rate': -200000.0,   # Very strong negative sensitivity to vacancy rates
            'cap_rate': -250000.0        # Extremely strong negative sensitivity to cap rates
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -50000.0,
            'treasury_1y': -75000.0,
            'treasury_2y': -100000.0,
            'treasury_3y': -125000.0,
            'treasury_5y': -150000.0,
            'treasury_10y': -175000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.90  # Synthetic R-squared value


class TreasuryPortfolioEVEModel(EVEModel):
    """EVE model for treasury portfolio."""
    
    def __init__(self, name: str = "treasury_portfolio_eve", 
                description: str = "EVE model for treasury portfolio"):
        """Initialize the treasury portfolio EVE model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for treasury portfolio
        self.intercept = 10000000.0  # Base portfolio value
        self.coefficients = {
            'sofr_rate': -100000.0,     # Strong negative sensitivity to overnight rate
            'treasury_1y': -300000.0,   # Very strong negative sensitivity to 1Y rate
            'treasury_2y': -400000.0,   # Extremely strong negative sensitivity to 2Y rate
            'treasury_3y': -500000.0,   # Extremely strong negative sensitivity to 3Y rate
            'treasury_5y': -600000.0,   # Extremely strong negative sensitivity to 5Y rate
            'treasury_10y': -700000.0,  # Extremely strong negative sensitivity to 10Y rate
            'yield_curve_slope': 200000.0, # Strong positive sensitivity to yield curve slope
            'liquidity_premium': -150000.0 # Strong negative sensitivity to liquidity premium
        }
        
        # Rate sensitivities
        self.rate_sensitivities = {
            'sofr_rate': -100000.0,
            'treasury_1y': -300000.0,
            'treasury_2y': -400000.0,
            'treasury_3y': -500000.0,
            'treasury_5y': -600000.0,
            'treasury_10y': -700000.0
        }
        
        self.is_fitted = True
        self.r_squared = 0.95  # Synthetic R-squared value


# EVS Models

class RetailMortgageEVSModel(EVSModel):
    """EVS model for retail mortgage portfolio."""
    
    def __init__(self, name: str = "retail_mortgage_evs", 
                description: str = "EVS model for retail mortgage portfolio"):
        """Initialize the retail mortgage EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.5,
            'treasury_1y': 0.8,
            'treasury_2y': 1.2,
            'treasury_3y': 1.5,
            'treasury_5y': 2.0,
            'treasury_10y': 2.5
        }
        
        self.convexities = {
            'sofr_rate': 0.05,
            'treasury_1y': 0.08,
            'treasury_2y': 0.12,
            'treasury_3y': 0.15,
            'treasury_5y': 0.20,
            'treasury_10y': 0.25
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'prepayment_speed': 0.3,
            'credit_spread': 0.2
        })
        
        self.intercept = -0.05  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.85  # Synthetic R-squared value


class CommercialLoanEVSModel(EVSModel):
    """EVS model for commercial loan portfolio."""
    
    def __init__(self, name: str = "commercial_loan_evs", 
                description: str = "EVS model for commercial loan portfolio"):
        """Initialize the commercial loan EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.3,
            'treasury_1y': 0.5,
            'treasury_2y': 0.8,
            'treasury_3y': 1.0,
            'treasury_5y': 1.2,
            'treasury_10y': 1.5
        }
        
        self.convexities = {
            'sofr_rate': 0.03,
            'treasury_1y': 0.05,
            'treasury_2y': 0.08,
            'treasury_3y': 0.10,
            'treasury_5y': 0.12,
            'treasury_10y': 0.15
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'credit_spread': 0.4,
            'default_rate': 0.6
        })
        
        self.intercept = -0.08  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.88  # Synthetic R-squared value


class FixedIncomeEVSModel(EVSModel):
    """EVS model for fixed income securities portfolio."""
    
    def __init__(self, name: str = "fixed_income_evs", 
                description: str = "EVS model for fixed income securities portfolio"):
        """Initialize the fixed income EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.2,
            'treasury_1y': 1.0,
            'treasury_2y': 2.0,
            'treasury_3y': 3.0,
            'treasury_5y': 5.0,
            'treasury_10y': 8.0
        }
        
        self.convexities = {
            'sofr_rate': 0.01,
            'treasury_1y': 0.10,
            'treasury_2y': 0.20,
            'treasury_3y': 0.30,
            'treasury_5y': 0.50,
            'treasury_10y': 0.80
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'credit_spread': 0.5,
            'liquidity_premium': 0.3
        })
        
        self.intercept = -0.10  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.92  # Synthetic R-squared value


class DepositEVSModel(EVSModel):
    """EVS model for deposit portfolio."""
    
    def __init__(self, name: str = "deposit_evs", 
                description: str = "EVS model for deposit portfolio"):
        """Initialize the deposit EVS model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for deposits
        self.intercept = 0.02  # Base sensitivity (positive)
        self.coefficients = {
            'sofr_rate': 0.05,          # Positive sensitivity to overnight rate
            'treasury_1y': 0.04,        # Positive sensitivity to 1Y rate
            'treasury_2y': 0.03,        # Positive sensitivity to 2Y rate
            'treasury_3y': 0.02,        # Positive sensitivity to 3Y rate
            'treasury_5y': 0.01,        # Weak positive sensitivity to 5Y rate
            'treasury_10y': 0.005,      # Very weak positive sensitivity to 10Y rate
            'deposit_beta': -0.08,      # Negative sensitivity to deposit beta
            'deposit_runoff': -0.12     # Negative sensitivity to deposit runoff
        }
        
        self.is_fitted = True
        self.r_squared = 0.80  # Synthetic R-squared value


class CreditCardEVSModel(EVSModel):
    """EVS model for credit card portfolio."""
    
    def __init__(self, name: str = "credit_card_evs", 
                description: str = "EVS model for credit card portfolio"):
        """Initialize the credit card EVS model."""
        super().__init__(name, description, model_type="linear")
        
        # Pre-defined coefficients for credit cards
        self.intercept = -0.01  # Base sensitivity
        self.coefficients = {
            'sofr_rate': -0.02,         # Weak negative sensitivity to overnight rate
            'treasury_1y': -0.025,      # Weak negative sensitivity to 1Y rate
            'treasury_2y': -0.015,      # Very weak negative sensitivity to 2Y rate
            'treasury_3y': -0.01,       # Very weak negative sensitivity to 3Y rate
            'treasury_5y': -0.005,      # Minimal negative sensitivity to 5Y rate
            'treasury_10y': -0.002,     # Minimal negative sensitivity to 10Y rate
            'credit_spread': -0.05,     # Moderate negative sensitivity to credit spreads
            'default_rate': -0.20,      # Strong negative sensitivity to default rates
            'utilization_rate': 0.10    # Positive sensitivity to utilization rate
        }
        
        self.is_fitted = True
        self.r_squared = 0.83  # Synthetic R-squared value


class AutoLoanEVSModel(EVSModel):
    """EVS model for auto loan portfolio."""
    
    def __init__(self, name: str = "auto_loan_evs", 
                description: str = "EVS model for auto loan portfolio"):
        """Initialize the auto loan EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.2,
            'treasury_1y': 0.4,
            'treasury_2y': 0.6,
            'treasury_3y': 0.8,
            'treasury_5y': 1.0,
            'treasury_10y': 1.2
        }
        
        self.convexities = {
            'sofr_rate': 0.02,
            'treasury_1y': 0.04,
            'treasury_2y': 0.06,
            'treasury_3y': 0.08,
            'treasury_5y': 0.10,
            'treasury_10y': 0.12
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'credit_spread': 0.3,
            'default_rate': 0.5,
            'prepayment_speed': 0.2
        })
        
        self.intercept = -0.03  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.81  # Synthetic R-squared value


class StudentLoanEVSModel(EVSModel):
    """EVS model for student loan portfolio."""
    
    def __init__(self, name: str = "student_loan_evs", 
                description: str = "EVS model for student loan portfolio"):
        """Initialize the student loan EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.1,
            'treasury_1y': 0.3,
            'treasury_2y': 0.5,
            'treasury_3y': 1.0,
            'treasury_5y': 2.0,
            'treasury_10y': 3.0
        }
        
        self.convexities = {
            'sofr_rate': 0.01,
            'treasury_1y': 0.03,
            'treasury_2y': 0.05,
            'treasury_3y': 0.10,
            'treasury_5y': 0.20,
            'treasury_10y': 0.30
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'credit_spread': 0.2,
            'default_rate': 0.4,
            'unemployment_rate': 0.3
        })
        
        self.intercept = -0.02  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.79  # Synthetic R-squared value


class CommercialRealEstateEVSModel(EVSModel):
    """EVS model for commercial real estate portfolio."""
    
    def __init__(self, name: str = "commercial_real_estate_evs", 
                description: str = "EVS model for commercial real estate portfolio"):
        """Initialize the commercial real estate EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.5,
            'treasury_1y': 1.0,
            'treasury_2y': 1.5,
            'treasury_3y': 2.0,
            'treasury_5y': 3.0,
            'treasury_10y': 4.0
        }
        
        self.convexities = {
            'sofr_rate': 0.05,
            'treasury_1y': 0.10,
            'treasury_2y': 0.15,
            'treasury_3y': 0.20,
            'treasury_5y': 0.30,
            'treasury_10y': 0.40
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'credit_spread': 0.4,
            'vacancy_rate': 0.6,
            'cap_rate': 0.7
        })
        
        self.intercept = -0.05  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.90  # Synthetic R-squared value


class TreasuryPortfolioEVSModel(EVSModel):
    """EVS model for treasury portfolio."""
    
    def __init__(self, name: str = "treasury_portfolio_evs", 
                description: str = "EVS model for treasury portfolio"):
        """Initialize the treasury portfolio EVS model."""
        super().__init__(name, description, model_type="linear", duration_based=True)
        
        # Pre-defined durations and convexities
        self.durations = {
            'sofr_rate': 0.1,
            'treasury_1y': 1.0,
            'treasury_2y': 2.0,
            'treasury_3y': 3.0,
            'treasury_5y': 5.0,
            'treasury_10y': 9.0
        }
        
        self.convexities = {
            'sofr_rate': 0.01,
            'treasury_1y': 0.10,
            'treasury_2y': 0.20,
            'treasury_3y': 0.30,
            'treasury_5y': 0.50,
            'treasury_10y': 0.90
        }
        
        # Coefficients based on durations
        self.coefficients = {rate: -duration for rate, duration in self.durations.items()}
        self.coefficients.update({
            'yield_curve_slope': 0.5,
            'liquidity_premium': 0.3
        })
        
        self.intercept = -0.15  # Base sensitivity
        self.is_fitted = True
        self.r_squared = 0.95  # Synthetic R-squared value
