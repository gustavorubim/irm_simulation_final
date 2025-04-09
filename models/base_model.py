"""
Base model class for EVE (Economic Value of Equity) and EVS (Economic Value Sensitivity) models.
This module provides the foundation for all regression models in the system.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union


class BaseRegressionModel(ABC):
    """
    Abstract base class for all regression models in the system.
    
    This class defines the interface that all regression models must implement
    and provides common functionality for model evaluation and sensitivity analysis.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the base regression model.
        
        Args:
            name: Unique identifier for the model
            description: Human-readable description of the model
        """
        self.name = name
        self.description = description
        self.coefficients = {}
        self.intercept = 0.0
        self.r_squared = 0.0
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the regression model to the provided data.
        
        Args:
            X: Features/independent variables
            y: Target/dependent variable
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features/independent variables
            
        Returns:
            Predicted values
        """
        pass
    
    def calculate_sensitivity(self, X: pd.DataFrame, factor_name: str, 
                             delta: float = 0.01) -> np.ndarray:
        """
        Calculate sensitivity of the model to a specific factor.
        
        Args:
            X: Features/independent variables
            factor_name: Name of the factor to analyze sensitivity for
            delta: Change in factor value for sensitivity calculation
            
        Returns:
            Sensitivity values for each input row
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating sensitivity")
            
        if factor_name not in X.columns:
            raise ValueError(f"Factor {factor_name} not found in input data")
            
        # Create a copy with increased factor values
        X_plus = X.copy()
        X_plus[factor_name] += delta
        
        # Calculate predictions for original and modified inputs
        y_pred = self.predict(X)
        y_pred_plus = self.predict(X_plus)
        
        # Calculate sensitivity as change in output divided by change in input
        sensitivity = (y_pred_plus - y_pred) / delta
        
        return sensitivity
    
    def calculate_all_sensitivities(self, X: pd.DataFrame, 
                                   delta: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Calculate sensitivity of the model to all factors.
        
        Args:
            X: Features/independent variables
            delta: Change in factor value for sensitivity calculation
            
        Returns:
            Dictionary mapping factor names to sensitivity values
        """
        sensitivities = {}
        
        for factor in X.columns:
            sensitivities[factor] = self.calculate_sensitivity(X, factor, delta)
            
        return sensitivities
    
    def summary(self) -> Dict:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.name,
            "description": self.description,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "is_fitted": self.is_fitted
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        if not self.is_fitted:
            return f"{self.name}: Not fitted"
        
        coef_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.coefficients.items()])
        return f"{self.name}: y = {self.intercept:.4f} + {coef_str}, R² = {self.r_squared:.4f}"


class LinearRegressionModel(BaseRegressionModel):
    """
    Linear regression model implementation.
    
    This class implements a simple linear regression model with ordinary least squares.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the linear regression model using ordinary least squares.
        
        Args:
            X: Features/independent variables
            y: Target/dependent variable
        """
        # Add constant term for intercept
        X_with_const = X.copy()
        X_with_const['const'] = 1.0
        
        # Calculate coefficients using normal equation: β = (X'X)^(-1)X'y
        X_matrix = X_with_const.values
        y_vector = y.values
        
        # Calculate coefficients
        try:
            beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
        
        # Extract intercept and coefficients
        self.intercept = beta[-1]
        self.coefficients = {col: beta[i] for i, col in enumerate(X.columns)}
        
        # Calculate R-squared
        y_pred = self.predict(X)
        y_mean = np.mean(y_vector)
        ss_total = np.sum((y_vector - y_mean) ** 2)
        ss_residual = np.sum((y_vector - y_pred) ** 2)
        self.r_squared = 1 - (ss_residual / ss_total)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted linear model.
        
        Args:
            X: Features/independent variables
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Calculate predictions
        predictions = np.zeros(len(X))
        predictions += self.intercept
        
        for col, coef in self.coefficients.items():
            if col in X.columns:
                predictions += coef * X[col].values
        
        return predictions


class EVEModel(BaseRegressionModel):
    """
    Economic Value of Equity (EVE) model.
    
    This model estimates the change in economic value of equity based on
    interest rate movements and other factors.
    """
    
    def __init__(self, name: str, description: str = "", model_type: str = "linear"):
        """
        Initialize the EVE model.
        
        Args:
            name: Unique identifier for the model
            description: Human-readable description of the model
            model_type: Type of regression model to use (linear, log-linear, etc.)
        """
        super().__init__(name, description)
        self.model_type = model_type
        self.rate_sensitivities = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the EVE model to the provided data.
        
        Args:
            X: Features/independent variables (including interest rates)
            y: Target/dependent variable (EVE values)
        """
        # Identify interest rate columns (assuming they contain 'rate' in the name)
        rate_columns = [col for col in X.columns if 'rate' in col.lower()]
        
        # Fit linear regression model
        if self.model_type == "linear":
            # Add constant term for intercept
            X_with_const = X.copy()
            X_with_const['const'] = 1.0
            
            # Calculate coefficients using normal equation
            X_matrix = X_with_const.values
            y_vector = y.values
            
            try:
                beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
            
            # Extract intercept and coefficients
            self.intercept = beta[-1]
            self.coefficients = {col: beta[i] for i, col in enumerate(X.columns)}
            
            # Calculate R-squared
            y_pred = self.predict(X)
            y_mean = np.mean(y_vector)
            ss_total = np.sum((y_vector - y_mean) ** 2)
            ss_residual = np.sum((y_vector - y_pred) ** 2)
            self.r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate rate sensitivities
            for rate_col in rate_columns:
                self.rate_sensitivities[rate_col] = self.coefficients.get(rate_col, 0.0)
        
        elif self.model_type == "log-linear":
            # Take log of dependent variable
            log_y = np.log(y.values)
            
            # Fit linear model to log-transformed data
            X_with_const = X.copy()
            X_with_const['const'] = 1.0
            
            X_matrix = X_with_const.values
            y_vector = log_y
            
            try:
                beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
            
            self.intercept = beta[-1]
            self.coefficients = {col: beta[i] for i, col in enumerate(X.columns)}
            
            # Calculate R-squared on original scale
            y_pred = self.predict(X)
            y_mean = np.mean(y.values)
            ss_total = np.sum((y.values - y_mean) ** 2)
            ss_residual = np.sum((y.values - y_pred) ** 2)
            self.r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate rate sensitivities (semi-elasticities for log-linear model)
            for rate_col in rate_columns:
                self.rate_sensitivities[rate_col] = self.coefficients.get(rate_col, 0.0)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted EVE model.
        
        Args:
            X: Features/independent variables
            
        Returns:
            Predicted EVE values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == "linear":
            # Linear prediction
            predictions = np.zeros(len(X))
            predictions += self.intercept
            
            for col, coef in self.coefficients.items():
                if col in X.columns:
                    predictions += coef * X[col].values
                    
            return predictions
        
        elif self.model_type == "log-linear":
            # Log-linear prediction (exponentiate the linear prediction)
            log_predictions = np.zeros(len(X))
            log_predictions += self.intercept
            
            for col, coef in self.coefficients.items():
                if col in X.columns:
                    log_predictions += coef * X[col].values
                    
            return np.exp(log_predictions)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def calculate_rate_sensitivity(self, rate_name: str) -> float:
        """
        Get the sensitivity of EVE to a specific interest rate.
        
        Args:
            rate_name: Name of the interest rate
            
        Returns:
            Sensitivity coefficient
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating sensitivity")
            
        return self.rate_sensitivities.get(rate_name, 0.0)


class EVSModel(BaseRegressionModel):
    """
    Economic Value Sensitivity (EVS) model.
    
    This model estimates the sensitivity of economic value to
    interest rate movements and other factors.
    """
    
    def __init__(self, name: str, description: str = "", model_type: str = "linear",
                duration_based: bool = False):
        """
        Initialize the EVS model.
        
        Args:
            name: Unique identifier for the model
            description: Human-readable description of the model
            model_type: Type of regression model to use (linear, log-linear, etc.)
            duration_based: Whether to use duration-based approach for sensitivity
        """
        super().__init__(name, description)
        self.model_type = model_type
        self.duration_based = duration_based
        self.durations = {}
        self.convexities = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the EVS model to the provided data.
        
        Args:
            X: Features/independent variables (including interest rates)
            y: Target/dependent variable (EVS values)
        """
        # Identify interest rate columns
        rate_columns = [col for col in X.columns if 'rate' in col.lower()]
        
        if self.duration_based:
            # Duration-based approach
            # For each rate, calculate duration and convexity
            for rate_col in rate_columns:
                # Calculate duration (first derivative of value with respect to rate)
                X_plus = X.copy()
                X_minus = X.copy()
                delta = 0.0001  # Small change for numerical differentiation
                
                X_plus[rate_col] += delta
                X_minus[rate_col] -= delta
                
                # Fit a simple model to estimate values at different rate levels
                temp_model = LinearRegressionModel(f"temp_{rate_col}")
                temp_model.fit(X, y)
                
                # Calculate values at different rate levels
                v0 = temp_model.predict(X)
                v_plus = temp_model.predict(X_plus)
                v_minus = temp_model.predict(X_minus)
                
                # Calculate duration (negative of first derivative divided by value)
                duration = -(v_plus - v_minus) / (2 * delta) / v0
                self.durations[rate_col] = np.mean(duration)
                
                # Calculate convexity (second derivative divided by value)
                convexity = (v_plus + v_minus - 2 * v0) / (delta ** 2) / v0
                self.convexities[rate_col] = np.mean(convexity)
            
            # Store coefficients based on durations
            self.coefficients = {rate_col: -self.durations[rate_col] 
                                for rate_col in rate_columns}
            self.intercept = np.mean(y.values)
            
            # Calculate R-squared
            y_pred = self.predict(X)
            y_mean = np.mean(y.values)
            ss_total = np.sum((y.values - y_mean) ** 2)
            ss_residual = np.sum((y.values - y_pred) ** 2)
            self.r_squared = 1 - (ss_residual / ss_total)
            
        else:
            # Regression-based approach
            if self.model_type == "linear":
                # Add constant term for intercept
                X_with_const = X.copy()
                X_with_const['const'] = 1.0
                
                # Calculate coefficients using normal equation
                X_matrix = X_with_const.values
                y_vector = y.values
                
                try:
                    beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
                
                # Extract intercept and coefficients
                self.intercept = beta[-1]
                self.coefficients = {col: beta[i] for i, col in enumerate(X.columns)}
                
                # Calculate R-squared
                y_pred = self.predict(X)
                y_mean = np.mean(y_vector)
                ss_total = np.sum((y_vector - y_mean) ** 2)
                ss_residual = np.sum((y_vector - y_pred) ** 2)
                self.r_squared = 1 - (ss_residual / ss_total)
            
            elif self.model_type == "log-linear":
                # Take log of dependent variable (if positive)
                if np.any(y <= 0):
                    raise ValueError("Log-linear model requires positive values")
                    
                log_y = np.log(y.values)
                
                # Fit linear model to log-transformed data
                X_with_const = X.copy()
                X_with_const['const'] = 1.0
                
                X_matrix = X_with_const.values
                y_vector = log_y
                
                try:
                    beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
                
                self.intercept = beta[-1]
                self.coefficients = {col: beta[i] for i, col in enumerate(X.columns)}
                
                # Calculate R-squared on original scale
                y_pred = self.predict(X)
                y_mean = np.mean(y.values)
                ss_total = np.sum((y.values - y_mean) ** 2)
                ss_residual = np.sum((y.values - y_pred) ** 2)
                self.r_squared = 1 - (ss_residual / ss_total)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted EVS model.
        
        Args:
            X: Features/independent variables
            
        Returns:
            Predicted EVS values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.duration_based:
            # Duration-based prediction
            predictions = np.zeros(len(X)) + self.intercept
            
            # Apply duration and convexity adjustments
            for rate_col, duration in self.durations.items():
                if rate_col in X.columns:
                    # First-order (duration) effect
                    predictions -= duration * X[rate_col].values
                    
                    # Second-order (convexity) effect if available
                    if rate_col in self.convexities:
                        convexity = self.convexities[rate_col]
                        predictions += 0.5 * convexity * (X[rate_col].values ** 2)
            
            return predictions
            
        elif self.model_type == "linear":
            # Linear prediction
            predictions = np.zeros(len(X))
            predictions += self.intercept
            
            for col, coef in self.coefficients.items():
                if col in X.columns:
                    predictions += coef * X[col].values
                    
            return predictions
        
        elif self.model_type == "log-linear":
            # Log-linear prediction
            log_predictions = np.zeros(len(X))
            log_predictions += self.intercept
            
            for col, coef in self.coefficients.items():
                if col in X.columns:
                    log_predictions += coef * X[col].values
                    
            return np.exp(log_predictions)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def get_duration(self, rate_name: str) -> float:
        """
        Get the duration with respect to a specific interest rate.
        
        Args:
            rate_name: Name of the interest rate
            
        Returns:
            Duration value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting duration")
            
        if not self.duration_based:
            raise ValueError("Duration is only available for duration-based models")
            
        return self.durations.get(rate_name, 0.0)
    
    def get_convexity(self, rate_name: str) -> float:
        """
        Get the convexity with respect to a specific interest rate.
        
        Args:
            rate_name: Name of the interest rate
            
        Returns:
            Convexity value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting convexity")
            
        if not self.duration_based:
            raise ValueError("Convexity is only available for duration-based models")
            
        return self.convexities.get(rate_name, 0.0)
