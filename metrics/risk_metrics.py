"""
Metrics calculation module for EVE and EVS analysis.
This module provides comprehensive risk metrics for interest rate risk management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
import seaborn as sns


class RiskMetricsCalculator:
    """
    Calculator for risk metrics related to EVE and EVS analysis.
    
    This class provides methods to calculate various risk metrics
    from simulation results, including VaR, Expected Shortfall,
    duration, convexity, and other risk measures.
    """
    
    def __init__(self):
        """Initialize the risk metrics calculator."""
        pass
    
    def calculate_var(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            data: Array of values
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            VaR value
        """
        return np.percentile(data, 100 * (1 - confidence_level))
    
    def calculate_expected_shortfall(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (ES).
        
        Args:
            data: Array of values
            confidence_level: Confidence level for ES calculation (default: 0.95)
            
        Returns:
            ES value
        """
        var = self.calculate_var(data, confidence_level)
        tail_values = data[data <= var]
        return np.mean(tail_values) if len(tail_values) > 0 else var
    
    def calculate_conditional_var(self, data: np.ndarray, condition_data: np.ndarray, 
                                condition_threshold: float, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CoVaR).
        
        Args:
            data: Array of values
            condition_data: Array of condition values
            condition_threshold: Threshold for condition
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            CoVaR value
        """
        # Filter data based on condition
        filtered_data = data[condition_data <= condition_threshold]
        
        # Calculate VaR on filtered data
        if len(filtered_data) > 0:
            return np.percentile(filtered_data, 100 * (1 - confidence_level))
        else:
            return np.nan
    
    def calculate_marginal_var(self, portfolio_data: np.ndarray, component_data: np.ndarray,
                             confidence_level: float = 0.95) -> float:
        """
        Calculate Marginal Value at Risk (MVaR).
        
        Args:
            portfolio_data: Array of portfolio values
            component_data: Array of component values
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            MVaR value
        """
        # Calculate portfolio VaR
        portfolio_var = self.calculate_var(portfolio_data, confidence_level)
        
        # Find scenarios where portfolio value is at or below VaR
        var_scenarios = portfolio_data <= portfolio_var
        
        # Calculate average component value in those scenarios
        component_var = np.mean(component_data[var_scenarios]) if np.any(var_scenarios) else np.nan
        
        # Calculate marginal VaR
        return component_var
    
    def calculate_component_var(self, portfolio_data: np.ndarray, component_data: np.ndarray,
                              confidence_level: float = 0.95) -> float:
        """
        Calculate Component Value at Risk (CVaR).
        
        Args:
            portfolio_data: Array of portfolio values
            component_data: Array of component values
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            CVaR value
        """
        # Calculate marginal VaR
        mvar = self.calculate_marginal_var(portfolio_data, component_data, confidence_level)
        
        # Calculate component weight
        component_weight = np.mean(component_data) / np.mean(portfolio_data)
        
        # Calculate component VaR
        return mvar * component_weight
    
    def calculate_incremental_var(self, portfolio_data: np.ndarray, portfolio_without_component_data: np.ndarray,
                                confidence_level: float = 0.95) -> float:
        """
        Calculate Incremental Value at Risk (IVaR).
        
        Args:
            portfolio_data: Array of portfolio values
            portfolio_without_component_data: Array of portfolio values without the component
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            IVaR value
        """
        # Calculate VaR for full portfolio
        portfolio_var = self.calculate_var(portfolio_data, confidence_level)
        
        # Calculate VaR for portfolio without component
        portfolio_without_component_var = self.calculate_var(portfolio_without_component_data, confidence_level)
        
        # Calculate incremental VaR
        return portfolio_var - portfolio_without_component_var
    
    def calculate_duration(self, values: np.ndarray, rates: np.ndarray, delta: float = 0.0001) -> float:
        """
        Calculate duration.
        
        Args:
            values: Array of values
            rates: Array of rates
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Duration value
        """
        # Calculate average value
        avg_value = np.mean(values)
        
        # Calculate average rate
        avg_rate = np.mean(rates)
        
        # Calculate derivative using linear regression
        X = rates.reshape(-1, 1)
        y = values
        
        # Add constant term for intercept
        X_with_const = np.hstack([X, np.ones_like(X)])
        
        # Calculate coefficients using normal equation
        try:
            beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
            
        # Extract slope
        slope = beta[0]
        
        # Calculate duration (negative of derivative divided by value)
        duration = -slope / avg_value
        
        return duration
    
    def calculate_convexity(self, values: np.ndarray, rates: np.ndarray, delta: float = 0.0001) -> float:
        """
        Calculate convexity.
        
        Args:
            values: Array of values
            rates: Array of rates
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Convexity value
        """
        # Calculate average value
        avg_value = np.mean(values)
        
        # Calculate average rate
        avg_rate = np.mean(rates)
        
        # Calculate second derivative using quadratic regression
        X = rates.reshape(-1, 1)
        y = values
        
        # Add squared term and constant term
        X_with_squared = np.hstack([X**2, X, np.ones_like(X)])
        
        # Calculate coefficients using normal equation
        try:
            beta = np.linalg.inv(X_with_squared.T @ X_with_squared) @ X_with_squared.T @ y
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_with_squared.T @ X_with_squared) @ X_with_squared.T @ y
            
        # Extract coefficient of squared term
        second_derivative = 2 * beta[0]
        
        # Calculate convexity (second derivative divided by value)
        convexity = second_derivative / avg_value
        
        return convexity
    
    def calculate_basis_point_value(self, values: np.ndarray, rates: np.ndarray, bp: float = 0.0001) -> float:
        """
        Calculate Basis Point Value (BPV).
        
        Args:
            values: Array of values
            rates: Array of rates
            bp: Basis point value (default: 0.0001 for 1bp)
            
        Returns:
            BPV value
        """
        # Calculate duration
        duration = self.calculate_duration(values, rates)
        
        # Calculate average value
        avg_value = np.mean(values)
        
        # Calculate BPV
        bpv = duration * avg_value * bp
        
        return bpv
    
    def calculate_key_rate_duration(self, model_results: Dict[str, np.ndarray], 
                                  rate_paths: Dict[str, np.ndarray],
                                  rate_tenor: str, delta: float = 0.0001) -> Dict[str, float]:
        """
        Calculate Key Rate Duration (KRD).
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            rate_tenor: Rate tenor to calculate KRD for
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Dictionary mapping model names to KRD values
        """
        krd_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate KRD for this model
            values = model_results_array[:, -1]  # Use last time step
            rates = rate_paths[rate_tenor][:, -1]  # Use last time step
            
            krd = self.calculate_duration(values, rates, delta)
            krd_values[model_name] = krd
            
        return krd_values
    
    def calculate_key_rate_convexity(self, model_results: Dict[str, np.ndarray], 
                                   rate_paths: Dict[str, np.ndarray],
                                   rate_tenor: str, delta: float = 0.0001) -> Dict[str, float]:
        """
        Calculate Key Rate Convexity (KRC).
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            rate_tenor: Rate tenor to calculate KRC for
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Dictionary mapping model names to KRC values
        """
        krc_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate KRC for this model
            values = model_results_array[:, -1]  # Use last time step
            rates = rate_paths[rate_tenor][:, -1]  # Use last time step
            
            krc = self.calculate_convexity(values, rates, delta)
            krc_values[model_name] = krc
            
        return krc_values
    
    def calculate_effective_duration(self, model_results: Dict[str, np.ndarray], 
                                   rate_paths: Dict[str, np.ndarray],
                                   delta: float = 0.0001) -> Dict[str, float]:
        """
        Calculate Effective Duration.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Dictionary mapping model names to Effective Duration values
        """
        duration_values = {}
        
        # Calculate weighted average rate
        all_rates = np.concatenate([rates[:, -1] for rates in rate_paths.values()])
        avg_rate = np.mean(all_rates)
        
        for model_name, model_results_array in model_results.items():
            # Calculate Effective Duration for this model
            values = model_results_array[:, -1]  # Use last time step
            
            # Create composite rate array
            composite_rates = np.zeros_like(values)
            for tenor, rates in rate_paths.items():
                composite_rates += rates[:, -1]
            composite_rates /= len(rate_paths)
            
            duration = self.calculate_duration(values, composite_rates, delta)
            duration_values[model_name] = duration
            
        return duration_values
    
    def calculate_effective_convexity(self, model_results: Dict[str, np.ndarray], 
                                    rate_paths: Dict[str, np.ndarray],
                                    delta: float = 0.0001) -> Dict[str, float]:
        """
        Calculate Effective Convexity.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            delta: Rate change for numerical differentiation (default: 0.0001)
            
        Returns:
            Dictionary mapping model names to Effective Convexity values
        """
        convexity_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate Effective Convexity for this model
            values = model_results_array[:, -1]  # Use last time step
            
            # Create composite rate array
            composite_rates = np.zeros_like(values)
            for tenor, rates in rate_paths.items():
                composite_rates += rates[:, -1]
            composite_rates /= len(rate_paths)
            
            convexity = self.calculate_convexity(values, composite_rates, delta)
            convexity_values[model_name] = convexity
            
        return convexity_values
    
    def calculate_rate_sensitivity(self, model_results: Dict[str, np.ndarray], 
                                 rate_paths: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate rate sensitivity.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            
        Returns:
            Dictionary mapping model names to dictionaries mapping rate tenors to sensitivity values
        """
        sensitivity_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate sensitivity for this model
            model_sensitivities = {}
            
            for tenor, rates in rate_paths.items():
                values = model_results_array[:, -1]  # Use last time step
                rates_last_step = rates[:, -1]  # Use last time step
                
                # Calculate correlation
                correlation = np.corrcoef(values, rates_last_step)[0, 1]
                
                # Calculate regression coefficient
                X = rates_last_step.reshape(-1, 1)
                y = values
                
                # Add constant term for intercept
                X_with_const = np.hstack([X, np.ones_like(X)])
                
                # Calculate coefficients using normal equation
                try:
                    beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
                    
                # Extract slope
                sensitivity = beta[0]
                
                model_sensitivities[tenor] = sensitivity
                
            sensitivity_values[model_name] = model_sensitivities
            
        return sensitivity_values
    
    def calculate_correlation_matrix(self, model_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate correlation matrix between model results.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            
        Returns:
            DataFrame containing correlation matrix
        """
        # Extract last time step values for each model
        last_step_values = {}
        for model_name, model_results_array in model_results.items():
            last_step_values[model_name] = model_results_array[:, -1]
            
        # Create DataFrame
        df = pd.DataFrame(last_step_values)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        return correlation_matrix
    
    def calculate_covariance_matrix(self, model_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate covariance matrix between model results.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            
        Returns:
            DataFrame containing covariance matrix
        """
        # Extract last time step values for each model
        last_step_values = {}
        for model_name, model_results_array in model_results.items():
            last_step_values[model_name] = model_results_array[:, -1]
            
        # Create DataFrame
        df = pd.DataFrame(last_step_values)
        
        # Calculate covariance matrix
        covariance_matrix = df.cov()
        
        return covariance_matrix
    
    def calculate_beta(self, model_results: Dict[str, np.ndarray], 
                     benchmark_model: str) -> Dict[str, float]:
        """
        Calculate beta of models relative to a benchmark model.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            benchmark_model: Name of the benchmark model
            
        Returns:
            Dictionary mapping model names to beta values
        """
        if benchmark_model not in model_results:
            raise ValueError(f"Benchmark model '{benchmark_model}' not found in model results")
            
        beta_values = {}
        benchmark_values = model_results[benchmark_model][:, -1]  # Use last time step
        
        for model_name, model_results_array in model_results.items():
            if model_name == benchmark_model:
                beta_values[model_name] = 1.0
                continue
                
            # Calculate beta for this model
            values = model_results_array[:, -1]  # Use last time step
            
            # Calculate covariance
            covariance = np.cov(values, benchmark_values)[0, 1]
            
            # Calculate variance of benchmark
            benchmark_variance = np.var(benchmark_values)
            
            # Calculate beta
            beta = covariance / benchmark_variance
            
            beta_values[model_name] = beta
            
        return beta_values
    
    def calculate_tracking_error(self, model_results: Dict[str, np.ndarray], 
                               benchmark_model: str) -> Dict[str, float]:
        """
        Calculate tracking error of models relative to a benchmark model.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            benchmark_model: Name of the benchmark model
            
        Returns:
            Dictionary mapping model names to tracking error values
        """
        if benchmark_model not in model_results:
            raise ValueError(f"Benchmark model '{benchmark_model}' not found in model results")
            
        tracking_error_values = {}
        benchmark_values = model_results[benchmark_model]  # All time steps
        
        for model_name, model_results_array in model_results.items():
            if model_name == benchmark_model:
                tracking_error_values[model_name] = 0.0
                continue
                
            # Calculate tracking error for this model
            values = model_results_array  # All time steps
            
            # Calculate difference at each time step
            diff = values - benchmark_values
            
            # Calculate tracking error (standard deviation of differences)
            tracking_error = np.std(diff[:, -1])  # Use last time step
            
            tracking_error_values[model_name] = tracking_error
            
        return tracking_error_values
    
    def calculate_information_ratio(self, model_results: Dict[str, np.ndarray], 
                                  benchmark_model: str) -> Dict[str, float]:
        """
        Calculate information ratio of models relative to a benchmark model.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            benchmark_model: Name of the benchmark model
            
        Returns:
            Dictionary mapping model names to information ratio values
        """
        if benchmark_model not in model_results:
            raise ValueError(f"Benchmark model '{benchmark_model}' not found in model results")
            
        information_ratio_values = {}
        benchmark_values = model_results[benchmark_model][:, -1]  # Use last time step
        benchmark_mean = np.mean(benchmark_values)
        
        for model_name, model_results_array in model_results.items():
            if model_name == benchmark_model:
                information_ratio_values[model_name] = 0.0
                continue
                
            # Calculate information ratio for this model
            values = model_results_array[:, -1]  # Use last time step
            model_mean = np.mean(values)
            
            # Calculate tracking error
            diff = values - benchmark_values
            tracking_error = np.std(diff)
            
            # Calculate information ratio
            if tracking_error > 0:
                information_ratio = (model_mean - benchmark_mean) / tracking_error
            else:
                information_ratio = 0.0
                
            information_ratio_values[model_name] = information_ratio
            
        return information_ratio_values
    
    def calculate_sharpe_ratio(self, model_results: Dict[str, np.ndarray], 
                             risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate Sharpe ratio.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Dictionary mapping model names to Sharpe ratio values
        """
        sharpe_ratio_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate Sharpe ratio for this model
            values = model_results_array[:, -1]  # Use last time step
            model_mean = np.mean(values)
            model_std = np.std(values)
            
            # Calculate Sharpe ratio
            if model_std > 0:
                sharpe_ratio = (model_mean - risk_free_rate) / model_std
            else:
                sharpe_ratio = 0.0
                
            sharpe_ratio_values[model_name] = sharpe_ratio
            
        return sharpe_ratio_values
    
    def calculate_sortino_ratio(self, model_results: Dict[str, np.ndarray], 
                              risk_free_rate: float = 0.0,
                              target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Sortino ratio.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            risk_free_rate: Risk-free rate (default: 0.0)
            target_return: Target return (default: None, uses risk-free rate)
            
        Returns:
            Dictionary mapping model names to Sortino ratio values
        """
        if target_return is None:
            target_return = risk_free_rate
            
        sortino_ratio_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate Sortino ratio for this model
            values = model_results_array[:, -1]  # Use last time step
            model_mean = np.mean(values)
            
            # Calculate downside deviation
            downside_diff = np.minimum(values - target_return, 0)
            downside_deviation = np.sqrt(np.mean(downside_diff**2))
            
            # Calculate Sortino ratio
            if downside_deviation > 0:
                sortino_ratio = (model_mean - risk_free_rate) / downside_deviation
            else:
                sortino_ratio = 0.0
                
            sortino_ratio_values[model_name] = sortino_ratio
            
        return sortino_ratio_values
    
    def calculate_maximum_drawdown(self, model_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate Maximum Drawdown.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            
        Returns:
            Dictionary mapping model names to Maximum Drawdown values
        """
        max_drawdown_values = {}
        
        for model_name, model_results_array in model_results.items():
            # Calculate Maximum Drawdown for this model
            # For each scenario, calculate the maximum drawdown
            scenario_drawdowns = []
            
            for scenario in range(model_results_array.shape[0]):
                scenario_values = model_results_array[scenario, :]
                
                # Calculate cumulative maximum
                cumulative_max = np.maximum.accumulate(scenario_values)
                
                # Calculate drawdown
                drawdown = (cumulative_max - scenario_values) / cumulative_max
                
                # Calculate maximum drawdown
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                
                scenario_drawdowns.append(max_drawdown)
                
            # Calculate average maximum drawdown across scenarios
            max_drawdown_values[model_name] = np.mean(scenario_drawdowns)
            
        return max_drawdown_values
    
    def calculate_calmar_ratio(self, model_results: Dict[str, np.ndarray], 
                             risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate Calmar ratio.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Dictionary mapping model names to Calmar ratio values
        """
        calmar_ratio_values = {}
        
        # Calculate Maximum Drawdown
        max_drawdown_values = self.calculate_maximum_drawdown(model_results)
        
        for model_name, model_results_array in model_results.items():
            # Calculate Calmar ratio for this model
            values = model_results_array[:, -1]  # Use last time step
            model_mean = np.mean(values)
            
            # Get maximum drawdown
            max_drawdown = max_drawdown_values[model_name]
            
            # Calculate Calmar ratio
            if max_drawdown > 0:
                calmar_ratio = (model_mean - risk_free_rate) / max_drawdown
            else:
                calmar_ratio = 0.0
                
            calmar_ratio_values[model_name] = calmar_ratio
            
        return calmar_ratio_values
    
    def calculate_comprehensive_metrics(self, model_results: Dict[str, np.ndarray], 
                                      rate_paths: Dict[str, np.ndarray],
                                      risk_free_rate: float = 0.0,
                                      benchmark_model: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive metrics for all models.
        
        Args:
            model_results: Dictionary mapping model names to arrays of results
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            risk_free_rate: Risk-free rate (default: 0.0)
            benchmark_model: Name of the benchmark model (optional)
            
        Returns:
            Dictionary mapping model names to dictionaries of metrics
        """
        comprehensive_metrics = {}
        
        # If benchmark model not specified, use first model
        if benchmark_model is None and len(model_results) > 0:
            benchmark_model = next(iter(model_results.keys()))
            
        # Calculate metrics for each model
        for model_name, model_results_array in model_results.items():
            # Initialize metrics dictionary
            metrics = {}
            
            # Basic statistics
            values = model_results_array[:, -1]  # Use last time step
            metrics['mean'] = np.mean(values)
            metrics['std'] = np.std(values)
            metrics['min'] = np.min(values)
            metrics['max'] = np.max(values)
            metrics['median'] = np.median(values)
            metrics['skewness'] = stats.skew(values)
            metrics['kurtosis'] = stats.kurtosis(values)
            
            # Risk metrics
            metrics['var_95'] = self.calculate_var(values, 0.95)
            metrics['var_99'] = self.calculate_var(values, 0.99)
            metrics['es_95'] = self.calculate_expected_shortfall(values, 0.95)
            metrics['es_99'] = self.calculate_expected_shortfall(values, 0.99)
            
            # Performance metrics
            metrics['sharpe_ratio'] = (metrics['mean'] - risk_free_rate) / metrics['std'] if metrics['std'] > 0 else 0.0
            
            # Calculate downside deviation
            downside_diff = np.minimum(values - risk_free_rate, 0)
            downside_deviation = np.sqrt(np.mean(downside_diff**2))
            metrics['sortino_ratio'] = (metrics['mean'] - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            
            # Calculate maximum drawdown
            scenario_drawdowns = []
            for scenario in range(model_results_array.shape[0]):
                scenario_values = model_results_array[scenario, :]
                cumulative_max = np.maximum.accumulate(scenario_values)
                drawdown = (cumulative_max - scenario_values) / cumulative_max
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                scenario_drawdowns.append(max_drawdown)
            metrics['max_drawdown'] = np.mean(scenario_drawdowns)
            
            # Calculate Calmar ratio
            metrics['calmar_ratio'] = (metrics['mean'] - risk_free_rate) / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0.0
            
            # Rate sensitivity metrics
            for tenor, rates in rate_paths.items():
                rates_last_step = rates[:, -1]  # Use last time step
                
                # Calculate correlation
                metrics[f'correlation_{tenor}'] = np.corrcoef(values, rates_last_step)[0, 1]
                
                # Calculate duration
                metrics[f'duration_{tenor}'] = self.calculate_duration(values, rates_last_step)
                
                # Calculate convexity
                metrics[f'convexity_{tenor}'] = self.calculate_convexity(values, rates_last_step)
                
                # Calculate BPV
                metrics[f'bpv_{tenor}'] = self.calculate_basis_point_value(values, rates_last_step)
            
            # Benchmark-relative metrics
            if benchmark_model is not None and benchmark_model in model_results:
                benchmark_values = model_results[benchmark_model][:, -1]  # Use last time step
                benchmark_mean = np.mean(benchmark_values)
                
                # Calculate tracking error
                diff = values - benchmark_values
                tracking_error = np.std(diff)
                metrics['tracking_error'] = tracking_error
                
                # Calculate information ratio
                metrics['information_ratio'] = (metrics['mean'] - benchmark_mean) / tracking_error if tracking_error > 0 else 0.0
                
                # Calculate beta
                covariance = np.cov(values, benchmark_values)[0, 1]
                benchmark_variance = np.var(benchmark_values)
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                
                # Calculate alpha
                metrics['alpha'] = metrics['mean'] - (risk_free_rate + metrics['beta'] * (benchmark_mean - risk_free_rate))
            
            comprehensive_metrics[model_name] = metrics
            
        return comprehensive_metrics
    
    def plot_metrics_heatmap(self, metrics: Dict[str, Dict[str, float]], 
                           metric_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot heatmap of metrics.
        
        Args:
            metrics: Dictionary mapping model names to dictionaries of metrics
            metric_names: List of metric names to include in heatmap (optional)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        # If metric names not specified, use common metrics
        if metric_names is None:
            # Find common metrics across all models
            common_metrics = set()
            for model_metrics in metrics.values():
                if not common_metrics:
                    common_metrics = set(model_metrics.keys())
                else:
                    common_metrics &= set(model_metrics.keys())
            metric_names = sorted(common_metrics)
            
        # Create DataFrame
        data = []
        for model_name, model_metrics in metrics.items():
            row = {'model': model_name}
            for metric_name in metric_names:
                if metric_name in model_metrics:
                    row[metric_name] = model_metrics[metric_name]
                else:
                    row[metric_name] = np.nan
            data.append(row)
            
        df = pd.DataFrame(data)
        df.set_index('model', inplace=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(df, annot=True, cmap='coolwarm', ax=ax)
        
        # Add labels and title
        ax.set_title('Risk Metrics Heatmap')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_radar(self, metrics: Dict[str, Dict[str, float]], 
                         model_names: Optional[List[str]] = None,
                         metric_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot radar chart of metrics.
        
        Args:
            metrics: Dictionary mapping model names to dictionaries of metrics
            model_names: List of model names to include in radar chart (optional)
            metric_names: List of metric names to include in radar chart (optional)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        # If model names not specified, use all models
        if model_names is None:
            model_names = sorted(metrics.keys())
            
        # If metric names not specified, use common metrics
        if metric_names is None:
            # Find common metrics across specified models
            common_metrics = set()
            for model_name in model_names:
                if model_name in metrics:
                    if not common_metrics:
                        common_metrics = set(metrics[model_name].keys())
                    else:
                        common_metrics &= set(metrics[model_name].keys())
            metric_names = sorted(common_metrics)
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create radar chart
        num_metrics = len(metric_names)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = fig.add_subplot(111, polar=True)
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        
        # Normalize metrics for radar chart
        normalized_metrics = {}
        for metric_name in metric_names:
            values = [metrics[model_name][metric_name] for model_name in model_names if model_name in metrics]
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                normalized_metrics[metric_name] = {model_name: (metrics[model_name][metric_name] - min_val) / (max_val - min_val)
                                                for model_name in model_names if model_name in metrics}
            else:
                normalized_metrics[metric_name] = {model_name: 0.5 for model_name in model_names if model_name in metrics}
        
        # Plot each model
        for model_name in model_names:
            if model_name not in metrics:
                continue
                
            values = [normalized_metrics[metric_name][model_name] for metric_name in metric_names]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
            
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title('Risk Metrics Radar Chart')
        
        return fig


class PortfolioAnalyzer:
    """
    Analyzer for portfolio-level metrics.
    
    This class provides methods to analyze portfolio-level metrics
    based on component model results.
    """
    
    def __init__(self, risk_metrics_calculator: RiskMetricsCalculator):
        """
        Initialize the portfolio analyzer.
        
        Args:
            risk_metrics_calculator: Risk metrics calculator
        """
        self.risk_metrics_calculator = risk_metrics_calculator
        
    def calculate_portfolio_value(self, component_results: Dict[str, np.ndarray], 
                                weights: Dict[str, float]) -> np.ndarray:
        """
        Calculate portfolio value based on component results and weights.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            weights: Dictionary mapping component names to weights
            
        Returns:
            Array of portfolio values
        """
        # Validate inputs
        for component_name in weights:
            if component_name not in component_results:
                raise ValueError(f"Component '{component_name}' not found in component results")
                
        # Get dimensions
        first_component = next(iter(component_results.values()))
        num_scenarios, num_steps = first_component.shape
        
        # Initialize portfolio values
        portfolio_values = np.zeros((num_scenarios, num_steps))
        
        # Calculate weighted sum
        for component_name, component_weight in weights.items():
            portfolio_values += component_weight * component_results[component_name]
            
        return portfolio_values
    
    def calculate_component_contributions(self, component_results: Dict[str, np.ndarray], 
                                        weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate component contributions to portfolio risk.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            weights: Dictionary mapping component names to weights
            
        Returns:
            Dictionary mapping component names to contribution values
        """
        # Calculate portfolio value
        portfolio_values = self.calculate_portfolio_value(component_results, weights)
        
        # Calculate portfolio VaR
        portfolio_var = self.risk_metrics_calculator.calculate_var(portfolio_values[:, -1], 0.95)
        
        # Calculate component contributions
        contributions = {}
        
        for component_name, component_results_array in component_results.items():
            # Calculate component VaR
            component_var = self.risk_metrics_calculator.calculate_component_var(
                portfolio_values[:, -1], component_results_array[:, -1], 0.95)
            
            # Calculate contribution
            contributions[component_name] = component_var / portfolio_var if portfolio_var != 0 else 0.0
            
        return contributions
    
    def calculate_risk_decomposition(self, component_results: Dict[str, np.ndarray], 
                                   weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk decomposition of portfolio.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            weights: Dictionary mapping component names to weights
            
        Returns:
            Dictionary mapping component names to risk contribution values
        """
        # Extract last time step values for each component
        last_step_values = {}
        for component_name, component_results_array in component_results.items():
            last_step_values[component_name] = component_results_array[:, -1]
            
        # Create DataFrame
        df = pd.DataFrame(last_step_values)
        
        # Calculate covariance matrix
        cov_matrix = df.cov()
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        for i, component_i in enumerate(weights):
            for j, component_j in enumerate(weights):
                portfolio_variance += weights[component_i] * weights[component_j] * cov_matrix.iloc[i, j]
                
        # Calculate risk contributions
        risk_contributions = {}
        
        for i, component_i in enumerate(weights):
            contribution = 0.0
            for j, component_j in enumerate(weights):
                contribution += weights[component_i] * weights[component_j] * cov_matrix.iloc[i, j]
            
            # Normalize by portfolio variance
            risk_contributions[component_i] = contribution / portfolio_variance if portfolio_variance != 0 else 0.0
            
        return risk_contributions
    
    def calculate_diversification_benefit(self, component_results: Dict[str, np.ndarray], 
                                        weights: Dict[str, float]) -> float:
        """
        Calculate diversification benefit of portfolio.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            weights: Dictionary mapping component names to weights
            
        Returns:
            Diversification benefit value
        """
        # Calculate portfolio value
        portfolio_values = self.calculate_portfolio_value(component_results, weights)
        
        # Calculate portfolio VaR
        portfolio_var = self.risk_metrics_calculator.calculate_var(portfolio_values[:, -1], 0.95)
        
        # Calculate weighted sum of component VaRs
        weighted_sum_var = 0.0
        for component_name, component_results_array in component_results.items():
            component_var = self.risk_metrics_calculator.calculate_var(component_results_array[:, -1], 0.95)
            weighted_sum_var += weights.get(component_name, 0.0) * component_var
            
        # Calculate diversification benefit
        diversification_benefit = 1.0 - (portfolio_var / weighted_sum_var) if weighted_sum_var != 0 else 0.0
        
        return diversification_benefit
    
    def calculate_optimal_weights(self, component_results: Dict[str, np.ndarray], 
                                objective: str = 'min_variance',
                                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            objective: Optimization objective ('min_variance', 'max_sharpe', 'min_var', 'equal_risk_contribution')
            constraints: Dictionary of constraints (optional)
            
        Returns:
            Dictionary mapping component names to optimal weights
        """
        # Extract last time step values for each component
        last_step_values = {}
        for component_name, component_results_array in component_results.items():
            last_step_values[component_name] = component_results_array[:, -1]
            
        # Create DataFrame
        df = pd.DataFrame(last_step_values)
        
        # Calculate mean returns and covariance matrix
        mean_returns = df.mean()
        cov_matrix = df.cov()
        
        # Get component names
        component_names = list(component_results.keys())
        num_components = len(component_names)
        
        # Set default constraints if not provided
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'sum_weights': 1.0
            }
            
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weights = constraints.get('sum_weights', 1.0)
        
        # Define optimization problem
        from scipy.optimize import minimize
        
        if objective == 'min_variance':
            # Minimize portfolio variance
            def objective_function(weights):
                weights_array = np.array(weights)
                portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
                return portfolio_variance
            
            # Constraint: sum of weights equals sum_weights
            def constraint_sum(weights):
                return np.sum(weights) - sum_weights
            
            constraints_list = [{'type': 'eq', 'fun': constraint_sum}]
            
            # Bounds: min_weight <= weight <= max_weight
            bounds = [(min_weight, max_weight) for _ in range(num_components)]
            
            # Initial guess: equal weights
            initial_weights = np.full(num_components, sum_weights / num_components)
            
            # Solve optimization problem
            result = minimize(objective_function, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints_list)
            
            optimal_weights = result.x
            
        elif objective == 'max_sharpe':
            # Maximize Sharpe ratio
            def objective_function(weights):
                weights_array = np.array(weights)
                portfolio_return = np.sum(mean_returns.values * weights_array)
                portfolio_volatility = np.sqrt(weights_array.T @ cov_matrix.values @ weights_array)
                return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
            
            # Constraint: sum of weights equals sum_weights
            def constraint_sum(weights):
                return np.sum(weights) - sum_weights
            
            constraints_list = [{'type': 'eq', 'fun': constraint_sum}]
            
            # Bounds: min_weight <= weight <= max_weight
            bounds = [(min_weight, max_weight) for _ in range(num_components)]
            
            # Initial guess: equal weights
            initial_weights = np.full(num_components, sum_weights / num_components)
            
            # Solve optimization problem
            result = minimize(objective_function, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints_list)
            
            optimal_weights = result.x
            
        elif objective == 'min_var':
            # Minimize Value at Risk (VaR)
            def objective_function(weights):
                weights_array = np.array(weights)
                portfolio_values = np.zeros(df.shape[0])
                
                for i, component_name in enumerate(component_names):
                    portfolio_values += weights_array[i] * df[component_name].values
                    
                portfolio_var = np.percentile(portfolio_values, 5)  # 95% VaR
                return portfolio_var
            
            # Constraint: sum of weights equals sum_weights
            def constraint_sum(weights):
                return np.sum(weights) - sum_weights
            
            constraints_list = [{'type': 'eq', 'fun': constraint_sum}]
            
            # Bounds: min_weight <= weight <= max_weight
            bounds = [(min_weight, max_weight) for _ in range(num_components)]
            
            # Initial guess: equal weights
            initial_weights = np.full(num_components, sum_weights / num_components)
            
            # Solve optimization problem
            result = minimize(objective_function, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints_list)
            
            optimal_weights = result.x
            
        elif objective == 'equal_risk_contribution':
            # Equal Risk Contribution (ERC) portfolio
            def objective_function(weights):
                weights_array = np.array(weights)
                portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
                
                # Calculate risk contributions
                marginal_contributions = cov_matrix.values @ weights_array
                risk_contributions = weights_array * marginal_contributions
                
                # Calculate sum of squared differences between risk contributions
                target_risk_contribution = portfolio_variance / num_components
                sum_squared_diff = np.sum((risk_contributions - target_risk_contribution)**2)
                
                return sum_squared_diff
            
            # Constraint: sum of weights equals sum_weights
            def constraint_sum(weights):
                return np.sum(weights) - sum_weights
            
            constraints_list = [{'type': 'eq', 'fun': constraint_sum}]
            
            # Bounds: min_weight <= weight <= max_weight
            bounds = [(min_weight, max_weight) for _ in range(num_components)]
            
            # Initial guess: equal weights
            initial_weights = np.full(num_components, sum_weights / num_components)
            
            # Solve optimization problem
            result = minimize(objective_function, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints_list)
            
            optimal_weights = result.x
            
        else:
            raise ValueError(f"Unsupported objective: {objective}")
            
        # Create dictionary of optimal weights
        optimal_weights_dict = {component_name: weight for component_name, weight in zip(component_names, optimal_weights)}
        
        return optimal_weights_dict
    
    def plot_efficient_frontier(self, component_results: Dict[str, np.ndarray], 
                              num_portfolios: int = 100,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot efficient frontier.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            num_portfolios: Number of portfolios to generate (default: 100)
            figsize: Figure size (default: (10, 6))
            
        Returns:
            Matplotlib figure object
        """
        # Extract last time step values for each component
        last_step_values = {}
        for component_name, component_results_array in component_results.items():
            last_step_values[component_name] = component_results_array[:, -1]
            
        # Create DataFrame
        df = pd.DataFrame(last_step_values)
        
        # Calculate mean returns and covariance matrix
        mean_returns = df.mean()
        cov_matrix = df.cov()
        
        # Get component names
        component_names = list(component_results.keys())
        num_components = len(component_names)
        
        # Generate random portfolios
        np.random.seed(42)  # For reproducibility
        
        all_weights = np.zeros((num_portfolios, num_components))
        returns = np.zeros(num_portfolios)
        volatilities = np.zeros(num_portfolios)
        sharpe_ratios = np.zeros(num_portfolios)
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_components)
            weights /= np.sum(weights)
            all_weights[i, :] = weights
            
            # Calculate portfolio return
            returns[i] = np.sum(mean_returns.values * weights)
            
            # Calculate portfolio volatility
            volatilities[i] = np.sqrt(weights.T @ cov_matrix.values @ weights)
            
            # Calculate Sharpe ratio
            sharpe_ratios[i] = returns[i] / volatilities[i] if volatilities[i] > 0 else 0.0
            
        # Find portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_return = returns[max_sharpe_idx]
        max_sharpe_volatility = volatilities[max_sharpe_idx]
        
        # Find portfolio with minimum volatility
        min_vol_idx = np.argmin(volatilities)
        min_vol_return = returns[min_vol_idx]
        min_vol_volatility = volatilities[min_vol_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot random portfolios
        scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
        
        # Plot maximum Sharpe ratio portfolio
        ax.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200, label='Maximum Sharpe Ratio')
        
        # Plot minimum volatility portfolio
        ax.scatter(min_vol_volatility, min_vol_return, marker='*', color='g', s=200, label='Minimum Volatility')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Add labels and title
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')
        ax.set_title('Efficient Frontier')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_risk_contributions(self, component_results: Dict[str, np.ndarray], 
                              weights: Dict[str, float],
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot risk contributions.
        
        Args:
            component_results: Dictionary mapping component names to arrays of results
            weights: Dictionary mapping component names to weights
            figsize: Figure size (default: (10, 6))
            
        Returns:
            Matplotlib figure object
        """
        # Calculate risk decomposition
        risk_contributions = self.calculate_risk_decomposition(component_results, weights)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot risk contributions
        components = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())
        
        ax.bar(components, contributions)
        
        # Add labels and title
        ax.set_xlabel('Component')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Portfolio Risk Decomposition')
        ax.grid(True)
        
        # Rotate x-axis labels if many components
        if len(components) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        # Add weight annotations
        for i, component in enumerate(components):
            ax.annotate(f'Weight: {weights[component]:.2f}',
                       (i, contributions[i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
            
        # Adjust layout
        plt.tight_layout()
        
        return fig
