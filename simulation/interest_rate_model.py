"""
Interest rate simulation module using the Heath-Jarrow-Morton (HJM) model.
This module provides functionality to simulate interest rate paths for multiple rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import norm
import matplotlib.pyplot as plt


class HJMModel:
    """
    Heath-Jarrow-Morton (HJM) model for interest rate simulation.
    
    This class implements the HJM model for simulating consistent interest rate
    paths across the yield curve, ensuring no-arbitrage conditions are satisfied.
    """
    
    def __init__(self, rate_tenors: List[str], initial_rates: Dict[str, float],
                volatility_params: Dict[str, float], mean_reversion_params: Dict[str, float],
                correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize the HJM model.
        
        Args:
            rate_tenors: List of rate tenors to simulate (e.g., ['sofr_rate', 'treasury_1y'])
            initial_rates: Dictionary mapping rate tenors to initial rate values
            volatility_params: Dictionary mapping rate tenors to volatility parameters
            mean_reversion_params: Dictionary mapping rate tenors to mean reversion parameters
            correlation_matrix: Correlation matrix between rate innovations (optional)
        """
        self.rate_tenors = rate_tenors
        self.num_rates = len(rate_tenors)
        
        # Validate inputs
        for tenor in rate_tenors:
            if tenor not in initial_rates:
                raise ValueError(f"Initial rate not provided for tenor: {tenor}")
            if tenor not in volatility_params:
                raise ValueError(f"Volatility parameter not provided for tenor: {tenor}")
            if tenor not in mean_reversion_params:
                raise ValueError(f"Mean reversion parameter not provided for tenor: {tenor}")
        
        self.initial_rates = {tenor: initial_rates[tenor] for tenor in rate_tenors}
        self.volatility_params = {tenor: volatility_params[tenor] for tenor in rate_tenors}
        self.mean_reversion_params = {tenor: mean_reversion_params[tenor] for tenor in rate_tenors}
        
        # Set up correlation matrix
        if correlation_matrix is None:
            # Default to identity matrix (uncorrelated)
            self.correlation_matrix = np.eye(self.num_rates)
        else:
            # Validate correlation matrix dimensions
            if correlation_matrix.shape != (self.num_rates, self.num_rates):
                raise ValueError(f"Correlation matrix must be {self.num_rates}x{self.num_rates}")
            self.correlation_matrix = correlation_matrix
            
        # Compute Cholesky decomposition for correlated random numbers
        try:
            self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, use nearest positive definite matrix
            print("Warning: Correlation matrix is not positive definite. Using nearest approximation.")
            # Simple adjustment to make it positive definite
            eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            self.correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
    
    def simulate_rates(self, num_scenarios: int, time_steps: int, time_horizon: float,
                      dt: float = 1/252) -> Dict[str, np.ndarray]:
        """
        Simulate interest rate paths using the HJM model.
        
        Args:
            num_scenarios: Number of scenarios to simulate
            time_steps: Number of time steps in the simulation
            time_horizon: Time horizon in years
            dt: Time step size in years (default: 1/252 for daily steps)
            
        Returns:
            Dictionary mapping rate tenors to simulated rate paths
            Each path is a 2D array of shape (num_scenarios, time_steps+1)
        """
        # Validate inputs
        if time_steps <= 0:
            raise ValueError("Number of time steps must be positive")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if dt <= 0:
            raise ValueError("Time step size must be positive")
        if num_scenarios <= 0:
            raise ValueError("Number of scenarios must be positive")
            
        # Initialize rate paths
        rate_paths = {}
        for tenor in self.rate_tenors:
            # Initialize with initial rates
            paths = np.zeros((num_scenarios, time_steps + 1))
            paths[:, 0] = self.initial_rates[tenor]
            rate_paths[tenor] = paths
            
        # Generate correlated random numbers for all scenarios and time steps
        # Shape: (num_scenarios, time_steps, num_rates)
        random_numbers = np.random.normal(0, 1, (num_scenarios, time_steps, self.num_rates))
        
        # Apply correlation structure
        for i in range(num_scenarios):
            for j in range(time_steps):
                random_numbers[i, j] = self.cholesky_matrix @ random_numbers[i, j]
                
        # Simulate rate paths
        for t in range(time_steps):
            for i, tenor in enumerate(self.rate_tenors):
                # Extract parameters
                current_rates = rate_paths[tenor][:, t]
                volatility = self.volatility_params[tenor]
                mean_reversion = self.mean_reversion_params[tenor]
                
                # HJM drift adjustment to ensure no-arbitrage
                drift_adjustment = 0.5 * volatility**2 * (1 - np.exp(-2 * mean_reversion * dt)) / mean_reversion
                
                # Mean-reverting drift term
                drift = mean_reversion * (self.initial_rates[tenor] - current_rates) * dt
                
                # Volatility term
                vol_term = volatility * np.sqrt(dt) * random_numbers[:, t, i]
                
                # Update rates
                new_rates = current_rates + drift + drift_adjustment + vol_term
                
                # Ensure rates are non-negative (optional, can be removed if negative rates are allowed)
                new_rates = np.maximum(new_rates, 0)
                
                # Store new rates
                rate_paths[tenor][:, t+1] = new_rates
                
        return rate_paths
    
    def plot_rate_paths(self, rate_paths: Dict[str, np.ndarray], num_paths_to_plot: int = 10,
                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot simulated rate paths.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            num_paths_to_plot: Number of paths to plot (default: 10)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, axes = plt.subplots(len(self.rate_tenors), 1, figsize=figsize, sharex=True)
        
        # If only one rate tenor, wrap axes in a list
        if len(self.rate_tenors) == 1:
            axes = [axes]
            
        # Plot paths for each rate tenor
        for i, tenor in enumerate(self.rate_tenors):
            paths = rate_paths[tenor]
            num_scenarios, num_steps = paths.shape
            
            # Select random paths to plot
            if num_scenarios > num_paths_to_plot:
                indices = np.random.choice(num_scenarios, num_paths_to_plot, replace=False)
            else:
                indices = np.arange(num_scenarios)
                
            # Plot selected paths
            for idx in indices:
                axes[i].plot(paths[idx], alpha=0.5, linewidth=1)
                
            # Plot mean path
            mean_path = np.mean(paths, axis=0)
            axes[i].plot(mean_path, color='black', linewidth=2, label='Mean')
            
            # Add labels and title
            axes[i].set_ylabel(f'{tenor} (%)')
            axes[i].set_title(f'{tenor} Simulated Paths')
            axes[i].grid(True)
            
        # Add x-axis label to bottom subplot
        axes[-1].set_xlabel('Time Step')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def calculate_rate_statistics(self, rate_paths: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate statistics for simulated rate paths.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            
        Returns:
            Dictionary mapping rate tenors to dictionaries of statistics
            Each statistics dictionary contains 'mean', 'std', 'min', 'max', 'median', 'percentile_5', 'percentile_95'
        """
        statistics = {}
        
        for tenor in self.rate_tenors:
            paths = rate_paths[tenor]
            
            # Calculate statistics
            stats = {
                'mean': np.mean(paths, axis=0),
                'std': np.std(paths, axis=0),
                'min': np.min(paths, axis=0),
                'max': np.max(paths, axis=0),
                'median': np.median(paths, axis=0),
                'percentile_5': np.percentile(paths, 5, axis=0),
                'percentile_95': np.percentile(paths, 95, axis=0)
            }
            
            statistics[tenor] = stats
            
        return statistics


class InterestRateSimulator:
    """
    Interest rate simulator for generating scenarios.
    
    This class provides a high-level interface for simulating interest rate
    scenarios using the HJM model and other configuration options.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the interest rate simulator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.default_config = {
            'rate_tenors': ['sofr_rate', 'treasury_1y', 'treasury_2y', 'treasury_3y', 'treasury_5y', 'treasury_10y'],
            'initial_rates': {
                'sofr_rate': 0.0525,     # 5.25%
                'treasury_1y': 0.0510,   # 5.10%
                'treasury_2y': 0.0490,   # 4.90%
                'treasury_3y': 0.0470,   # 4.70%
                'treasury_5y': 0.0450,   # 4.50%
                'treasury_10y': 0.0430   # 4.30%
            },
            'volatility_params': {
                'sofr_rate': 0.010,      # 1.0%
                'treasury_1y': 0.009,    # 0.9%
                'treasury_2y': 0.008,    # 0.8%
                'treasury_3y': 0.007,    # 0.7%
                'treasury_5y': 0.006,    # 0.6%
                'treasury_10y': 0.005    # 0.5%
            },
            'mean_reversion_params': {
                'sofr_rate': 0.15,       # Mean reversion speed
                'treasury_1y': 0.12,
                'treasury_2y': 0.10,
                'treasury_3y': 0.08,
                'treasury_5y': 0.06,
                'treasury_10y': 0.04
            },
            'correlation_matrix': np.array([
                [1.00, 0.95, 0.90, 0.85, 0.80, 0.75],  # SOFR
                [0.95, 1.00, 0.95, 0.90, 0.85, 0.80],  # 1Y
                [0.90, 0.95, 1.00, 0.95, 0.90, 0.85],  # 2Y
                [0.85, 0.90, 0.95, 1.00, 0.95, 0.90],  # 3Y
                [0.80, 0.85, 0.90, 0.95, 1.00, 0.95],  # 5Y
                [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]   # 10Y
            ]),
            'num_scenarios': 1000,
            'time_horizon': 3.0,         # 3 years
            'time_steps_per_year': 12,   # Monthly steps
            'dt': 1/12                   # Monthly time step size
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
            
        # Create HJM model
        self.hjm_model = HJMModel(
            rate_tenors=self.config['rate_tenors'],
            initial_rates=self.config['initial_rates'],
            volatility_params=self.config['volatility_params'],
            mean_reversion_params=self.config['mean_reversion_params'],
            correlation_matrix=self.config['correlation_matrix']
        )
        
    def update_config(self, new_config: Dict) -> None:
        """
        Update the simulator configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Ensure correlation matrix is a numpy array before passing to HJMModel
        if 'correlation_matrix' in self.config and isinstance(self.config['correlation_matrix'], list):
            self.config['correlation_matrix'] = np.array(self.config['correlation_matrix'])

        # Recreate HJM model with updated configuration
        self.hjm_model = HJMModel(
            rate_tenors=self.config['rate_tenors'],
            initial_rates=self.config['initial_rates'],
            volatility_params=self.config['volatility_params'],
            mean_reversion_params=self.config['mean_reversion_params'],
            correlation_matrix=self.config['correlation_matrix']
        )
        
    def simulate(self) -> Dict[str, np.ndarray]:
        """
        Simulate interest rate scenarios.
        
        Returns:
            Dictionary mapping rate tenors to simulated rate paths
        """
        # Calculate time steps
        time_steps = int(self.config['time_horizon'] * self.config['time_steps_per_year'])
        
        # Simulate rate paths
        rate_paths = self.hjm_model.simulate_rates(
            num_scenarios=self.config['num_scenarios'],
            time_steps=time_steps,
            time_horizon=self.config['time_horizon'],
            dt=self.config['dt']
        )
        
        return rate_paths
    
    def plot_simulations(self, rate_paths: Optional[Dict[str, np.ndarray]] = None,
                        num_paths_to_plot: int = 10) -> plt.Figure:
        """
        Plot simulated rate paths.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths (optional)
            num_paths_to_plot: Number of paths to plot (default: 10)
            
        Returns:
            Matplotlib figure object
        """
        if rate_paths is None:
            rate_paths = self.simulate()
            
        return self.hjm_model.plot_rate_paths(rate_paths, num_paths_to_plot)
    
    def get_statistics(self, rate_paths: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate statistics for simulated rate paths.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths (optional)
            
        Returns:
            Dictionary mapping rate tenors to dictionaries of statistics
        """
        if rate_paths is None:
            rate_paths = self.simulate()
            
        return self.hjm_model.calculate_rate_statistics(rate_paths)
    
    def save_simulations(self, rate_paths: Dict[str, np.ndarray], file_path: str) -> None:
        """
        Save simulated rate paths to a file.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            file_path: Path to save the file
        """
        # Convert to dictionary of lists for easier serialization
        save_data = {
            'config': self.config,
            'rate_paths': {tenor: paths.tolist() for tenor, paths in rate_paths.items()}
        }
        
        # Save to file
        np.save(file_path, save_data, allow_pickle=True)
        
    def load_simulations(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load simulated rate paths from a file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            Dictionary mapping rate tenors to simulated rate paths
        """
        # Load from file
        load_data = np.load(file_path, allow_pickle=True).item()
        
        # Update configuration
        self.update_config(load_data['config'])
        
        # Convert back to numpy arrays
        rate_paths = {tenor: np.array(paths) for tenor, paths in load_data['rate_paths'].items()}
        
        return rate_paths
    
    def convert_to_dataframe(self, rate_paths: Dict[str, np.ndarray], 
                           scenario_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Convert simulated rate paths to a pandas DataFrame.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            scenario_ids: List of scenario IDs (optional)
            
        Returns:
            DataFrame with columns for scenario_id, time_step, and each rate tenor
        """
        # Get dimensions
        num_scenarios, num_steps = next(iter(rate_paths.values())).shape
        
        # Create scenario IDs if not provided
        if scenario_ids is None:
            scenario_ids = list(range(num_scenarios))
        elif len(scenario_ids) != num_scenarios:
            raise ValueError(f"Length of scenario_ids ({len(scenario_ids)}) must match number of scenarios ({num_scenarios})")
            
        # Create time steps
        time_steps = list(range(num_steps))
        
        # Create empty DataFrame
        data = []
        
        # Fill DataFrame
        for s, scenario_id in enumerate(scenario_ids):
            for t in time_steps:
                row = {'scenario_id': scenario_id, 'time_step': t}
                for tenor in rate_paths:
                    row[tenor] = rate_paths[tenor][s, t]
                data.append(row)
                
        return pd.DataFrame(data)


class StressTestGenerator:
    """
    Generator for stress test scenarios.
    
    This class provides functionality to generate stress test scenarios
    for interest rates based on historical events or hypothetical scenarios.
    """
    
    def __init__(self, rate_tenors: List[str], base_rates: Dict[str, float]):
        """
        Initialize the stress test generator.
        
        Args:
            rate_tenors: List of rate tenors to generate stress tests for
            base_rates: Dictionary mapping rate tenors to base rate values
        """
        self.rate_tenors = rate_tenors
        self.base_rates = base_rates
        
        # Predefined stress scenarios
        self.predefined_scenarios = {
            'parallel_up_100bp': self._create_parallel_shift(0.01),
            'parallel_up_200bp': self._create_parallel_shift(0.02),
            'parallel_up_300bp': self._create_parallel_shift(0.03),
            'parallel_down_100bp': self._create_parallel_shift(-0.01),
            'parallel_down_200bp': self._create_parallel_shift(-0.02),
            'steepener_100bp': self._create_steepener(0.01),
            'flattener_100bp': self._create_flattener(0.01),
            'short_up_100bp': self._create_short_rate_shift(0.01),
            'long_up_100bp': self._create_long_rate_shift(0.01),
            'global_financial_crisis': self._create_gfc_scenario(),
            'taper_tantrum': self._create_taper_tantrum_scenario(),
            'covid_shock': self._create_covid_shock_scenario(),
            'inflation_surge': self._create_inflation_surge_scenario()
        }
        
    def _create_parallel_shift(self, shift: float) -> Dict[str, float]:
        """
        Create a parallel shift stress scenario.
        
        Args:
            shift: Amount to shift all rates by
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        return {tenor: self.base_rates[tenor] + shift for tenor in self.rate_tenors}
    
    def _create_steepener(self, magnitude: float) -> Dict[str, float]:
        """
        Create a steepener stress scenario.
        
        Args:
            magnitude: Magnitude of the steepening
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Map tenors to relative position on the curve (0 to 1)
        tenor_positions = {
            'sofr_rate': 0.0,
            'treasury_1y': 0.1,
            'treasury_2y': 0.2,
            'treasury_3y': 0.3,
            'treasury_5y': 0.5,
            'treasury_10y': 1.0
        }
        
        # Create steepener (short rates down, long rates up)
        stressed_rates = {}
        for tenor in self.rate_tenors:
            position = tenor_positions.get(tenor, 0.5)  # Default to middle if tenor not found
            shift = magnitude * (2 * position - 1)  # -magnitude at short end, +magnitude at long end
            stressed_rates[tenor] = self.base_rates[tenor] + shift
            
        return stressed_rates
    
    def _create_flattener(self, magnitude: float) -> Dict[str, float]:
        """
        Create a flattener stress scenario.
        
        Args:
            magnitude: Magnitude of the flattening
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Map tenors to relative position on the curve (0 to 1)
        tenor_positions = {
            'sofr_rate': 0.0,
            'treasury_1y': 0.1,
            'treasury_2y': 0.2,
            'treasury_3y': 0.3,
            'treasury_5y': 0.5,
            'treasury_10y': 1.0
        }
        
        # Create flattener (short rates up, long rates down)
        stressed_rates = {}
        for tenor in self.rate_tenors:
            position = tenor_positions.get(tenor, 0.5)  # Default to middle if tenor not found
            shift = magnitude * (1 - 2 * position)  # +magnitude at short end, -magnitude at long end
            stressed_rates[tenor] = self.base_rates[tenor] + shift
            
        return stressed_rates
    
    def _create_short_rate_shift(self, shift: float) -> Dict[str, float]:
        """
        Create a short rate shift stress scenario.
        
        Args:
            shift: Amount to shift short rates by
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Map tenors to relative position on the curve (0 to 1)
        tenor_positions = {
            'sofr_rate': 0.0,
            'treasury_1y': 0.1,
            'treasury_2y': 0.2,
            'treasury_3y': 0.3,
            'treasury_5y': 0.5,
            'treasury_10y': 1.0
        }
        
        # Create short rate shift (larger shift at short end, diminishing to long end)
        stressed_rates = {}
        for tenor in self.rate_tenors:
            position = tenor_positions.get(tenor, 0.5)  # Default to middle if tenor not found
            factor = max(0, 1 - 2 * position)  # 1 at short end, 0 at position 0.5 and beyond
            stressed_rates[tenor] = self.base_rates[tenor] + shift * factor
            
        return stressed_rates
    
    def _create_long_rate_shift(self, shift: float) -> Dict[str, float]:
        """
        Create a long rate shift stress scenario.
        
        Args:
            shift: Amount to shift long rates by
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Map tenors to relative position on the curve (0 to 1)
        tenor_positions = {
            'sofr_rate': 0.0,
            'treasury_1y': 0.1,
            'treasury_2y': 0.2,
            'treasury_3y': 0.3,
            'treasury_5y': 0.5,
            'treasury_10y': 1.0
        }
        
        # Create long rate shift (larger shift at long end, diminishing to short end)
        stressed_rates = {}
        for tenor in self.rate_tenors:
            position = tenor_positions.get(tenor, 0.5)  # Default to middle if tenor not found
            factor = min(1, 2 * position)  # 0 at position 0, 1 at position 0.5 and beyond
            stressed_rates[tenor] = self.base_rates[tenor] + shift * factor
            
        return stressed_rates
    
    def _create_gfc_scenario(self) -> Dict[str, float]:
        """
        Create a Global Financial Crisis-like stress scenario.
        
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # GFC was characterized by short rates near zero and long rates falling
        stressed_rates = {}
        for tenor in self.rate_tenors:
            if tenor in ['sofr_rate', 'treasury_1y']:
                stressed_rates[tenor] = 0.001  # Near zero
            elif tenor == 'treasury_2y':
                stressed_rates[tenor] = 0.005  # 0.5%
            elif tenor == 'treasury_3y':
                stressed_rates[tenor] = 0.010  # 1.0%
            elif tenor == 'treasury_5y':
                stressed_rates[tenor] = 0.020  # 2.0%
            elif tenor == 'treasury_10y':
                stressed_rates[tenor] = 0.030  # 3.0%
            else:
                stressed_rates[tenor] = self.base_rates[tenor] * 0.5  # 50% of base rate
                
        return stressed_rates
    
    def _create_taper_tantrum_scenario(self) -> Dict[str, float]:
        """
        Create a Taper Tantrum-like stress scenario.
        
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Taper Tantrum was characterized by a sharp rise in long-term rates
        stressed_rates = {}
        for tenor in self.rate_tenors:
            if tenor in ['sofr_rate', 'treasury_1y']:
                stressed_rates[tenor] = self.base_rates[tenor] + 0.005  # +50bp
            elif tenor == 'treasury_2y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.010  # +100bp
            elif tenor == 'treasury_3y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.015  # +150bp
            elif tenor == 'treasury_5y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.020  # +200bp
            elif tenor == 'treasury_10y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.025  # +250bp
            else:
                stressed_rates[tenor] = self.base_rates[tenor] * 1.5  # 150% of base rate
                
        return stressed_rates
    
    def _create_covid_shock_scenario(self) -> Dict[str, float]:
        """
        Create a COVID-19 shock-like stress scenario.
        
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # COVID shock was characterized by rates falling across the curve
        stressed_rates = {}
        for tenor in self.rate_tenors:
            if tenor in ['sofr_rate', 'treasury_1y']:
                stressed_rates[tenor] = 0.001  # Near zero
            elif tenor == 'treasury_2y':
                stressed_rates[tenor] = 0.002  # 0.2%
            elif tenor == 'treasury_3y':
                stressed_rates[tenor] = 0.003  # 0.3%
            elif tenor == 'treasury_5y':
                stressed_rates[tenor] = 0.005  # 0.5%
            elif tenor == 'treasury_10y':
                stressed_rates[tenor] = 0.007  # 0.7%
            else:
                stressed_rates[tenor] = self.base_rates[tenor] * 0.2  # 20% of base rate
                
        return stressed_rates
    
    def _create_inflation_surge_scenario(self) -> Dict[str, float]:
        """
        Create an inflation surge stress scenario.
        
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Inflation surge characterized by rates rising across the curve, with more impact on short-term rates
        stressed_rates = {}
        for tenor in self.rate_tenors:
            if tenor in ['sofr_rate', 'treasury_1y']:
                stressed_rates[tenor] = self.base_rates[tenor] + 0.030  # +300bp
            elif tenor == 'treasury_2y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.025  # +250bp
            elif tenor == 'treasury_3y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.020  # +200bp
            elif tenor == 'treasury_5y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.015  # +150bp
            elif tenor == 'treasury_10y':
                stressed_rates[tenor] = self.base_rates[tenor] + 0.010  # +100bp
            else:
                stressed_rates[tenor] = self.base_rates[tenor] * 1.5  # 150% of base rate
                
        return stressed_rates
    
    def get_predefined_scenario(self, scenario_name: str) -> Dict[str, float]:
        """
        Get a predefined stress scenario.
        
        Args:
            scenario_name: Name of the predefined scenario
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        if scenario_name not in self.predefined_scenarios:
            raise ValueError(f"Predefined scenario '{scenario_name}' not found")
            
        return self.predefined_scenarios[scenario_name]
    
    def list_predefined_scenarios(self) -> List[str]:
        """
        List all predefined stress scenarios.
        
        Returns:
            List of predefined scenario names
        """
        return list(self.predefined_scenarios.keys())
    
    def create_custom_scenario(self, scenario_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Create a custom stress scenario.
        
        Args:
            scenario_dict: Dictionary mapping rate tenors to stressed rate values
            
        Returns:
            Dictionary mapping rate tenors to stressed rate values
        """
        # Validate that all required tenors are present
        for tenor in self.rate_tenors:
            if tenor not in scenario_dict:
                raise ValueError(f"Tenor '{tenor}' not found in custom scenario")
                
        return {tenor: scenario_dict[tenor] for tenor in self.rate_tenors}
    
    def generate_all_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Generate all predefined stress scenarios.
        
        Returns:
            Dictionary mapping scenario names to dictionaries of stressed rates
        """
        return self.predefined_scenarios.copy()
    
    def convert_to_dataframe(self, scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Convert stress scenarios to a pandas DataFrame.
        
        Args:
            scenarios: Dictionary mapping scenario names to dictionaries of stressed rates (optional)
            
        Returns:
            DataFrame with columns for scenario_name and each rate tenor
        """
        if scenarios is None:
            scenarios = self.generate_all_scenarios()
            
        # Create empty DataFrame
        data = []
        
        # Fill DataFrame
        for scenario_name, rates in scenarios.items():
            row = {'scenario_name': scenario_name}
            row.update(rates)
            data.append(row)
                
        return pd.DataFrame(data)
