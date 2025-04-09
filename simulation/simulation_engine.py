"""
Simulation engine for running EVE and EVS models with interest rate scenarios.
This module connects regression models with interest rate simulations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json

# Import from other modules
from models.base_model import BaseRegressionModel, EVEModel, EVSModel
from models.model_factory import ModelFactory
from simulation.interest_rate_model import InterestRateSimulator, StressTestGenerator

# Top-level worker function for parallel simulation (to avoid pickling issues)
def _simulation_worker(model_factory: ModelFactory, model_name: str, scenario_inputs: List[np.ndarray]) -> Tuple[str, np.ndarray]:
    """Worker function to run a model for given scenarios."""
    model = model_factory.get_model(model_name)
    scenario_results = []
    
    for X in scenario_inputs:
        # Run model for this scenario
        predictions = model.predict(X)
        scenario_results.append(predictions)
        
    return model_name, np.array(scenario_results)



class SimulationEngine:
    """
    Engine for running simulations with EVE and EVS models.
    
    This class connects regression models with interest rate simulations
    to generate results for analysis.
    """
    
    def __init__(self, model_factory: ModelFactory, rate_simulator: InterestRateSimulator):
        """
        Initialize the simulation engine.
        
        Args:
            model_factory: Factory containing regression models
            rate_simulator: Simulator for generating interest rate scenarios
        """
        self.model_factory = model_factory
        self.rate_simulator = rate_simulator
        self.results = {}
        self.stress_test_results = {}
        
    def prepare_model_inputs(self, rate_paths: Dict[str, np.ndarray], 
                           additional_factors: Optional[Dict[str, np.ndarray]] = None) -> List[pd.DataFrame]:
        """
        Prepare inputs for regression models from rate paths.
        
        Args:
            rate_paths: Dictionary mapping rate tenors to simulated rate paths
            additional_factors: Dictionary mapping factor names to factor values (optional)
            
        Returns:
            List of DataFrames containing model inputs for each scenario
        """
        # Get dimensions
        num_scenarios, num_steps = next(iter(rate_paths.values())).shape
        
        # Create list of DataFrames (one per scenario)
        inputs = []
        
        for s in range(num_scenarios):
            # Create DataFrame for this scenario
            scenario_data = {}
            
            # Add rate paths
            for tenor, paths in rate_paths.items():
                scenario_data[tenor] = paths[s, :]
                
            # Add additional factors if provided
            if additional_factors is not None:
                for factor, values in additional_factors.items():
                    if isinstance(values, np.ndarray):
                        if values.shape[0] == num_scenarios:
                            # Scenario-specific constant value
                            scenario_data[factor] = np.full(num_steps, values[s])
                        elif values.shape == (num_scenarios, num_steps):
                            # Scenario and time-specific values
                            scenario_data[factor] = values[s, :]
                        else:
                            # Constant value for all scenarios
                            scenario_data[factor] = np.full(num_steps, values[0])
                    else:
                        # Constant value
                        scenario_data[factor] = np.full(num_steps, values)
            
            # Create DataFrame
            inputs.append(pd.DataFrame(scenario_data))
            
        return inputs
    
    def run_simulation(self, model_names: List[str], 
                      rate_paths: Optional[Dict[str, np.ndarray]] = None,
                      additional_factors: Optional[Dict[str, Any]] = None,
                      parallel: bool = True, num_processes: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run simulation with specified models and rate paths.
        
        Args:
            model_names: List of model names to run
            rate_paths: Dictionary mapping rate tenors to simulated rate paths (optional)
            additional_factors: Dictionary mapping factor names to factor values (optional)
            parallel: Whether to use parallel processing (default: True)
            num_processes: Number of processes to use (default: None, uses CPU count)
            
        Returns:
            Dictionary mapping model names to dictionaries of results
        """
        # Generate rate paths if not provided
        if rate_paths is None:
            rate_paths = self.rate_simulator.simulate()
            
        # Get dimensions
        num_scenarios, num_steps = next(iter(rate_paths.values())).shape
        
        # Prepare model inputs
        inputs = self.prepare_model_inputs(rate_paths, additional_factors)
        
        # Initialize results dictionary
        results = {}
        
        # Run models
        if parallel and num_scenarios > 1:
            # Use parallel processing
            if num_processes is None:
                num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
                # Worker function is now defined at the top level (_simulation_worker)
                # to avoid pickling issues.
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit tasks
                futures = []
                for model_name in model_names:
                    futures.append(executor.submit(_simulation_worker, self.model_factory, model_name, inputs))
                    
                # Collect results
                for future in as_completed(futures):
                    model_name, model_results = future.result()
                    results[model_name] = model_results
        else:
            # Run sequentially
            for model_name in model_names:
                model = self.model_factory.get_model(model_name)
                model_results = []
                
                for X in inputs:
                    # Run model for this scenario
                    predictions = model.predict(X)
                    model_results.append(predictions)
                    
                results[model_name] = np.array(model_results)
                
        # Store results
        self.results = results
        
        return results
    
    def run_stress_tests(self, model_names: List[str], 
                        stress_scenarios: Dict[str, Dict[str, float]],
                        additional_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run stress tests with specified models and stress scenarios.
        
        Args:
            model_names: List of model names to run
            stress_scenarios: Dictionary mapping scenario names to dictionaries of stressed rates
            additional_factors: Dictionary mapping factor names to factor values (optional)
            
        Returns:
            Dictionary mapping scenario names to dictionaries mapping model names to results
        """
        # Initialize results dictionary
        results = {}
        
        # Run stress tests
        for scenario_name, stressed_rates in stress_scenarios.items():
            # Create DataFrame with stressed rates
            X = pd.DataFrame({tenor: [rate] for tenor, rate in stressed_rates.items()})
            
            # Add additional factors if provided
            if additional_factors is not None:
                for factor, value in additional_factors.items():
                    X[factor] = value
                    
            # Run models
            scenario_results = {}
            for model_name in model_names:
                model = self.model_factory.get_model(model_name)
                prediction = model.predict(X)[0]  # Get first (and only) prediction
                scenario_results[model_name] = prediction
                
            results[scenario_name] = scenario_results
            
        # Store results
        self.stress_test_results = results
        
        return results
    
    def calculate_statistics(self, results: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate statistics for simulation results.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            
        Returns:
            Dictionary mapping model names to dictionaries of statistics
        """
        if results is None:
            results = self.results
            
        statistics = {}
        
        for model_name, model_results in results.items():
            # Calculate statistics
            stats = {
                'mean': np.mean(model_results, axis=0),
                'std': np.std(model_results, axis=0),
                'min': np.min(model_results, axis=0),
                'max': np.max(model_results, axis=0),
                'median': np.median(model_results, axis=0),
                'percentile_5': np.percentile(model_results, 5, axis=0),
                'percentile_95': np.percentile(model_results, 95, axis=0)
            }
            
            statistics[model_name] = stats
            
        return statistics
    
    def calculate_var(self, results: Optional[Dict[str, np.ndarray]] = None, 
                     confidence_level: float = 0.95, time_step: int = -1) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for simulation results.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            time_step: Time step to calculate VaR for (default: -1, last step)
            
        Returns:
            Dictionary mapping model names to VaR values
        """
        if results is None:
            results = self.results
            
        var_values = {}
        
        for model_name, model_results in results.items():
            # Extract results for specified time step
            if time_step == -1:
                time_step_results = model_results[:, -1]
            else:
                time_step_results = model_results[:, time_step]
                
            # Calculate VaR
            var = np.percentile(time_step_results, 100 * (1 - confidence_level))
            var_values[model_name] = var
            
        return var_values
    
    def calculate_expected_shortfall(self, results: Optional[Dict[str, np.ndarray]] = None, 
                                   confidence_level: float = 0.95, time_step: int = -1) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (ES) for simulation results.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            confidence_level: Confidence level for ES calculation (default: 0.95)
            time_step: Time step to calculate ES for (default: -1, last step)
            
        Returns:
            Dictionary mapping model names to ES values
        """
        if results is None:
            results = self.results
            
        es_values = {}
        
        for model_name, model_results in results.items():
            # Extract results for specified time step
            if time_step == -1:
                time_step_results = model_results[:, -1]
            else:
                time_step_results = model_results[:, time_step]
                
            # Calculate VaR
            var = np.percentile(time_step_results, 100 * (1 - confidence_level))
            
            # Calculate ES
            tail_values = time_step_results[time_step_results <= var]
            es = np.mean(tail_values) if len(tail_values) > 0 else var
            es_values[model_name] = es
            
        return es_values
    
    def calculate_sensitivity(self, model_names: List[str], factor_name: str,
                            base_value: float, delta: float = 0.01) -> Dict[str, float]:
        """
        Calculate sensitivity of models to a specific factor.
        
        Args:
            model_names: List of model names to calculate sensitivity for
            factor_name: Name of the factor to analyze sensitivity for
            base_value: Base value of the factor
            delta: Change in factor value for sensitivity calculation
            
        Returns:
            Dictionary mapping model names to sensitivity values
        """
        # Create base scenario
        base_scenario = {factor_name: base_value}
        
        # Create shocked scenario
        shocked_scenario = {factor_name: base_value + delta}
        
        # Run models with base scenario
        base_results = {}
        for model_name in model_names:
            model = self.model_factory.get_model(model_name)
            X = pd.DataFrame(base_scenario, index=[0])
            base_results[model_name] = model.predict(X)[0]
            
        # Run models with shocked scenario
        shocked_results = {}
        for model_name in model_names:
            model = self.model_factory.get_model(model_name)
            X = pd.DataFrame(shocked_scenario, index=[0])
            shocked_results[model_name] = model.predict(X)[0]
            
        # Calculate sensitivities
        sensitivities = {}
        for model_name in model_names:
            sensitivity = (shocked_results[model_name] - base_results[model_name]) / delta
            sensitivities[model_name] = sensitivity
            
        return sensitivities
    
    def calculate_sensitivity_grid(self, model_names: List[str], factor_name: str,
                                 base_value: float, deltas: List[float]) -> Dict[str, List[float]]:
        """
        Calculate sensitivity grid of models to a specific factor.
        
        Args:
            model_names: List of model names to calculate sensitivity for
            factor_name: Name of the factor to analyze sensitivity for
            base_value: Base value of the factor
            deltas: List of changes in factor value for sensitivity calculation
            
        Returns:
            Dictionary mapping model names to lists of sensitivity values
        """
        # Initialize results
        sensitivity_grid = {model_name: [] for model_name in model_names}
        
        # Calculate sensitivity for each delta
        for delta in deltas:
            sensitivities = self.calculate_sensitivity(model_names, factor_name, base_value, delta)
            
            for model_name, sensitivity in sensitivities.items():
                sensitivity_grid[model_name].append(sensitivity)
                
        return sensitivity_grid
    
    def plot_results(self, results: Optional[Dict[str, np.ndarray]] = None,
                   model_names: Optional[List[str]] = None,
                   num_scenarios_to_plot: int = 10,
                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot simulation results.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            model_names: List of model names to plot (optional)
            num_scenarios_to_plot: Number of scenarios to plot (default: 10)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        if results is None:
            results = self.results
            
        if model_names is None:
            model_names = list(results.keys())
            
        # Create figure
        fig, axes = plt.subplots(len(model_names), 1, figsize=figsize, sharex=True)
        
        # If only one model, wrap axes in a list
        if len(model_names) == 1:
            axes = [axes]
            
        # Plot results for each model
        for i, model_name in enumerate(model_names):
            if model_name not in results:
                continue
                
            model_results = results[model_name]
            num_scenarios, num_steps = model_results.shape
            
            # Select random scenarios to plot
            if num_scenarios > num_scenarios_to_plot:
                indices = np.random.choice(num_scenarios, num_scenarios_to_plot, replace=False)
            else:
                indices = np.arange(num_scenarios)
                
            # Plot selected scenarios
            for idx in indices:
                axes[i].plot(model_results[idx], alpha=0.5, linewidth=1)
                
            # Plot mean, median, and percentiles
            mean = np.mean(model_results, axis=0)
            median = np.median(model_results, axis=0)
            percentile_5 = np.percentile(model_results, 5, axis=0)
            percentile_95 = np.percentile(model_results, 95, axis=0)
            
            axes[i].plot(mean, color='black', linewidth=2, label='Mean')
            axes[i].plot(median, color='blue', linewidth=2, label='Median')
            axes[i].plot(percentile_5, color='red', linewidth=2, label='5th Percentile')
            axes[i].plot(percentile_95, color='green', linewidth=2, label='95th Percentile')
            
            # Add labels and title
            axes[i].set_ylabel(f'{model_name}')
            axes[i].set_title(f'{model_name} Simulation Results')
            axes[i].grid(True)
            axes[i].legend()
            
        # Add x-axis label to bottom subplot
        axes[-1].set_xlabel('Time Step')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_histogram(self, results: Optional[Dict[str, np.ndarray]] = None,
                     model_names: Optional[List[str]] = None,
                     time_step: int = -1, bins: int = 50,
                     figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot histogram of simulation results.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            model_names: List of model names to plot (optional)
            time_step: Time step to plot histogram for (default: -1, last step)
            bins: Number of bins for histogram (default: 50)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        if results is None:
            results = self.results
            
        if model_names is None:
            model_names = list(results.keys())
            
        # Create figure
        fig, axes = plt.subplots(len(model_names), 1, figsize=figsize)
        
        # If only one model, wrap axes in a list
        if len(model_names) == 1:
            axes = [axes]
            
        # Plot histogram for each model
        for i, model_name in enumerate(model_names):
            if model_name not in results:
                continue
                
            model_results = results[model_name]
            
            # Extract results for specified time step
            if time_step == -1:
                time_step_results = model_results[:, -1]
            else:
                time_step_results = model_results[:, time_step]
                
            # Plot histogram
            axes[i].hist(time_step_results, bins=bins, alpha=0.7)
            
            # Add VaR and ES lines
            var_95 = np.percentile(time_step_results, 5)
            var_99 = np.percentile(time_step_results, 1)
            
            # Calculate ES
            es_95_values = time_step_results[time_step_results <= var_95]
            es_99_values = time_step_results[time_step_results <= var_99]
            es_95 = np.mean(es_95_values) if len(es_95_values) > 0 else var_95
            es_99 = np.mean(es_99_values) if len(es_99_values) > 0 else var_99
            
            # Add lines
            axes[i].axvline(var_95, color='red', linestyle='--', linewidth=2, label='VaR 95%')
            axes[i].axvline(var_99, color='darkred', linestyle='--', linewidth=2, label='VaR 99%')
            axes[i].axvline(es_95, color='orange', linestyle='--', linewidth=2, label='ES 95%')
            axes[i].axvline(es_99, color='darkorange', linestyle='--', linewidth=2, label='ES 99%')
            
            # Add labels and title
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{model_name} Distribution (Time Step {time_step})')
            axes[i].grid(True)
            axes[i].legend()
            
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_stress_test_results(self, results: Optional[Dict[str, Dict[str, float]]] = None,
                               model_names: Optional[List[str]] = None,
                               scenario_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot stress test results.
        
        Args:
            results: Dictionary mapping scenario names to dictionaries mapping model names to results (optional)
            model_names: List of model names to plot (optional)
            scenario_names: List of scenario names to plot (optional)
            figsize: Figure size (default: (15, 10))
            
        Returns:
            Matplotlib figure object
        """
        if results is None:
            results = self.stress_test_results
            
        if model_names is None:
            # Get unique model names across all scenarios
            model_names = set()
            for scenario_results in results.values():
                model_names.update(scenario_results.keys())
            model_names = sorted(model_names)
            
        if scenario_names is None:
            scenario_names = sorted(results.keys())
            
        # Create figure
        fig, axes = plt.subplots(len(model_names), 1, figsize=figsize, sharex=True)
        
        # If only one model, wrap axes in a list
        if len(model_names) == 1:
            axes = [axes]
            
        # Plot results for each model
        for i, model_name in enumerate(model_names):
            # Extract results for this model
            model_results = []
            for scenario_name in scenario_names:
                if scenario_name in results and model_name in results[scenario_name]:
                    model_results.append(results[scenario_name][model_name])
                else:
                    model_results.append(0)  # Default value if missing
                    
            # Plot bar chart
            axes[i].bar(scenario_names, model_results)
            
            # Add labels and title
            axes[i].set_ylabel(f'{model_name}')
            axes[i].set_title(f'{model_name} Stress Test Results')
            axes[i].grid(True)
            
            # Rotate x-axis labels if many scenarios
            if len(scenario_names) > 5:
                plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
            
        # Add x-axis label to bottom subplot
        axes[-1].set_xlabel('Scenario')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_results(self, results: Dict[str, np.ndarray], file_path: str) -> None:
        """
        Save simulation results to a file.
        
        Args:
            results: Dictionary mapping model names to arrays of results
            file_path: Path to save the file
        """
        # Convert to dictionary of lists for easier serialization
        save_data = {model_name: paths.tolist() for model_name, paths in results.items()}
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(save_data, f)
        
    def load_results(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load simulation results from a file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            Dictionary mapping model names to arrays of results
        """
        # Load from file
        with open(file_path, 'r') as f:
            load_data = json.load(f)
        
        # Convert back to numpy arrays
        results = {model_name: np.array(paths) for model_name, paths in load_data.items()}
        
        # Store results
        self.results = results
        
        return results
    
    def save_stress_test_results(self, results: Dict[str, Dict[str, float]], file_path: str) -> None:
        """
        Save stress test results to a file.
        
        Args:
            results: Dictionary mapping scenario names to dictionaries mapping model names to results
            file_path: Path to save the file
        """
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f)
        
    def load_stress_test_results(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load stress test results from a file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            Dictionary mapping scenario names to dictionaries mapping model names to results
        """
        # Load from file
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Store results
        self.stress_test_results = results
        
        return results
    
    def convert_results_to_dataframe(self, results: Optional[Dict[str, np.ndarray]] = None,
                                   scenario_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Convert simulation results to a pandas DataFrame.
        
        Args:
            results: Dictionary mapping model names to arrays of results (optional)
            scenario_ids: List of scenario IDs (optional)
            
        Returns:
            DataFrame with columns for scenario_id, time_step, and each model
        """
        if results is None:
            results = self.results
            
        # Get dimensions
        model_name = next(iter(results.keys()))
        num_scenarios, num_steps = results[model_name].shape
        
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
                for model_name, model_results in results.items():
                    row[model_name] = model_results[s, t]
                data.append(row)
                
        return pd.DataFrame(data)
    
    def convert_stress_test_results_to_dataframe(self, results: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Convert stress test results to a pandas DataFrame.
        
        Args:
            results: Dictionary mapping scenario names to dictionaries mapping model names to results (optional)
            
        Returns:
            DataFrame with columns for scenario_name and each model
        """
        if results is None:
            results = self.stress_test_results
            
        # Create empty DataFrame
        data = []
        
        # Fill DataFrame
        for scenario_name, scenario_results in results.items():
            row = {'scenario_name': scenario_name}
            row.update(scenario_results)
            data.append(row)
                
        return pd.DataFrame(data)
