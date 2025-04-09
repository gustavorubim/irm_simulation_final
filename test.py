#!/usr/bin/env python3
"""
Test script for the EVE and EVS Stochastic Simulation System.
This script runs tests on all components to ensure they work correctly.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.model_factory import ModelFactory
from models.specific_models import *
from simulation.interest_rate_model import InterestRateSimulator, StressTestGenerator
from simulation.simulation_engine import SimulationEngine
from metrics.risk_metrics import RiskMetricsCalculator, PortfolioAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_models(model_factory):
    """Initialize EVE and EVS models."""
    logger.info("Initializing models...")
    
    # EVE Models
    model_factory.models["retail_mortgage_eve"] = RetailMortgageEVEModel()
    model_factory.models["commercial_loan_eve"] = CommercialLoanEVEModel()
    model_factory.models["fixed_income_eve"] = FixedIncomeEVEModel()
    model_factory.models["deposit_eve"] = DepositEVEModel()
    model_factory.models["credit_card_eve"] = CreditCardEVEModel()
    model_factory.models["auto_loan_eve"] = AutoLoanEVEModel()
    model_factory.models["student_loan_eve"] = StudentLoanEVEModel()
    model_factory.models["commercial_real_estate_eve"] = CommercialRealEstateEVEModel()
    model_factory.models["treasury_portfolio_eve"] = TreasuryPortfolioEVEModel()
    
    # EVS Models
    model_factory.models["retail_mortgage_evs"] = RetailMortgageEVSModel()
    model_factory.models["commercial_loan_evs"] = CommercialLoanEVSModel()
    model_factory.models["fixed_income_evs"] = FixedIncomeEVSModel()
    model_factory.models["deposit_evs"] = DepositEVSModel()
    model_factory.models["credit_card_evs"] = CreditCardEVSModel()
    model_factory.models["auto_loan_evs"] = AutoLoanEVSModel()
    model_factory.models["student_loan_evs"] = StudentLoanEVSModel()
    model_factory.models["commercial_real_estate_evs"] = CommercialRealEstateEVSModel()
    model_factory.models["treasury_portfolio_evs"] = TreasuryPortfolioEVSModel()
    
    logger.info(f"Initialized {len(model_factory.models)} models")
    return model_factory.list_models()

def test_models(model_factory):
    """Test all models."""
    logger.info("Testing models...")
    
    # Create test rate data
    test_rates = {
        'sofr_rate': 0.03,
        'treasury_1y': 0.035,
        'treasury_2y': 0.04,
        'treasury_3y': 0.042,
        'treasury_5y': 0.045,
        'treasury_10y': 0.05
    }
    
    # Test each model
    results = {}
    for model_name, model in model_factory.models.items():
        try:
            value = model.calculate(test_rates)
            results[model_name] = value
            logger.info(f"Model {model_name}: {value}")
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            results[model_name] = None
    
    return results

def test_interest_rate_simulation(num_scenarios=100, time_horizon=3.0):
    """Test interest rate simulation."""
    logger.info("Testing interest rate simulation...")
    
    # Initialize simulator
    simulator = InterestRateSimulator()
    
    # Run simulation
    try:
        rate_paths = simulator.simulate(num_scenarios=num_scenarios, time_horizon=time_horizon)
        
        # Check results
        for tenor, paths in rate_paths.items():
            logger.info(f"Tenor {tenor}: Shape {paths.shape}")
            logger.info(f"  Mean: {np.mean(paths[:, -1]):.4f}")
            logger.info(f"  Std: {np.std(paths[:, -1]):.4f}")
            logger.info(f"  Min: {np.min(paths[:, -1]):.4f}")
            logger.info(f"  Max: {np.max(paths[:, -1]):.4f}")
        
        # Test stress scenarios
        stress_generator = StressTestGenerator(
            simulator.config['rate_tenors'],
            simulator.config['initial_rates']
        )
        scenarios = stress_generator.generate_all_scenarios()
        logger.info(f"Generated {len(scenarios)} stress scenarios")
        
        return rate_paths, scenarios
    
    except Exception as e:
        logger.error(f"Error testing interest rate simulation: {e}")
        return None, None

def test_simulation_engine(model_factory, rate_paths, stress_scenarios):
    """Test simulation engine."""
    logger.info("Testing simulation engine...")
    
    # Initialize simulator and engine
    simulator = InterestRateSimulator()
    engine = SimulationEngine(model_factory, simulator)
    
    # Run simulation
    try:
        # Select a subset of models for testing
        test_models = model_factory.list_models()[:5]
        logger.info(f"Running simulation with models: {test_models}")
        
        # Run Monte Carlo simulation
        results = engine.run_simulation(test_models, rate_paths)
        
        # Check results
        for model_name, values in results.items():
            logger.info(f"Model {model_name}: Shape {values.shape}")
            logger.info(f"  Mean: {np.mean(values[:, -1]):.4f}")
            logger.info(f"  Std: {np.std(values[:, -1]):.4f}")
            logger.info(f"  Min: {np.min(values[:, -1]):.4f}")
            logger.info(f"  Max: {np.max(values[:, -1]):.4f}")
        
        # Calculate statistics
        statistics = engine.calculate_statistics(results)
        logger.info(f"Calculated statistics for {len(statistics)} models")
        
        # Run stress tests
        stress_results = engine.run_stress_tests(test_models, stress_scenarios)
        logger.info(f"Ran {len(stress_scenarios)} stress tests for {len(test_models)} models")
        
        return results, statistics, stress_results
    
    except Exception as e:
        logger.error(f"Error testing simulation engine: {e}")
        return None, None, None

def test_metrics_calculation(simulation_results, rate_paths):
    """Test metrics calculation."""
    logger.info("Testing metrics calculation...")
    
    # Initialize metrics calculator
    calculator = RiskMetricsCalculator()
    
    try:
        # Calculate comprehensive metrics
        metrics = calculator.calculate_comprehensive_metrics(simulation_results, rate_paths)
        logger.info(f"Calculated metrics for {len(metrics)} models")
        
        # Calculate correlation matrix
        correlation_matrix = calculator.calculate_correlation_matrix(simulation_results)
        logger.info(f"Calculated correlation matrix of shape {correlation_matrix.shape}")
        
        # Calculate rate sensitivity
        sensitivities = calculator.calculate_rate_sensitivity(simulation_results, rate_paths)
        logger.info(f"Calculated rate sensitivities for {len(sensitivities)} models")
        
        # Initialize portfolio analyzer
        analyzer = PortfolioAnalyzer(calculator)
        
        # Select components for portfolio
        components = list(simulation_results.keys())[:3]
        weights = {component: 1.0/len(components) for component in components}
        
        # Calculate portfolio value
        portfolio_values = analyzer.calculate_portfolio_value(
            {c: simulation_results[c] for c in components}, 
            weights
        )
        logger.info(f"Calculated portfolio values of shape {portfolio_values.shape}")
        
        # Calculate risk decomposition
        risk_contributions = analyzer.calculate_risk_decomposition(
            {c: simulation_results[c] for c in components}, 
            weights
        )
        logger.info(f"Calculated risk contributions for {len(risk_contributions)} components")
        
        # Calculate diversification benefit
        diversification_benefit = analyzer.calculate_diversification_benefit(
            {c: simulation_results[c] for c in components}, 
            weights
        )
        logger.info(f"Calculated diversification benefit: {diversification_benefit:.4f}")
        
        # Calculate optimal weights
        optimal_weights = analyzer.calculate_optimal_weights(
            {c: simulation_results[c] for c in components}, 
            'min_variance'
        )
        logger.info(f"Calculated optimal weights for {len(optimal_weights)} components")
        
        return metrics, correlation_matrix, sensitivities, portfolio_values, risk_contributions, diversification_benefit, optimal_weights
    
    except Exception as e:
        logger.error(f"Error testing metrics calculation: {e}")
        return None, None, None, None, None, None, None

def generate_test_report(
    model_results, 
    rate_paths, 
    simulation_results, 
    statistics, 
    stress_results, 
    metrics, 
    correlation_matrix, 
    sensitivities, 
    portfolio_values, 
    risk_contributions, 
    diversification_benefit, 
    optimal_weights
):
    """Generate test report."""
    logger.info("Generating test report...")
    
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_report')
    os.makedirs(report_dir, exist_ok=True)
    
    try:
        # Write model results
        with open(os.path.join(report_dir, 'model_results.txt'), 'w') as f:
            f.write("Model Test Results\n")
            f.write("=================\n\n")
            for model_name, value in model_results.items():
                f.write(f"{model_name}: {value}\n")
        
        # Plot rate paths
        plt.figure(figsize=(10, 6))
        for tenor, paths in rate_paths.items():
            plt.plot(np.mean(paths, axis=0), label=tenor)
        plt.title('Average Interest Rate Paths')
        plt.xlabel('Time Step')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, 'rate_paths.png'))
        plt.close()
        
        # Plot simulation results
        plt.figure(figsize=(10, 6))
        for model_name, values in simulation_results.items():
            plt.plot(np.mean(values, axis=0), label=model_name)
        plt.title('Average Simulation Results')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, 'simulation_results.png'))
        plt.close()
        
        # Write statistics
        with open(os.path.join(report_dir, 'statistics.txt'), 'w') as f:
            f.write("Simulation Statistics\n")
            f.write("====================\n\n")
            for model_name, stats in statistics.items():
                f.write(f"{model_name}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value[-1]}\n")
                f.write("\n")
        
        # Write stress test results
        with open(os.path.join(report_dir, 'stress_test_results.txt'), 'w') as f:
            f.write("Stress Test Results\n")
            f.write("==================\n\n")
            for scenario_name, scenario_results in stress_results.items():
                f.write(f"{scenario_name}:\n")
                for model_name, value in scenario_results.items():
                    f.write(f"  {model_name}: {value}\n")
                f.write("\n")
        
        # Write metrics
        with open(os.path.join(report_dir, 'metrics.txt'), 'w') as f:
            f.write("Risk Metrics\n")
            f.write("===========\n\n")
            for model_name, model_metrics in metrics.items():
                f.write(f"{model_name}:\n")
                for metric_name, metric_value in model_metrics.items():
                    f.write(f"  {metric_name}: {metric_value}\n")
                f.write("\n")
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
        plt.title('Correlation Matrix')
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Write sensitivities
        with open(os.path.join(report_dir, 'sensitivities.txt'), 'w') as f:
            f.write("Rate Sensitivities\n")
            f.write("================\n\n")
            for model_name, model_sensitivities in sensitivities.items():
                f.write(f"{model_name}:\n")
                for tenor, sensitivity in model_sensitivities.items():
                    f.write(f"  {tenor}: {sensitivity}\n")
                f.write("\n")
        
        # Plot portfolio values
        plt.figure(figsize=(10, 6))
        plt.plot(np.mean(portfolio_values, axis=0))
        plt.title('Average Portfolio Value')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, 'portfolio_values.png'))
        plt.close()
        
        # Write risk contributions
        with open(os.path.join(report_dir, 'risk_contributions.txt'), 'w') as f:
            f.write("Risk Contributions\n")
            f.write("================\n\n")
            for component, contribution in risk_contributions.items():
                f.write(f"{component}: {contribution}\n")
            f.write(f"\nDiversification Benefit: {diversification_benefit}\n")
        
        # Write optimal weights
        with open(os.path.join(report_dir, 'optimal_weights.txt'), 'w') as f:
            f.write("Optimal Weights (Min Variance)\n")
            f.write("============================\n\n")
            for component, weight in optimal_weights.items():
                f.write(f"{component}: {weight}\n")
        
        logger.info(f"Test report generated in {report_dir}")
        return report_dir
    
    except Exception as e:
        logger.error(f"Error generating test report: {e}")
        return None

def main():
    """Main function to run tests."""
    logger.info("Starting tests...")
    
    # Initialize model factory
    model_factory = ModelFactory()
    model_names = initialize_models(model_factory)
    
    # Test models
    model_results = test_models(model_factory)
    
    # Test interest rate simulation
    rate_paths, stress_scenarios = test_interest_rate_simulation()
    
    # Test simulation engine
    simulation_results, statistics, stress_results = test_simulation_engine(
        model_factory, rate_paths, stress_scenarios
    )
    
    # Test metrics calculation
    metrics, correlation_matrix, sensitivities, portfolio_values, risk_contributions, diversification_benefit, optimal_weights = test_metrics_calculation(
        simulation_results, rate_paths
    )
    
    # Generate test report
    report_dir = generate_test_report(
        model_results, 
        rate_paths, 
        simulation_results, 
        statistics, 
        stress_results, 
        metrics, 
        correlation_matrix, 
        sensitivities, 
        portfolio_values, 
        risk_contributions, 
        diversification_benefit, 
        optimal_weights
    )
    
    logger.info("Tests completed successfully")
    logger.info(f"Test report available in {report_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
