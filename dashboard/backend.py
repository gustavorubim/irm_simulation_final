"""
Backend API for the EVE and EVS simulation dashboard.
This module provides API endpoints for the React frontend to access simulation data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg') # Set non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from other modules
from models.model_factory import ModelFactory
from models.specific_models import *
from simulation.interest_rate_model import InterestRateSimulator, StressTestGenerator
from simulation.simulation_engine import SimulationEngine
from metrics.risk_metrics import RiskMetricsCalculator, PortfolioAnalyzer

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/build')
CORS(app)  # Enable CORS for all routes

# Initialize components
model_factory = ModelFactory()
rate_simulator = InterestRateSimulator()
simulation_engine = SimulationEngine(model_factory, rate_simulator)
risk_metrics_calculator = RiskMetricsCalculator()
portfolio_analyzer = PortfolioAnalyzer(risk_metrics_calculator)

# Initialize models
def initialize_models():
    """Initialize EVE and EVS models."""
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

# Initialize models
initialize_models()

# Global variables to store simulation results
simulation_results = {}
rate_paths = {}
stress_test_results = {}
metrics = {}

# API Routes
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    models = model_factory.list_models()
    return jsonify({
        'eve_models': [m for m in models if m.endswith('_eve')],
        'evs_models': [m for m in models if m.endswith('_evs')]
    })

@app.route('/api/model-details', methods=['GET'])
def get_model_details():
    """Get details of all models."""
    model_summaries = model_factory.get_model_summaries()
    return jsonify(model_summaries)

@app.route('/api/simulation/config', methods=['GET'])
def get_simulation_config():
    """Get current simulation configuration."""
    # Create a copy to avoid modifying the original config
    config_copy = rate_simulator.config.copy()
    
    # Convert numpy array to list for JSON serialization
    if 'correlation_matrix' in config_copy and isinstance(config_copy['correlation_matrix'], np.ndarray):
        config_copy['correlation_matrix'] = config_copy['correlation_matrix'].tolist()
        
    return jsonify(config_copy)

@app.route('/api/simulation/config', methods=['POST'])
def update_simulation_config():
    """Update simulation configuration."""
    config = request.json
    rate_simulator.update_config(config)
    return jsonify({"status": "success", "config": rate_simulator.config})

@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Run simulation with specified models."""
    data = request.json
    model_names = data.get('model_names', [])
    num_scenarios = data.get('num_scenarios', 1000)
    time_horizon = data.get('time_horizon', 3.0)
    
    # Update simulator config if provided
    if 'config' in data:
        rate_simulator.update_config(data['config'])
    
    # Set number of scenarios and time horizon
    rate_simulator.config['num_scenarios'] = num_scenarios
    rate_simulator.config['time_horizon'] = time_horizon
    
    # Run simulation
    global rate_paths, simulation_results
    rate_paths = rate_simulator.simulate()
    simulation_results = simulation_engine.run_simulation(model_names, rate_paths)
    
    # Calculate statistics
    statistics = simulation_engine.calculate_statistics(simulation_results)
    
    # Calculate comprehensive metrics
    global metrics
    metrics = risk_metrics_calculator.calculate_comprehensive_metrics(
        simulation_results, rate_paths)
    
    return jsonify({
        "status": "success",
        "statistics": {k: {sk: sv.tolist() for sk, sv in v.items()} 
                      for k, v in statistics.items()},
        "metrics": metrics
    })

@app.route('/api/simulation/results', methods=['GET'])
def get_simulation_results():
    """Get simulation results."""
    if not simulation_results:
        return jsonify({"status": "error", "message": "No simulation results available"})
    
    # Convert results to lists for JSON serialization
    results_json = {k: v[:, -1].tolist() for k, v in simulation_results.items()}
    
    return jsonify({
        "status": "success",
        "results": results_json
    })

@app.route('/api/simulation/timeseries', methods=['GET'])
def get_simulation_timeseries():
    """Get simulation time series data."""
    if not simulation_results:
        return jsonify({"status": "error", "message": "No simulation results available"})
    
    model_name = request.args.get('model', next(iter(simulation_results.keys())))
    num_scenarios = int(request.args.get('scenarios', 10))
    
    if model_name not in simulation_results:
        return jsonify({"status": "error", "message": f"Model {model_name} not found in results"})
    
    # Get model results
    model_results = simulation_results[model_name]
    num_total_scenarios, num_steps = model_results.shape
    
    # Select random scenarios
    if num_total_scenarios > num_scenarios:
        indices = np.random.choice(num_total_scenarios, num_scenarios, replace=False)
    else:
        indices = np.arange(num_total_scenarios)
    
    # Extract selected scenarios
    selected_scenarios = model_results[indices, :]
    
    # Calculate statistics
    mean = np.mean(model_results, axis=0).tolist()
    median = np.median(model_results, axis=0).tolist()
    percentile_5 = np.percentile(model_results, 5, axis=0).tolist()
    percentile_95 = np.percentile(model_results, 95, axis=0).tolist()
    
    return jsonify({
        "status": "success",
        "model": model_name,
        "scenarios": selected_scenarios.tolist(),
        "statistics": {
            "mean": mean,
            "median": median,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95
        },
        "steps": list(range(num_steps))
    })

@app.route('/api/rates/timeseries', methods=['GET'])
def get_rates_timeseries():
    """Get interest rate time series data."""
    if not rate_paths:
        return jsonify({"status": "error", "message": "No rate paths available"})
    
    rate_tenor = request.args.get('tenor', next(iter(rate_paths.keys())))
    num_scenarios = int(request.args.get('scenarios', 10))
    
    if rate_tenor not in rate_paths:
        return jsonify({"status": "error", "message": f"Rate tenor {rate_tenor} not found in paths"})
    
    # Get rate paths
    tenor_paths = rate_paths[rate_tenor]
    num_total_scenarios, num_steps = tenor_paths.shape
    
    # Select random scenarios
    if num_total_scenarios > num_scenarios:
        indices = np.random.choice(num_total_scenarios, num_scenarios, replace=False)
    else:
        indices = np.arange(num_total_scenarios)
    
    # Extract selected scenarios
    selected_scenarios = tenor_paths[indices, :]
    
    # Calculate statistics
    mean = np.mean(tenor_paths, axis=0).tolist()
    median = np.median(tenor_paths, axis=0).tolist()
    percentile_5 = np.percentile(tenor_paths, 5, axis=0).tolist()
    percentile_95 = np.percentile(tenor_paths, 95, axis=0).tolist()
    
    return jsonify({
        "status": "success",
        "tenor": rate_tenor,
        "scenarios": selected_scenarios.tolist(),
        "statistics": {
            "mean": mean,
            "median": median,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95
        },
        "steps": list(range(num_steps))
    })

@app.route('/api/stress-test/run', methods=['POST'])
def run_stress_test():
    """Run stress tests with specified models."""
    data = request.json
    model_names = data.get('model_names', [])
    
    # Get stress test generator
    stress_generator = StressTestGenerator(
        rate_simulator.config['rate_tenors'],
        rate_simulator.config['initial_rates']
    )
    
    # Get stress scenarios
    if 'scenarios' in data:
        # Use provided scenario names to fetch full scenario details
        scenario_names = data['scenarios']
        if not isinstance(scenario_names, list):
            # Ensure it's a list
            abort(400, description="Invalid format for 'scenarios'. Expected a list of scenario names.")
        
        stress_scenarios_dict = {}
        try:
            for name in scenario_names:
                if not isinstance(name, str):
                     # Ensure names are strings
                    abort(400, description=f"Invalid scenario name type: {type(name)}. Expected string.")
                # Fetch details for each scenario name using the generator
                # The stress_generator was instantiated earlier in the function
                scenario_details = stress_generator.get_predefined_scenario(name)
                stress_scenarios_dict[name] = scenario_details
        except ValueError as e:
            # Handle cases where get_predefined_scenario raises ValueError (e.g., name not found)
            abort(400, description=str(e))
        except Exception as e:
            # Catch unexpected errors during scenario fetching
            # Consider adding logging here if available: logging.error(f"Error fetching scenario: {e}")
            abort(500, description="Internal server error processing scenarios.")
            
        stress_scenarios = stress_scenarios_dict # Assign the built dictionary
    else:
        # Use all predefined scenarios
        stress_scenarios = stress_generator.generate_all_scenarios()
    
    # Run stress tests
    global stress_test_results
    stress_test_results = simulation_engine.run_stress_tests(model_names, stress_scenarios)
    
    return jsonify({
        "status": "success",
        "results": stress_test_results
    })

@app.route('/api/stress-test/scenarios', methods=['GET'])
def get_stress_scenarios():
    """Get available stress test scenarios."""
    # Get stress test generator
    stress_generator = StressTestGenerator(
        rate_simulator.config['rate_tenors'],
        rate_simulator.config['initial_rates']
    )
    
    # Get scenario names
    scenario_names = stress_generator.list_predefined_scenarios()
    
    return jsonify({
        "status": "success",
        "scenarios": scenario_names
    })

@app.route('/api/stress-test/results', methods=['GET'])
def get_stress_test_results():
    """Get stress test results."""
    if not stress_test_results:
        return jsonify({"status": "error", "message": "No stress test results available"})
    
    return jsonify({
        "status": "success",
        "results": stress_test_results
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get calculated metrics."""
    if not metrics:
        return jsonify({"status": "error", "message": "No metrics available"})
    
    return jsonify({
        "status": "success",
        "metrics": metrics
    })

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio weights."""
    data = request.json
    component_names = data.get('components', [])
    objective = data.get('objective', 'min_variance')
    constraints = data.get('constraints', None)
    
    # Check if components exist in simulation results
    if not all(c in simulation_results for c in component_names):
        return jsonify({
            "status": "error", 
            "message": "Not all components found in simulation results"
        })
    
    # Extract component results
    component_results = {c: simulation_results[c] for c in component_names}
    
    # Optimize weights
    optimal_weights = portfolio_analyzer.calculate_optimal_weights(
        component_results, objective, constraints)
    
    return jsonify({
        "status": "success",
        "weights": optimal_weights
    })

@app.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    """Analyze portfolio with given weights."""
    data = request.json
    weights = data.get('weights', {})
    
    # Check if components exist in simulation results
    if not all(c in simulation_results for c in weights):
        return jsonify({
            "status": "error", 
            "message": "Not all components found in simulation results"
        })
    
    # Extract component results
    component_results = {c: simulation_results[c] for c in weights}
    
    # Calculate portfolio value
    portfolio_values = portfolio_analyzer.calculate_portfolio_value(
        component_results, weights)
    
    # Calculate risk contributions
    risk_contributions = portfolio_analyzer.calculate_risk_decomposition(
        component_results, weights)
    
    # Calculate diversification benefit
    diversification_benefit = portfolio_analyzer.calculate_diversification_benefit(
        component_results, weights)
    
    # Calculate portfolio metrics
    portfolio_metrics = risk_metrics_calculator.calculate_comprehensive_metrics(
        {"portfolio": portfolio_values}, rate_paths)
    
    return jsonify({
        "status": "success",
        "portfolio_values": portfolio_values[:, -1].tolist(),
        "risk_contributions": risk_contributions,
        "diversification_benefit": diversification_benefit,
        "metrics": portfolio_metrics
    })

@app.route('/api/charts/histogram', methods=['GET'])
def get_histogram_chart():
    """Generate histogram chart for model results."""
    if not simulation_results:
        return jsonify({"status": "error", "message": "No simulation results available"})
    
    model_name = request.args.get('model', next(iter(simulation_results.keys())))
    bins = int(request.args.get('bins', 50))
    
    if model_name not in simulation_results:
        return jsonify({"status": "error", "message": f"Model {model_name} not found in results"})
    
    # Get model results (last time step)
    values = simulation_results[model_name][:, -1]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(values, bins=bins, alpha=0.7)
    
    # Add VaR and ES lines
    var_95 = np.percentile(values, 5)
    var_99 = np.percentile(values, 1)
    
    # Calculate ES
    es_95_values = values[values <= var_95]
    es_99_values = values[values <= var_99]
    es_95 = np.mean(es_95_values) if len(es_95_values) > 0 else var_95
    es_99 = np.mean(es_99_values) if len(es_99_values) > 0 else var_99
    
    # Add lines
    plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}')
    plt.axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.2f}')
    plt.axvline(es_95, color='orange', linestyle='--', linewidth=2, label=f'ES 95%: {es_95:.2f}')
    plt.axvline(es_99, color='darkorange', linestyle='--', linewidth=2, label=f'ES 99%: {es_99:.2f}')
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} Distribution')
    plt.grid(True)
    plt.legend()
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "chart": f"data:image/png;base64,{img_str}",
        "statistics": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "var_95": float(var_95),
            "var_99": float(var_99),
            "es_95": float(es_95),
            "es_99": float(es_99)
        }
    })

@app.route('/api/charts/correlation', methods=['GET'])
def get_correlation_chart():
    """Generate correlation matrix chart for model results."""
    if not simulation_results:
        return jsonify({"status": "error", "message": "No simulation results available"})
    
    # Calculate correlation matrix
    correlation_matrix = risk_metrics_calculator.calculate_correlation_matrix(simulation_results)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(label='Correlation')
    
    # Add labels and title
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title('Correlation Matrix')
    
    # Add correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "chart": f"data:image/png;base64,{img_str}",
        "correlation_matrix": correlation_matrix.to_dict()
    })

@app.route('/api/charts/sensitivity', methods=['GET'])
def get_sensitivity_chart():
    """Generate rate sensitivity chart for model results."""
    if not simulation_results or not rate_paths:
        return jsonify({"status": "error", "message": "No simulation results available"})
    
    # Calculate rate sensitivity
    sensitivities = risk_metrics_calculator.calculate_rate_sensitivity(
        simulation_results, rate_paths)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    models = list(sensitivities.keys())
    tenors = list(rate_paths.keys())
    
    # Create positions for bars
    x = np.arange(len(tenors))
    width = 0.8 / len(models)
    
    # Plot bars for each model
    for i, model in enumerate(models):
        values = [sensitivities[model][tenor] for tenor in tenors]
        plt.bar(x + i*width - 0.4 + width/2, values, width, label=model)
    
    # Add labels and title
    plt.xlabel('Rate Tenor')
    plt.ylabel('Sensitivity')
    plt.title('Rate Sensitivity Analysis')
    plt.xticks(x, tenors)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "chart": f"data:image/png;base64,{img_str}",
        "sensitivities": sensitivities
    })

@app.route('/api/charts/stress-test', methods=['GET'])
def get_stress_test_chart():
    """Generate stress test results chart."""
    if not stress_test_results:
        return jsonify({"status": "error", "message": "No stress test results available"})
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Get model names and scenario names
    model_names = set()
    for scenario_results in stress_test_results.values():
        model_names.update(scenario_results.keys())
    model_names = sorted(model_names)
    
    scenario_names = sorted(stress_test_results.keys())
    
    # Create subplots for each model
    fig, axes = plt.subplots(len(model_names), 1, figsize=(15, 3*len(model_names)), sharex=True)
    
    # If only one model, wrap axes in a list
    if len(model_names) == 1:
        axes = [axes]
    
    # Plot results for each model
    for i, model_name in enumerate(model_names):
        # Extract results for this model
        model_results = []
        for scenario_name in scenario_names:
            if scenario_name in stress_test_results and model_name in stress_test_results[scenario_name]:
                model_results.append(stress_test_results[scenario_name][model_name])
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
    
    # Save figure to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "chart": f"data:image/png;base64,{img_str}",
        "results": stress_test_results
    })

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React app."""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
