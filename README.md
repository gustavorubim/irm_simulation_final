#!/usr/bin/env python3
"""
README for the EVE and EVS Stochastic Simulation System.
This file provides documentation on how to use the system.
"""

# EVE and EVS Stochastic Simulation System
# ========================================

## Overview

This system provides a comprehensive stochastic simulation framework for Economic Value of Equity (EVE) and Economic Value Sensitivity (EVS) in the context of interest rate risk management. It includes:

1. A collection of ~20 different regression models with factor sensitivity to interest rate movements
2. Monte Carlo simulation using the Heath-Jarrow-Morton (HJM) model for interest rates
3. Comprehensive metrics calculation for risk and performance analysis
4. Modern, visually impressive dashboard for data visualization and analysis

## System Requirements

- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher

## Python Dependencies

- Flask
- Flask-CORS
- NumPy
- pandas
- matplotlib
- SciPy
- scikit-learn

## Installation

1. Clone the repository or extract the provided files to a directory of your choice.

2. Install Python dependencies:
   ```
   pip3 install flask flask-cors numpy pandas matplotlib scipy scikit-learn
   ```

3. Run the integration script to build the frontend and connect it with the backend:
   ```
   python3 integrate.py
   ```

## Running the System

1. Start the application:
   ```
   python3 main.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

## System Architecture

The system consists of the following main components:

### 1. Models Module

Located in the `models` directory, this module contains:
- `base_model.py`: Base class for all models
- `model_factory.py`: Factory class for creating and managing models
- `specific_models.py`: Implementation of specific EVE and EVS models

### 2. Simulation Module

Located in the `simulation` directory, this module contains:
- `interest_rate_model.py`: Implementation of the HJM model for interest rate simulation
- `simulation_engine.py`: Engine for running simulations with models and interest rates

### 3. Metrics Module

Located in the `metrics` directory, this module contains:
- `risk_metrics.py`: Implementation of risk metrics calculation and portfolio analysis

### 4. Dashboard Module

Located in the `dashboard` directory, this module contains:
- `backend.py`: Flask backend API for the dashboard
- `frontend`: React frontend for the dashboard

## Using the Dashboard

The dashboard consists of six main pages:

### 1. Dashboard

The main dashboard provides an overview of key metrics and visualizations, including:
- Key risk metrics (VaR, ES, Sharpe Ratio)
- Time series charts of model values
- Distribution histograms
- Correlation matrix
- Rate sensitivity analysis

### 2. Simulation

The simulation page allows you to:
- Select models for simulation
- Configure simulation parameters (number of scenarios, time horizon, etc.)
- Run Monte Carlo simulations
- View simulation results and metrics

### 3. Models

The models page provides detailed information about all EVE and EVS models, including:
- Model descriptions and types
- Rate sensitivities
- Model coefficients
- Technical documentation

### 4. Stress Test

The stress test page allows you to:
- Select models and stress scenarios
- Run stress tests to evaluate model performance under adverse conditions
- View stress test results and sensitivity analysis

### 5. Portfolio

The portfolio page provides portfolio optimization and analysis capabilities:
- Select portfolio components
- Configure component weights
- Analyze portfolio metrics (VaR, ES, Sharpe Ratio, etc.)
- Optimize portfolio weights based on different objectives
- View risk contributions and diversification benefits

### 6. Settings

The settings page allows you to:
- Configure simulation parameters
- Customize UI preferences
- Export and import configurations

## Testing

To run tests on all components of the system:
```
python3 test.py
```

This will generate a test report in the `test_report` directory with visualizations and detailed results.

## Customization

### Adding New Models

To add a new model:

1. Create a new model class in `models/specific_models.py` that inherits from `BaseModel`
2. Implement the `calculate` method to define how the model responds to interest rate changes
3. Add the model to the initialization in `main.py`

### Modifying Interest Rate Simulation

To modify the interest rate simulation:

1. Edit the configuration in `simulation/interest_rate_model.py`
2. Adjust parameters such as volatility, mean reversion, and correlation

### Extending the Dashboard

To add new visualizations or features to the dashboard:

1. Add new API endpoints in `dashboard/backend.py`
2. Create new React components in the frontend
3. Update the relevant pages to include the new components

## Troubleshooting

### Common Issues

1. **Backend server fails to start**
   - Check that all Python dependencies are installed
   - Verify that port 5000 is not in use by another application

2. **Frontend fails to build**
   - Check that Node.js and npm are installed
   - Verify that all frontend dependencies are installed

3. **Dashboard shows no data**
   - Ensure that a simulation has been run
   - Check browser console for any error messages
   - Verify that the backend API is accessible

### Getting Help

For additional assistance, please refer to the documentation in the code or contact the system administrator.
