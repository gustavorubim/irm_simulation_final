#!/usr/bin/env python3
"""
Main entry point for the EVE and EVS Stochastic Simulation System.
This script integrates all modules and starts the backend server.
"""

import os
import sys
import argparse
import logging
from dashboard.backend import app
from models.model_factory import ModelFactory
from models.specific_models import *
from simulation.interest_rate_model import InterestRateSimulator
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

def main():
    """Main function to start the application."""
    parser = argparse.ArgumentParser(description='EVE and EVS Stochastic Simulation System')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info("Starting EVE and EVS Stochastic Simulation System")
    
    # Initialize components
    model_factory = ModelFactory()
    initialize_models(model_factory)
    
    # Start the Flask server
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
