"""
Model factory for creating and managing EVE and EVS regression models.
This module provides functionality to create and manage multiple regression models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from .base_model import BaseRegressionModel, LinearRegressionModel, EVEModel, EVSModel


class ModelFactory:
    """
    Factory class for creating and managing regression models.
    
    This class provides functionality to create, store, and retrieve
    different types of regression models for EVE and EVS analysis.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.models = {}
        
    def create_model(self, model_type: str, name: str, description: str = "", **kwargs) -> BaseRegressionModel:
        """
        Create a new regression model.
        
        Args:
            model_type: Type of model to create ('linear', 'eve', 'evs')
            name: Unique identifier for the model
            description: Human-readable description of the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            Created regression model
        """
        if name in self.models:
            raise ValueError(f"Model with name '{name}' already exists")
            
        if model_type.lower() == 'linear':
            model = LinearRegressionModel(name, description)
        elif model_type.lower() == 'eve':
            model = EVEModel(name, description, **kwargs)
        elif model_type.lower() == 'evs':
            model = EVSModel(name, description, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.models[name] = model
        return model
    
    def get_model(self, name: str) -> BaseRegressionModel:
        """
        Retrieve a model by name.
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            Retrieved regression model
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
            
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """
        Get a list of all model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_model_summaries(self) -> List[Dict]:
        """
        Get summaries of all models.
        
        Returns:
            List of model summary dictionaries
        """
        return [model.summary() for model in self.models.values()]
    
    def delete_model(self, name: str) -> None:
        """
        Delete a model by name.
        
        Args:
            name: Name of the model to delete
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
            
        del self.models[name]
