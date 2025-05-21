#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid model combining CNN and LGBM for cryptocurrency price prediction.
This model uses a combination of convolutional layers for short-term patterns
and LGBM for effective long-term relationship modeling.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from datetime import datetime
import joblib

# Import other model components
from cnn_model import CNNModel
from lgbm_model import LGBMModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridModel:
    """
    Hybrid model combining CNN and LGBM for cryptocurrency price prediction.
    """
    
    def __init__(self, input_shape, lgbm_params=None):
        """
        Initialize the hybrid model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            lgbm_params: Parameters for the LGBM model
        """
        self.input_shape = input_shape
        self.lgbm_params = lgbm_params
        
        # Initialize the CNN component
        self.cnn_model = CNNModel(input_shape)
        
        # Initialize the LGBM component
        self.lgbm_model = LGBMModel(lgbm_params)
        
        logger.info(f"Hybrid model initialized with input shape {input_shape}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              cnn_epochs=100, cnn_batch_size=32, cnn_patience=20,
              lgbm_num_boost_round=1000, lgbm_early_stopping_rounds=50,
              model_dir=None):
        """
        Train the hybrid model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cnn_epochs: Number of epochs for CNN training
            cnn_batch_size: Batch size for CNN training
            cnn_patience: Early stopping patience for CNN
            lgbm_num_boost_round: Number of boosting rounds for LGBM
            lgbm_early_stopping_rounds: Early stopping rounds for LGBM
            model_dir: Directory to save model components
            
        Returns:
            Dictionary with training history
        """
        try:
            # Create model directory if provided
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
                
                # Define component model paths
                cnn_model_path = os.path.join(model_dir, "cnn_component.h5")
                lgbm_model_path = os.path.join(model_dir, "lgbm_component.txt")
            else:
                cnn_model_path = None
                lgbm_model_path = None
            
            # Train CNN component
            logger.info("Training CNN component...")
            cnn_history = self.cnn_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=cnn_epochs,
                batch_size=cnn_batch_size,
                patience=cnn_patience,
                model_path=cnn_model_path
            )
            
            # Train LGBM component
            logger.info("Training LGBM component...")
            lgbm_model = self.lgbm_model.train(
                X_train, y_train,
                X_val, y_val,
                num_boost_round=lgbm_num_boost_round,
                early_stopping_rounds=lgbm_early_stopping_rounds
            )
            
            # Save LGBM model if path is provided
            if lgbm_model_path:
                self.lgbm_model.save(lgbm_model_path)
            
            # Return training history
            history = {
                'cnn': cnn_history.history if cnn_history else None,
                'lgbm': {
                    'best_iteration': lgbm_model.best_iteration if lgbm_model else None
                }
            }
            
            return history
        
        except Exception as e:
            logger.error(f"Error training hybrid model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions using both model components and combine them.
        
        Args:
            X: Input features
            
        Returns:
            Combined predictions
        """
        try:
            # Get CNN predictions
            cnn_pred = self.cnn_model.predict(X)
            
            # Get LGBM predictions
            lgbm_pred = self.lgbm_model.predict(X)
            
            # Combine predictions (simple average)
            # This could be replaced with a more sophisticated ensemble method
            combined_pred = 0.4 * cnn_pred + 0.6 * lgbm_pred
            
            return combined_pred
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the hybrid model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get individual component predictions
            cnn_pred = self.cnn_model.predict(X_test)
            lgbm_pred = self.lgbm_model.predict(X_test)
            
            # Get combined predictions
            combined_pred = self.predict(X_test)
            
            # Calculate metrics for CNN
            cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_pred))
            cnn_mae = mean_absolute_error(y_test, cnn_pred)
            cnn_r2 = r2_score(y_test, cnn_pred)
            cnn_within_5_percent = np.mean(np.abs((y_test - cnn_pred) / y_test) < 0.05) * 100
            
            # Calculate metrics for LGBM
            lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
            lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
            lgbm_r2 = r2_score(y_test, lgbm_pred)
            lgbm_within_5_percent = np.mean(np.abs((y_test - lgbm_pred) / y_test) < 0.05) * 100
            
            # Calculate metrics for combined model
            combined_rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
            combined_mae = mean_absolute_error(y_test, combined_pred)
            combined_r2 = r2_score(y_test, combined_pred)
            combined_within_5_percent = np.mean(np.abs((y_test - combined_pred) / y_test) < 0.05) * 100
            
            # Log evaluation results
            logger.info("Evaluation Results:")
            logger.info(f"CNN - RMSE: {cnn_rmse:.4f}, MAE: {cnn_mae:.4f}, R²: {cnn_r2:.4f}, Within 5%: {cnn_within_5_percent:.2f}%")
            logger.info(f"LGBM - RMSE: {lgbm_rmse:.4f}, MAE: {lgbm_mae:.4f}, R²: {lgbm_r2:.4f}, Within 5%: {lgbm_within_5_percent:.2f}%")
            logger.info(f"Combined - RMSE: {combined_rmse:.4f}, MAE: {combined_mae:.4f}, R²: {combined_r2:.4f}, Within 5%: {combined_within_5_percent:.2f}%")
            
            # Return metrics
            return {
                'cnn': {
                    'rmse': cnn_rmse,
                    'mae': cnn_mae,
                    'r2': cnn_r2,
                    'within_5_percent': cnn_within_5_percent
                },
                'lgbm': {
                    'rmse': lgbm_rmse,
                    'mae': lgbm_mae,
                    'r2': lgbm_r2,
                    'within_5_percent': lgbm_within_5_percent
                },
                'combined': {
                    'rmse': combined_rmse,
                    'mae': combined_mae,
                    'r2': combined_r2,
                    'within_5_percent': combined_within_5_percent
                }
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def save(self, model_dir):
        """
        Save the hybrid model components.
        
        Args:
            model_dir: Directory to save model components
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save CNN component
            cnn_path = os.path.join(model_dir, "cnn_component.h5")
            self.cnn_model.save(cnn_path)
            
            # Save LGBM component
            lgbm_path = os.path.join(model_dir, "lgbm_component.txt")
            self.lgbm_model.save(lgbm_path)
            
            # Save model metadata
            metadata = {
                'input_shape': self.input_shape,
                'lgbm_params': self.lgbm_params,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = os.path.join(model_dir, "metadata.json")
            pd.DataFrame([metadata]).to_json(metadata_path, orient='records')
            
            logger.info(f"Hybrid model saved to {model_dir}")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, model_dir):
        """
        Load the hybrid model components.
        
        Args:
            model_dir: Directory with saved model components
            
        Returns:
            Loaded model
        """
        try:
            # Check if model directory exists
            if not os.path.exists(model_dir):
                logger.error(f"Model directory {model_dir} does not exist")
                return False
            
            # Load CNN component
            cnn_path = os.path.join(model_dir, "cnn_component.h5")
            if os.path.exists(cnn_path):
                self.cnn_model.load(cnn_path)
            else:
                logger.error(f"CNN component not found at {cnn_path}")
                return False
            
            # Load LGBM component
            lgbm_path = os.path.join(model_dir, "lgbm_component.txt")
            if os.path.exists(lgbm_path):
                self.lgbm_model.load(lgbm_path)
            else:
                logger.error(f"LGBM component not found at {lgbm_path}")
                return False
            
            logger.info(f"Hybrid model loaded from {model_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
