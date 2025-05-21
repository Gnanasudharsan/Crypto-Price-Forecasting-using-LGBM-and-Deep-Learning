#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LGBM (Light Gradient Boosted Machine) model implementation for cryptocurrency price prediction.
This model is optimized for processing large volumes of data quickly.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
import joblib
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lgbm_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LGBMModel:
    """
    LGBM model for cryptocurrency price prediction.
    """
    
    def __init__(self, params=None):
        """
        Initialize the LGBM model with given parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        # Default parameters
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        # Use provided parameters or default ones
        self.params = params if params is not None else self.default_params
        
        # Model will be initialized during training
        self.model = None
        
        logger.info("LGBM model initialized with parameters:")
        for param, value in self.params.items():
            logger.info(f"  {param}: {value}")
    
    def reshape_for_lgbm(self, X):
        """
        Reshape input data for LGBM.
        
        Args:
            X: Input data with shape (samples, sequence_length, features)
            
        Returns:
            Reshaped data suitable for LGBM
        """
        # Flatten the sequence dimension and feature dimension
        # LGBM expects 2D data: (samples, features)
        samples, seq_len, features = X.shape
        return X.reshape(samples, seq_len * features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=1000, early_stopping_rounds=50):
        """
        Train the LGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            num_boost_round: Number of boosting iterations
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Trained model
        """
        try:
            logger.info("Reshaping data for LGBM...")
            
            # Reshape data
            X_train_reshaped = self.reshape_for_lgbm(X_train)
            
            # Create dataset for training
            lgb_train = lgb.Dataset(X_train_reshaped, y_train)
            
            # Create validation dataset if provided
            lgb_val = None
            if X_val is not None and y_val is not None:
                X_val_reshaped = self.reshape_for_lgbm(X_val)
                lgb_val = lgb.Dataset(X_val_reshaped, y_val, reference=lgb_train)
            
            # Train model
            logger.info("Training LGBM model...")
            self.model = lgb.train(
                self.params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_val] if lgb_val is not None else [lgb_train],
                valid_names=['train', 'validation'] if lgb_val is not None else ['train'],
                early_stopping_rounds=early_stopping_rounds if lgb_val is not None else None,
                verbose_eval=100
            )
            
            logger.info(f"Model trained with {self.model.best_iteration} iterations")
            
            # If validation data is provided, evaluate on it
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                logger.info(f"Validation RMSE: {rmse:.4f}")
                logger.info(f"Validation MAE: {mae:.4f}")
                logger.info(f"Validation R²: {r2:.4f}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training LGBM model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            # Reshape data
            X_reshaped = self.reshape_for_lgbm(X)
            
            # Make predictions
            predictions = self.model.predict(X_reshaped)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy (percentage of predictions within 5% of actual value)
            within_5_percent = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
            
            # Log evaluation results
            logger.info(f"Test RMSE: {rmse:.4f}")
            logger.info(f"Test MAE: {mae:.4f}")
            logger.info(f"Test R²: {r2:.4f}")
            logger.info(f"Predictions within 5% of actual: {within_5_percent:.2f}%")
            
            # Return metrics
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'within_5_percent': within_5_percent
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def save(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            self.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            # Load the model
            self.model = lgb.Booster(model_file=model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, params=None, model_path=None):
    """
    Train and evaluate an LGBM model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        params: Model parameters
        model_path: Path to save the model
        
    Returns:
        Trained model and evaluation metrics
    """
    # Initialize model
    model = LGBMModel(params)
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model if path is provided
    if model_path:
        model.save(model_path)
    
    return model, metrics

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate LGBM model for cryptocurrency price prediction')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing split data files')
    parser.add_argument('--output-dir', type=str, default='models/lgbm',
                        help='Directory to save the trained model')
    parser.add_argument('--crypto', type=str, default='bitcoin',
                        help='Cryptocurrency to train model for')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    crypto_dir = os.path.join(args.data_dir, args.crypto)
    
    X_train = np.load(os.path.join(crypto_dir, "X_train.npy"))
    y_train = np.load(os.path.join(crypto_dir, "y_train.npy"))
    X_val = np.load(os.path.join(crypto_dir, "X_val.npy"))
    y_val = np.load(os.path.join(crypto_dir, "y_val.npy"))
    X_test = np.load(os.path.join(crypto_dir, "X_test.npy"))
    y_test = np.load(os.path.join(crypto_dir, "y_test.npy"))
    
    # Define model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"{args.crypto}_lgbm_{timestamp}.txt")
    
    # Train and evaluate model
    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_path=model_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.crypto}_lgbm_{timestamp}_metrics.json")
    pd.DataFrame([metrics]).to_json(metrics_path, orient='records')
    logger.info(f"Metrics saved to {metrics_path}")
