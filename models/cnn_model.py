#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN (Convolutional Neural Network) model implementation for cryptocurrency price prediction.
This model is designed to capture short-term patterns in cryptocurrency price data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cnn_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CNNModel:
    """
    CNN model for cryptocurrency price prediction.
    """
    
    def __init__(self, input_shape, output_units=1):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_units: Number of output units
        """
        self.input_shape = input_shape
        self.output_units = output_units
        self.model = self._build_model()
        
        logger.info(f"CNN model initialized with input shape {input_shape} and {output_units} output units")
    
    def _build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First Conv layer
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            
            # Second Conv layer
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            # Third Conv layer
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.output_units, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, patience=20, model_path=None):
        """
        Train the CNN model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            model_path: Path to save the best model
            
        Returns:
            Training history
        """
        try:
            # Prepare callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # Model checkpoint if model_path is provided
            if model_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                checkpoint = ModelCheckpoint(
                    model_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)
            
            # Train model
            logger.info("Training CNN model...")
            
            # Reshape y if needed (output_units > 1)
            if self.output_units > 1:
                if len(y_train.shape) == 1:
                    y_train = np.expand_dims(y_train, axis=-1)
                if X_val is not None and len(y_val.shape) == 1:
                    y_val = np.expand_dims(y_val, axis=-1)
            
            # Fit model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                callbacks=callbacks,
                verbose=2
            )
            
            logger.info("Model training completed")
            
            # Evaluate on validation data if available
            if X_val is not None and y_val is not None:
                val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                logger.info(f"Validation MAE: {val_mae:.4f}")
            
            return history
        
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        try:
            # Make predictions
            predictions = self.model.predict(X)
            
            # Reshape if needed
            if self.output_units == 1:
                predictions = predictions.flatten()
            
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
            
            # Model evaluation
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test MAE: {test_mae:.4f}")
            
            # Return metrics
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'within_5_percent': within_5_percent,
                'loss': test_loss
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
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            self.model.save(model_path)
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
            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, input_shape=None, output_units=1, model_path=None):
    """
    Train and evaluate a CNN model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        input_shape: Shape of input data
        output_units: Number of output units
        model_path: Path to save the model
        
    Returns:
        Trained model and evaluation metrics
    """
    # Determine input shape if not provided
    if input_shape is None:
        input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Initialize model
    model = CNNModel(input_shape, output_units)
    
    # Train model
    model.train(X_train, y_train, X_val, y_val, model_path=model_path)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate CNN model for cryptocurrency price prediction')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing split data files')
    parser.add_argument('--output-dir', type=str, default='models/cnn',
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
    model_path = os.path.join(args.output_dir, f"{args.crypto}_cnn_{timestamp}.h5")
    
    # Train and evaluate model
    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_path=model_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.crypto}_cnn_{timestamp}_metrics.json")
    pd.DataFrame([metrics]).to_json(metrics_path, orient='records')
    logger.info(f"Metrics saved to {metrics_path}")
