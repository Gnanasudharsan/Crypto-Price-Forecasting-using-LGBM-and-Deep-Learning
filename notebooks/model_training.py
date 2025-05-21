#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for training cryptocurrency price prediction models.
Supports training CNN, LGBM, and Hybrid models.
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Import model classes
from models.cnn_model import CNNModel
from models.lgbm_model import LGBMModel
from models.hybrid_model import HybridModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_training_history(history, output_path):
    """
    Plot training and validation loss.
    
    Args:
        history: Training history
        output_path: Path to save the plot
    """
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history['mean_absolute_error'])
    plt.plot(history['val_mean_absolute_error'])
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Training history plot saved to {output_path}")

def plot_predictions(y_true, y_pred, title, output_path):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Predictions plot saved to {output_path}")

def train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, args):
    """
    Train and evaluate a CNN model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        args: Command-line arguments
        
    Returns:
        Trained model and evaluation metrics
    """
    # Determine input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"cnn_{args.crypto}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Model path
    model_path = os.path.join(model_dir, "cnn_model.h5")
    
    # Initialize model
    model = CNNModel(input_shape)
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_path=model_path
    )
    
    # Plot training history
    if history:
        plot_training_history(
            history.history,
            os.path.join(model_dir, "training_history.png")
        )
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate predictions and plot
    y_pred = model.predict(X_test)
    plot_predictions(
        y_test, y_pred,
        f"CNN Model Predictions for {args.crypto.capitalize()}",
        os.path.join(model_dir, "predictions.png")
    )
    
    return model, metrics

def train_lgbm_model(X_train, y_train, X_val, y_val, X_test, y_test, args):
    """
    Train and evaluate an LGBM model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        args: Command-line arguments
        
    Returns:
        Trained model and evaluation metrics
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"lgbm_{args.crypto}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Model path
    model_path = os.path.join(model_dir, "lgbm_model.txt")
    
    # Define LGBM parameters
    lgbm_params = {
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
    
    # Initialize model
    model = LGBMModel(lgbm_params)
    
    # Train model
    model.train(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds
    )
    
    # Save model
    model.save(model_path)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate predictions and plot
    y_pred = model.predict(X_test)
    plot_predictions(
        y_test, y_pred,
        f"LGBM Model Predictions for {args.crypto.capitalize()}",
        os.path.join(model_dir, "predictions.png")
    )
    
    return model, metrics

def train_hybrid_model(X_train, y_train, X_val, y_val, X_test, y_test, args):
    """
    Train and evaluate a hybrid model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        args: Command-line arguments
        
    Returns:
        Trained model and evaluation metrics
    """
    # Determine input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"hybrid_{args.crypto}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Define LGBM parameters
    lgbm_params = {
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
    
    # Initialize model
    model = HybridModel(input_shape, lgbm_params)
    
    # Train model
    model.train(
        X_train, y_train,
        X_val, y_val,
        cnn_epochs=args.epochs,
        cnn_batch_size=args.batch_size,
        cnn_patience=args.patience,
        lgbm_num_boost_round=args.num_boost_round,
        lgbm_early_stopping_rounds=args.early_stopping_rounds,
        model_dir=model_dir
    )
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate predictions and plot
    y_pred = model.predict(X_test)
    plot_predictions(
        y_test, y_pred,
        f"Hybrid Model Predictions for {args.crypto.capitalize()}",
        os.path.join(model_dir, "predictions.png")
    )
    
    # Plot component model predictions
    cnn_pred = model.cnn_model.predict(X_test)
    lgbm_pred = model.lgbm_model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(cnn_pred, label='CNN')
    plt.plot(lgbm_pred, label='LGBM')
    plt.plot(y_pred, label='Hybrid')
    plt.title(f"Component Model Predictions for {args.crypto.capitalize()}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "component_predictions.png"))
    plt.close()
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction models')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing split data files')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--crypto', type=str, default='bitcoin',
                        help='Cryptocurrency to train model for')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'lgbm', 'hybrid'], default='hybrid',
                        help='Type of model to train')
    
    # CNN parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for CNN training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for CNN training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience for CNN')
    
    # LGBM parameters
    parser.add_argument('--num-boost-round', type=int, default=1000,
                        help='Number of boosting rounds for LGBM')
    parser.add_argument('--early-stopping-rounds', type=int, default=50,
                        help='Early stopping rounds for LGBM')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    crypto_dir = os.path.join(args.data_dir, args.crypto)
    
    if not os.path.exists(crypto_dir):
        logger.error(f"Data directory for {args.crypto} not found: {crypto_dir}")
        return
    
    logger.info(f"Loading {args.crypto} data from {crypto_dir}")
    
    X_train = np.load(os.path.join(crypto_dir, "X_train.npy"))
    y_train = np.load(os.path.join(crypto_dir, "y_train.npy"))
    X_val = np.load(os.path.join(crypto_dir, "X_val.npy"))
    y_val = np.load(os.path.join(crypto_dir, "y_val.npy"))
    X_test = np.load(os.path.join(crypto_dir, "X_test.npy"))
    y_test = np.load(os.path.join(crypto_dir, "y_test.npy"))
    
    logger.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, "
               f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
               f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Train selected model
    if args.model_type == 'cnn':
        logger.info("Training CNN model...")
        model, metrics = train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, args)
    elif args.model_type == 'lgbm':
        logger.info("Training LGBM model...")
        model, metrics = train_lgbm_model(X_train, y_train, X_val, y_val, X_test, y_test, args)
    else:  # hybrid
        logger.info("Training Hybrid model...")
        model, metrics = train_hybrid_model(X_train, y_train, X_val, y_val, X_test, y_test, args)
    
    # Log results
    logger.info(f"Model training completed. Metrics: {metrics}")

if __name__ == "__main__":
    main()
