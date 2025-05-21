#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for making cryptocurrency price predictions using trained models.
Can be used for real-time predictions or for generating predictions for a specific time range.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import requests
import json
import joblib

# Import model classes
from models.cnn_model import CNNModel
from models.lgbm_model import LGBMModel
from models.hybrid_model import HybridModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("price_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_current_data(crypto_symbol, num_days=60):
    """
    Fetch current cryptocurrency data from an API.
    
    Args:
        crypto_symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
        num_days: Number of past days to fetch
        
    Returns:
        DataFrame with historical data
    """
    try:
        # URL for CryptoCompare API
        # Note: In a real implementation, you might want to use a different API
        # or implement proper error handling and rate limiting
        url = f"https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': crypto_symbol,
            'tsym': 'USD',
            'limit': num_days,
            'api_key': 'YOUR_API_KEY'  # Replace with your actual API key
        }
        
        # Fetch data
        logger.info(f"Fetching data for {crypto_symbol} from CryptoCompare API")
        
        # Note: This would actually make an API call in a real implementation
        # For this demo, we'll simulate the response
        
        # Simulated data (in a real implementation, this would be the API response)
        # Generate some random price data for demonstration
        start_price = 50000 if crypto_symbol == 'BTC' else 2000  # Example starting prices
        np.random.seed(42)  # For reproducible results
        
        dates = [datetime.now() - timedelta(days=i) for i in range(num_days, -1, -1)]
        prices = [start_price]
        
        for i in range(1, num_days + 1):
            # Random daily change between -3% and +3%
            change = np.random.uniform(-0.03, 0.03)
            prices.append(prices[-1] * (1 + change))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Open': [price * np.random.uniform(0.98, 1.0) for price in prices],
            'High': [price * np.random.uniform(1.0, 1.05) for price in prices],
            'Low': [price * np.random.uniform(0.95, 1.0) for price in prices],
            'Volume': [np.random.uniform(1e9, 5e9) for _ in prices]
        })
        
        logger.info(f"Data fetched successfully. Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def preprocess_data(df, scaler_path=None):
    """
    Preprocess the data for prediction.
    
    Args:
        df: DataFrame with cryptocurrency data
        scaler_path: Path to the saved scaler
        
    Returns:
        Processed data ready for prediction
    """
    try:
        # Keep a copy of the original DataFrame
        df_original = df.copy()
        
        # Select features
        feature_cols = ['Price', 'Open', 'High', 'Low', 'Volume']
        
        # Make sure all required columns exist
        for col in feature_cols:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in data")
                return None
        
        # Normalize data if scaler is provided
        if scaler_path and os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            
            # Apply scaler
            df[feature_cols] = scaler.transform(df[feature_cols])
        
        # Create sequences for prediction
        sequence_length = 60  # Same as used during training
        
        if len(df) < sequence_length:
            logger.error(f"Not enough data points. Need at least {sequence_length}, got {len(df)}")
            return None
        
        # Use the last sequence_length data points for prediction
        prediction_sequence = df[feature_cols].values[-sequence_length:]
        
        # Reshape for model input (add batch dimension)
        prediction_sequence = np.expand_dims(prediction_sequence, axis=0)
        
        return {
            'sequence': prediction_sequence,
            'original_data': df_original,
            'feature_cols': feature_cols
        }
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

def load_model(model_dir, model_type):
    """
    Load a trained model.
    
    Args:
        model_dir: Directory containing the model
        model_type: Type of model to load ('cnn', 'lgbm', or 'hybrid')
        
    Returns:
        Loaded model
    """
    try:
        if model_type == 'cnn':
            model_path = os.path.join(model_dir, "cnn_model.h5")
            if os.path.exists(model_path):
                model = CNNModel(input_shape=(60, 5))  # Example input shape
                model.load(model_path)
                return model
            else:
                logger.error(f"CNN model not found at {model_path}")
                return None
        
        elif model_type == 'lgbm':
            model_path = os.path.join(model_dir, "lgbm_model.txt")
            if os.path.exists(model_path):
                model = LGBMModel()
                model.load(model_path)
                return model
            else:
                logger.error(f"LGBM model not found at {model_path}")
                return None
        
        elif model_type == 'hybrid':
            if os.path.exists(model_dir):
                model = HybridModel(input_shape=(60, 5))  # Example input shape
                success = model.load(model_dir)
                if success:
                    return model
                else:
                    logger.error(f"Failed to load hybrid model from {model_dir}")
                    return None
            else:
                logger.error(f"Hybrid model directory not found: {model_dir}")
                return None
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def make_prediction(model, data, model_type, scaler_path=None):
    """
    Make price prediction using the loaded model.
    
    Args:
        model: Loaded model
        data: Preprocessed data
        model_type: Type of model ('cnn', 'lgbm', or 'hybrid')
        scaler_path: Path to the saved scaler
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Make prediction
        logger.info(f"Making prediction using {model_type} model")
        
        prediction = model.predict(data['sequence'])
        
        # If scaler is provided, inverse transform the prediction
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
            # Create a dummy row with zeros
            dummy = np.zeros((1, len(data['feature_cols'])))
            
            # Replace the Price value with our prediction
            dummy[0, data['feature_cols'].index('Price')] = prediction[0]
            
            # Inverse transform
            dummy = scaler.inverse_transform(dummy)
            
            # Get the Price prediction
            prediction = dummy[0, data['feature_cols'].index('Price')]
        
        # Get the last known price
        last_price = data['original_data']['Price'].iloc[-1]
        
        # Calculate predicted price change
        price_change = prediction - last_price
        price_change_percent = (price_change / last_price) * 100
        
        # Prepare result
        result = {
            'current_price': last_price,
            'predicted_price': prediction,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def plot_prediction(data, result, output_path):
    """
    Plot historical data and prediction.
    
    Args:
        data: Preprocessed data
        result: Prediction result
        output_path: Path to save the plot
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot historical prices
        historical_dates = data['original_data']['Date']
        historical_prices = data['original_data']['Price']
        
        plt.plot(historical_dates, historical_prices, label='Historical Price')
        
        # Add prediction point
        prediction_date = datetime.now() + timedelta(days=1)
        plt.scatter(prediction_date, result['predicted_price'], color='red', s=50, label='Prediction')
        
        # Connect the last actual price with the prediction
        plt.plot([historical_dates.iloc[-1], prediction_date], 
                [historical_prices.iloc[-1], result['predicted_price']], 
                'r--', alpha=0.7)
        
        # Add annotations
        plt.annotate(f"Predicted: ${result['predicted_price']:.2f}", 
                    (prediction_date, result['predicted_price']),
                    xytext=(10, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        # Format plot
        plt.title('Cryptocurrency Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add prediction details as text
        text = (f"Current Price: ${result['current_price']:.2f}\n"
                f"Predicted Price: ${result['predicted_price']:.2f}\n"
                f"Change: ${result['price_change']:.2f} ({result['price_change_percent']:.2f}%)\n"
                f"Prediction Time: {result['prediction_time']}")
        
        plt.annotate(text, xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error plotting prediction: {e}")

def main():
    parser = argparse.ArgumentParser(description='Make cryptocurrency price predictions')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'lgbm', 'hybrid'], required=True,
                        help='Type of model to use')
    parser.add_argument('--crypto', type=str, default='bitcoin',
                        help='Cryptocurrency to predict')
    parser.add_argument('--symbol', type=str, default='BTC',
                        help='Symbol of the cryptocurrency for API requests')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to the saved scaler')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fetch current data
    df = fetch_current_data(args.symbol)
    
    if df is None:
        logger.error("Failed to fetch data. Exiting.")
        return
    
    # Preprocess data
    data = preprocess_data(df, args.scaler_path)
    
    if data is None:
        logger.error("Failed to preprocess data. Exiting.")
        return
    
    # Load model
    model = load_model(args.model_dir, args.model_type)
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Make prediction
    result = make_prediction(model, data, args.model_type, args.scaler_path)
    
    if result is None:
        logger.error("Failed to make prediction. Exiting.")
        return
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save prediction result
    result_path = os.path.join(args.output_dir, f"{args.crypto}_{timestamp}_prediction.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    logger.info(f"Prediction result saved to {result_path}")
    
    # Plot prediction
    plot_path = os.path.join(args.output_dir, f"{args.crypto}_{timestamp}_prediction.png")
    plot_prediction(data, result, plot_path)
    
    # Print prediction
    print("\n===== Cryptocurrency Price Prediction =====")
    print(f"Cryptocurrency: {args.crypto.capitalize()}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Change: ${result['price_change']:.2f} ({result['price_change_percent']:.2f}%)")
    print(f"Prediction Time: {result['prediction_time']}")
    print(f"Prediction Plot: {plot_path}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
