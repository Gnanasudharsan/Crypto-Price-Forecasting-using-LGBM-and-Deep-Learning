#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask web application for cryptocurrency price prediction.
Provides a web interface for users to make real-time predictions.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import logging
from datetime import datetime, timedelta
import requests
import json
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directory to path to import model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model classes
from models.cnn_model import CNNModel
from models.lgbm_model import LGBMModel
from models.hybrid_model import HybridModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Dictionary to store loaded models
loaded_models = {}

# Dictionary mapping cryptocurrency names to symbols
CRYPTO_SYMBOLS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'ripple': 'XRP',
    'litecoin': 'LTC',
    'monero': 'XMR',
    'tether': 'USDT',
    'iota': 'MIOTA'
}

def fetch_historical_data(crypto_symbol, days=60):
    """
    Fetch historical cryptocurrency data from an API.
    
    Args:
        crypto_symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
        days: Number of days of historical data to fetch
        
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
            'limit': days,
            'api_key': 'YOUR_API_KEY'  # Replace with your actual API key
        }
        
        # Fetch data
        logger.info(f"Fetching data for {crypto_symbol} from CryptoCompare API")
        
        # Note: This would actually make an API call in a real implementation
        # For this demo, we'll simulate the response
        
        # Simulated data (in a real implementation, this would be the API response)
        # Generate some random price data for demonstration
        start_price = 50000 if crypto_symbol == 'BTC' else (
            2000 if crypto_symbol == 'ETH' else (
                1 if crypto_symbol == 'XRP' else (
                    100 if crypto_symbol == 'LTC' else (
                        200 if crypto_symbol == 'XMR' else (
                            1 if crypto_symbol == 'USDT' else 0.5
                        )
                    )
                )
            )
        )
        
        np.random.seed(42)  # For reproducible results
        
        dates = [datetime.now() - timedelta(days=i) for i in range(days, -1, -1)]
        prices = [start_price]
        
        for i in range(1, days + 1):
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

def preprocess_data(df, crypto):
    """
    Preprocess the data for prediction.
    
    Args:
        df: DataFrame with cryptocurrency data
        crypto: Name of the cryptocurrency
        
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
        
        # Normalize data if scaler is available
        scaler_path = f"../models/scalers/{crypto}_scaler.pkl"
        
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            
            # Apply scaler
            df[feature_cols] = scaler.transform(df[feature_cols])
        else:
            logger.warning(f"Scaler not found at {scaler_path}. Using data without normalization.")
        
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

def load_model(crypto, model_type):
    """
    Load a trained model if not already loaded.
    
    Args:
        crypto: Name of the cryptocurrency
        model_type: Type of model to load ('cnn', 'lgbm', or 'hybrid')
        
    Returns:
        Loaded model or None if loading fails
    """
    # Create a key for the model
    model_key = f"{crypto}_{model_type}"
    
    # Check if model is already loaded
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    try:
        # Define model directories
        model_base_dir = f"../models/{model_type}"
        
        # Find the most recent model directory for this cryptocurrency
        model_dirs = [d for d in os.listdir(model_base_dir) 
                     if os.path.isdir(os.path.join(model_base_dir, d)) and d.startswith(f"{crypto}_")]
        
        if not model_dirs:
            logger.error(f"No {model_type} models found for {crypto}")
            return None
        
        # Sort by timestamp (assuming directory names end with timestamp)
        model_dirs.sort(reverse=True)
        model_dir = os.path.join(model_base_dir, model_dirs[0])
        
        logger.info(f"Loading {model_type} model for {crypto} from {model_dir}")
        
        # Load the model based on its type
        if model_type == 'cnn':
            model_path = os.path.join(model_dir, "cnn_model.h5")
            if os.path.exists(model_path):
                model = CNNModel(input_shape=(60, 5))  # Example input shape
                model.load(model_path)
                loaded_models[model_key] = model
                return model
            else:
                logger.error(f"CNN model not found at {model_path}")
                return None
        
        elif model_type == 'lgbm':
            model_path = os.path.join(model_dir, "lgbm_model.txt")
            if os.path.exists(model_path):
                model = LGBMModel()
                model.load(model_path)
                loaded_models[model_key] = model
                return model
            else:
                logger.error(f"LGBM model not found at {model_path}")
                return None
        
        elif model_type == 'hybrid':
            if os.path.exists(model_dir):
                model = HybridModel(input_shape=(60, 5))  # Example input shape
                success = model.load(model_dir)
                if success:
                    loaded_models[model_key] = model
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

def make_prediction(crypto, model_type):
    """
    Make a price prediction for the specified cryptocurrency.
    
    Args:
        crypto: Name of the cryptocurrency
        model_type: Type of model to use
        
    Returns:
        Dictionary with prediction results or None if prediction fails
    """
    try:
        # Get cryptocurrency symbol
        symbol = CRYPTO_SYMBOLS.get(crypto.lower(), 'BTC')
        
        # Fetch historical data
        df = fetch_historical_data(symbol)
        
        if df is None:
            logger.error(f"Failed to fetch data for {crypto}")
            return None
        
        # Preprocess data
        data = preprocess_data(df, crypto)
        
        if data is None:
            logger.error(f"Failed to preprocess data for {crypto}")
            return None
        
        # Load model
        model = load_model(crypto, model_type)
        
        if model is None:
            logger.error(f"Failed to load {model_type} model for {crypto}")
            return None
        
        # Make prediction
        logger.info(f"Making prediction for {crypto} using {model_type} model")
        
        prediction = model.predict(data['sequence'])
        
        # Get the last known price and calculate predicted change
        last_price = data['original_data']['Price'].iloc[-1]
        
        # Calculate predicted price change
        price_change = prediction - last_price
        price_change_percent = (price_change / last_price) * 100
        
        # Prepare result
        result = {
            'cryptocurrency': crypto.capitalize(),
            'symbol': symbol,
            'current_price': float(last_price),
            'predicted_price': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
            'price_change': float(price_change),
            'price_change_percent': float(price_change_percent),
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': model_type
        }
        
        logger.info(f"Prediction result: {result}")
        
        # Generate and save the prediction plot
        plot_filename = f"{crypto}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = os.path.join('static', 'plots', plot_filename)
        
        # Ensure the plots directory exists
        os.makedirs(os.path.join('static', 'plots'), exist_ok=True)
        
        # Create the plot
        plot_prediction(data, result, plot_path)
        
        # Add plot path to result
        result['plot_path'] = plot_path
        
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
        plt.title(f"{result['cryptocurrency']} Price Prediction ({result['model_type'].upper()} Model)")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format date axis
        date_formatter = DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.xticks(rotation=45)
        
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

# Define routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', cryptos=list(CRYPTO_SYMBOLS.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        crypto = request.form.get('cryptocurrency', 'bitcoin')
        model_type = request.form.get('model_type', 'hybrid')
        
        # Make prediction
        result = make_prediction(crypto, model_type)
        
        if result is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/compare', methods=['POST
