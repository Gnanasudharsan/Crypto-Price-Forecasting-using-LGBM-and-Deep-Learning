#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for feature engineering on cryptocurrency price data.
Creates technical indicators and additional features to improve model performance.
"""

import numpy as np
import pandas as pd
import argparse
import os
import logging
import glob
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def add_price_features(df):
    """
    Add price-based features to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional price features
    """
    logger.info("Adding price-based features...")
    
    # Make sure we have the required columns
    required_cols = ['Price', 'Open', 'High', 'Low']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column {col} not found in DataFrame")
            return df
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Simple Returns
    df_features['Daily_Return'] = df_features['Price'].pct_change() * 100
    
    # Logarithmic Returns
    df_features['Log_Return'] = np.log(df_features['Price'] / df_features['Price'].shift(1)) * 100
    
    # Price Momentum (percentage changes over different periods)
    for period in [2, 3, 5, 7, 14, 21, 30]:
        df_features[f'Price_Momentum_{period}d'] = df_features['Price'].pct_change(periods=period) * 100
    
    # Price Range
    df_features['Daily_Range'] = df_features['High'] - df_features['Low']
    df_features['Daily_Range_Pct'] = (df_features['High'] - df_features['Low']) / df_features['Open'] * 100
    
    # Gap (difference between current open and previous close)
    df_features['Gap'] = df_features['Open'] - df_features['Price'].shift(1)
    df_features['Gap_Pct'] = df_features['Gap'] / df_features['Price'].shift(1) * 100
    
    # Candlestick Pattern Features
    df_features['Candle_Body'] = df_features['Price'] - df_features['Open']
    df_features['Candle_Body_Pct'] = df_features['Candle_Body'] / df_features['Open'] * 100
    df_features['Upper_Shadow'] = df_features['High'] - df_features[['Open', 'Price']].max(axis=1)
    df_features['Lower_Shadow'] = df_features[['Open', 'Price']].min(axis=1) - df_features['Low']
    df_features['Upper_Shadow_Pct'] = df_features['Upper_Shadow'] / df_features['Open'] * 100
    df_features['Lower_Shadow_Pct'] = df_features['Lower_Shadow'] / df_features['Open'] * 100
    
    logger.info("Added price-based features")
    
    return df_features

def add_moving_averages(df):
    """
    Add moving average features to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional moving average features
    """
    logger.info("Adding moving average features...")
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Simple Moving Averages (SMA)
    for period in [5, 10, 20, 30, 50, 100, 200]:
        df_features[f'SMA_{period}'] = df_features['Price'].rolling(window=period).mean()
        
        # Price relative to SMA (shows if price is above or below average)
        df_features[f'Price_Rel_SMA_{period}'] = df_features['Price'] / df_features[f'SMA_{period}']
        
        # SMA percent change
        df_features[f'SMA_{period}_Pct_Change'] = df_features[f'SMA_{period}'].pct_change() * 100
        
    # Exponential Moving Averages (EMA)
    for period in [5, 10, 20, 30, 50, 100, 200]:
        df_features[f'EMA_{period}'] = df_features['Price'].ewm(span=period, adjust=False).mean()
        
        # Price relative to EMA
        df_features[f'Price_Rel_EMA_{period}'] = df_features['Price'] / df_features[f'EMA_{period}']
        
        # EMA percent change
        df_features[f'EMA_{period}_Pct_Change'] = df_features[f'EMA_{period}'].pct_change() * 100
    
    # Moving Average Convergence Divergence (MACD)
    df_features['EMA_12'] = df_features['Price'].ewm(span=12, adjust=False).mean()
    df_features['EMA_26'] = df_features['Price'].ewm(span=26, adjust=False).mean()
    df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    df_features['MACD_Histogram'] = df_features['MACD'] - df_features['MACD_Signal']
    
    # Moving Average Crossovers
    df_features['SMA_10_50_Crossover'] = np.where(
        df_features['SMA_10'] > df_features['SMA_50'], 1, 
        np.where(df_features['SMA_10'] < df_features['SMA_50'], -1, 0)
    )
    
    df_features['SMA_20_100_Crossover'] = np.where(
        df_features['SMA_20'] > df_features['SMA_100'], 1, 
        np.where(df_features['SMA_20'] < df_features['SMA_100'], -1, 0)
    )
    
    df_features['SMA_50_200_Crossover'] = np.where(
        df_features['SMA_50'] > df_features['SMA_200'], 1, 
        np.where(df_features['SMA_50'] < df_features['SMA_200'], -1, 0)
    )
    
    logger.info("Added moving average features")
    
    return df_features

def add_volatility_features(df):
    """
    Add volatility-based features to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional volatility features
    """
    logger.info("Adding volatility features...")
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Make sure we have daily returns
    if 'Daily_Return' not in df_features.columns:
        df_features['Daily_Return'] = df_features['Price'].pct_change() * 100
    
    # Rolling volatility (standard deviation of returns)
    for period in [5, 10, 20, 30, 60]:
        df_features[f'Volatility_{period}d'] = df_features['Daily_Return'].rolling(window=period).std()
    
    # Average True Range (ATR)
    df_features['TR1'] = abs(df_features['High'] - df_features['Low'])
    df_features['TR2'] = abs(df_features['High'] - df_features['Price'].shift(1))
    df_features['TR3'] = abs(df_features['Low'] - df_features['Price'].shift(1))
    df_features['True_Range'] = df_features[['TR1', 'TR2', 'TR3']].max(axis=1)
    df_features.drop(['TR1', 'TR2', 'TR3'], axis=1, inplace=True)
    
    for period in [5, 14, 30]:
        df_features[f'ATR_{period}'] = df_features['True_Range'].rolling(window=period).mean()
        df_features[f'ATR_{period}_Pct'] = df_features[f'ATR_{period}'] / df_features['Price'] * 100
    
    # Bollinger Bands
    for period in [20]:
        middle_band = df_features['Price'].rolling(window=period).mean()
        std_dev = df_features['Price'].rolling(window=period).std()
        
        df_features[f'BB_Middle_{period}'] = middle_band
        df_features[f'BB_Upper_{period}'] = middle_band + (std_dev * 2)
        df_features[f'BB_Lower_{period}'] = middle_band - (std_dev * 2)
        df_features[f'BB_Width_{period}'] = (df_features[f'BB_Upper_{period}'] - df_features[f'BB_Lower_{period}']) / df_features[f'BB_Middle_{period}']
        df_features[f'BB_Pct_B_{period}'] = (df_features['Price'] - df_features[f'BB_Lower_{period}']) / (df_features[f'BB_Upper_{period}'] - df_features[f'BB_Lower_{period}'])
    
    # Normalized Volatility (Volatility / Price)
    for period in [5, 10, 20, 30, 60]:
        df_features[f'Normalized_Volatility_{period}d'] = df_features[f'Volatility_{period}d'] / df_features['Price'] * 100
    
    logger.info("Added volatility features")
    
    return df_features

def add_volume_features(df):
    """
    Add volume-based features to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional volume features
    """
    logger.info("Adding volume features...")
    
    # Check if Volume column exists
    if 'Volume' not in df.columns:
        logger.warning("Volume column not found in DataFrame. Skipping volume features.")
        return df
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Volume change
    df_features['Volume_Change'] = df_features['Volume'].pct_change() * 100
    
    # Volume Moving Averages
    for period in [5, 10, 20, 50]:
        df_features[f'Volume_SMA_{period}'] = df_features['Volume'].rolling(window=period).mean()
        df_features[f'Volume_Rel_SMA_{period}'] = df_features['Volume'] / df_features[f'Volume_SMA_{period}']
    
    # Volume Indicators
    df_features['Price_Volume'] = df_features['Price'] * df_features['Volume']
    df_features['Price_Volume_SMA_5'] = df_features['Price_Volume'].rolling(window=5).mean()
    
    # On-Balance Volume (OBV)
    df_features['OBV_Change'] = np.where(
        df_features['Price'] > df_features['Price'].shift(1),
        df_features['Volume'],
        np.where(
            df_features['Price'] < df_features['Price'].shift(1),
            -df_features['Volume'],
            0
        )
    )
    df_features['OBV'] = df_features['OBV_Change'].cumsum()
    
    # Chaikin Money Flow (CMF)
    period = 20
    money_flow_multiplier = ((df_features['Price'] - df_features['Low']) - (df_features['High'] - df_features['Price'])) / (df_features['High'] - df_features['Low'])
    money_flow_volume = money_flow_multiplier * df_features['Volume']
    df_features['CMF'] = money_flow_volume.rolling(window=period).sum() / df_features['Volume'].rolling(window=period).sum()
    
    # Volume Oscillator
    df_features['Volume_EMA_5'] = df_features['Volume'].ewm(span=5, adjust=False).mean()
    df_features['Volume_EMA_10'] = df_features['Volume'].ewm(span=10, adjust=False).mean()
    df_features['Volume_Oscillator'] = ((df_features['Volume_EMA_5'] - df_features['Volume_EMA_10']) / df_features['Volume_EMA_10']) * 100
    
    # Volume Weighted Average Price (VWAP) - Simplified daily version
    df_features['Typical_Price'] = (df_features['High'] + df_features['Low'] + df_features['Price']) / 3
    df_features['TP_Volume'] = df_features['Typical_Price'] * df_features['Volume']
    df_features['VWAP_Daily'] = df_features['TP_Volume'].rolling(window=1).sum() / df_features['Volume'].rolling(window=1).sum()
    
    # Up/Down Volume
    df_features['Up_Volume'] = np.where(df_features['Price'] > df_features['Price'].shift(1), df_features['Volume'], 0)
    df_features['Down_Volume'] = np.where(df_features['Price'] < df_features['Price'].shift(1), df_features['Volume'], 0)
    df_features['Up_Volume_SMA_10'] = df_features['Up_Volume'].rolling(window=10).mean()
    df_features['Down_Volume_SMA_10'] = df_features['Down_Volume'].rolling(window=10).mean()
    df_features['Volume_Ratio'] = df_features['Up_Volume_SMA_10'] / df_features['Down_Volume_SMA_10']
    
    logger.info("Added volume features")
    
    return df_features

def add_momentum_indicators(df):
    """
    Add momentum indicators to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional momentum indicators
    """
    logger.info("Adding momentum indicators...")
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Relative Strength Index (RSI)
    for period in [6, 14, 30]:
        delta = df_features['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df_features[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    for period in [14]:
        # %K
        low_min = df_features['Low'].rolling(window=period).min()
        high_max = df_features['High'].rolling(window=period).max()
        
        df_features[f'Stochastic_%K_{period}'] = ((df_features['Price'] - low_min) / (high_max - low_min)) * 100
        
        # %D (3-day SMA of %K)
        df_features[f'Stochastic_%D_{period}'] = df_features[f'Stochastic_%K_{period}'].rolling(window=3).mean()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20, 50]:
        df_features[f'ROC_{period}'] = ((df_features['Price'] - df_features['Price'].shift(period)) / df_features['Price'].shift(period)) * 100
    
    # Commodity Channel Index (CCI)
    for period in [20]:
        typical_price = (df_features['High'] + df_features['Low'] + df_features['Price']) / 3
        moving_avg = typical_price.rolling(window=period).mean()
        mean_deviation = abs(typical_price - moving_avg).rolling(window=period).mean()
        
        df_features[f'CCI_{period}'] = (typical_price - moving_avg) / (0.015 * mean_deviation)
    
    # Williams %R
    for period in [14]:
        highest_high = df_features['High'].rolling(window=period).max()
        lowest_low = df_features['Low'].rolling(window=period).min()
        
        df_features[f'Williams_%R_{period}'] = ((highest_high - df_features['Price']) / (highest_high - lowest_low)) * (-100)
    
    # Average Directional Index (ADX)
    for period in [14]:
        # Calculate +DM and -DM
        df_features['UpMove'] = df_features['High'] - df_features['High'].shift(1)
        df_features['DownMove'] = df_features['Low'].shift(1) - df_features['Low']
        
        df_features['PlusDM'] = np.where(
            (df_features['UpMove'] > df_features['DownMove']) & (df_features['UpMove'] > 0),
            df_features['UpMove'],
            0
        )
        
        df_features['MinusDM'] = np.where(
            (df_features['DownMove'] > df_features['UpMove']) & (df_features['DownMove'] > 0),
            df_features['DownMove'],
            0
        )
        
        # Calculate True Range
        if 'True_Range' not in df_features.columns:
            df_features['TR1'] = abs(df_features['High'] - df_features['Low'])
            df_features['TR2'] = abs(df_features['High'] - df_features['Price'].shift(1))
            df_features['TR3'] = abs(df_features['Low'] - df_features['Price'].shift(1))
            df_features['True_Range'] = df_features[['TR1', 'TR2', 'TR3']].max(axis=1)
            df_features.drop(['TR1', 'TR2', 'TR3'], axis=1, inplace=True)
        
        # Calculate smoothed indicators
        df_features['ATR'] = df_features['True_Range'].rolling(window=period).mean()
        df_features['PlusDI'] = 100 * (df_features['PlusDM'].rolling(window=period).mean() / df_features['ATR'])
        df_features['MinusDI'] = 100 * (df_features['MinusDM'].rolling(window=period).mean() / df_features['ATR'])
        
        # Calculate DX and ADX
        df_features['DX'] = 100 * (abs(df_features['PlusDI'] - df_features['MinusDI']) / (df_features['PlusDI'] + df_features['MinusDI']))
        df_features[f'ADX_{period}'] = df_features['DX'].rolling(window=period).mean()
        
        # Clean up intermediate columns
        df_features.drop(['UpMove', 'DownMove', 'PlusDM', 'MinusDM', 'DX'], axis=1, inplace=True)
    
    logger.info("Added momentum indicators")
    
    return df_features

def add_cyclical_features(df):
    """
    Add cyclical time-based features to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with additional cyclical features
    """
    logger.info("Adding cyclical features...")
    
    # Check if Date column exists
    if 'Date' not in df.columns:
        logger.warning("Date column not found in DataFrame. Skipping cyclical features.")
        return df
    
    # Create copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Make sure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
        df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    # Extract time components
    df_features['Day_of_Week'] = df_features['Date'].dt.dayofweek
    df_features['Day_of_Month'] = df_features['Date'].dt.day
    df_features['Month'] = df_features['Date'].dt.month
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Quarter'] = df_features['Date'].dt.quarter
    
    # Add sinusoidal features to capture cyclical patterns
    
    # Day of week (cycle of 7)
    df_features['Day_of_Week_Sin'] = np.sin(2 * np.pi * df_features['Day_of_Week'] / 7)
    df_features['Day_of_Week_Cos'] = np.cos(2 * np.pi * df_features['Day_of_Week'] / 7)
    
    # Day of month (cycle of 30)
    df_features['Day_of_Month_Sin'] = np.sin(2 * np.pi * df_features['Day_of_Month'] / 30)
    df_features['Day_of_Month_Cos'] = np.cos(2 * np.pi * df_features['Day_of_Month'] / 30)
    
    # Month of year (cycle of 12)
    df_features['Month_Sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_Cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    
    # Quarter (cycle of 4)
    df_features['Quarter_Sin'] = np.sin(2 * np.pi * df_features['Quarter'] / 4)
    df_features['Quarter_Cos'] = np.cos(2 * np.pi * df_features['Quarter'] / 4)
    
    logger.info("Added cyclical features")
    
    return df_features

def engineer_features(df):
    """
    Apply all feature engineering functions to the DataFrame.
    
    Args:
        df: DataFrame with cryptocurrency price data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering process...")
    
    # Apply all feature functions
    df_features = df.copy()
    df_features = add_price_features(df_features)
    df_features = add_moving_averages(df_features)
    df_features = add_volatility_features(df_features)
    df_features = add_volume_features(df_features)
    df_features = add_momentum_indicators(df_features)
    df_features = add_cyclical_features(df_features)
    
    # Drop rows with NaN values (typically at the beginning due to rolling windows)
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    dropped_rows = initial_rows - len(df_features)
    
    logger.info(f"Feature engineering completed. Created {len(df_features.columns) - len(df.columns)} new features.")
    logger.info(f"Dropped {dropped_rows} rows with NaN values.")
    
    return df_features

def main():
    parser = argparse.ArgumentParser(description='Engineer features for cryptocurrency data')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                        help='Directory containing preprocessed cryptocurrency data files')
    parser.add_argument('--output-dir', type=str, default='data/features',
                        help='Directory to save feature-engineered data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(args.input_dir, "*_processed.csv"))
    
    if not csv_files:
        logger.error(f"No processed CSV files found in {args.input_dir}")
        return
    
    # Process each file
    for file_path in tqdm(csv_files, desc="Engineering features"):
        try:
            # Extract cryptocurrency name from filename
            crypto_name = os.path.basename(file_path).split('_')[0]
            
            logger.info(f"Processing {crypto_name} data from {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Ensure Date column is in datetime format if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Engineer features
            df_features = engineer_features(df)
            
            # Save feature-engineered data
            output_path = os.path.join(args.output_dir, f"{crypto_name}_features.csv")
            df_features.to_csv(output_path, index=False)
            
            logger.info(f"Feature-engineered data saved to {output_path}")
            
            # Print feature summary
            logger.info(f"Feature summary for {crypto_name}:")
            logger.info(f"Original features: {len(df.columns)}")
            logger.info(f"Engineered features: {len(df_features.columns)}")
            logger.info(f"Total features: {len(df_features.columns)}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
