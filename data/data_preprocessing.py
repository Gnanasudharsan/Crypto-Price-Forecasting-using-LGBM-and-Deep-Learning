#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for preprocessing cryptocurrency data.
Performs normalization, feature extraction, and prepares data for model training.
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
import glob
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load cryptocurrency data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        # Detect file format (different sources have different formats)
        df = pd.read_csv(file_path)
        
        # Check and standardize column names
        if 'Date' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
            
        if 'Date' not in df.columns and 'timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            
        # Convert Date to datetime if it's not already
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
        
        # Standard column set
        std_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume']
        
        # Map existing columns to standard columns
        column_mapping = {}
        
        if 'price' in df.columns or 'close' in df.columns:
            column_mapping['price' if 'price' in df.columns else 'close'] = 'Price'
            
        if 'open' in df.columns:
            column_mapping['open'] = 'Open'
            
        if 'high' in df.columns:
            column_mapping['high'] = 'High'
            
        if 'low' in df.columns:
            column_mapping['low'] = 'Low'
            
        if 'volume' in df.columns:
            column_mapping['volume'] = 'Volume'
        elif 'Vol.' in df.columns:
            column_mapping['Vol.'] = 'Volume'
            
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Check if we have the minimum required columns
        required_columns = ['Date', 'Price'] if 'Price' in df.columns else ['Date', 'Open', 'High', 'Low']
        
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in {file_path}")
                return None
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_data(df, output_path=None, scaler_path=None):
    """
    Preprocess cryptocurrency data using Min-Max normalization.
    
    Args:
        df: DataFrame with the data to preprocess
        output_path: Path to save the preprocessed data
        scaler_path: Path to save the scaler object
        
    Returns:
        DataFrame with the preprocessed data
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_processed = df.copy()
        
        # Check for missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Found missing values:\n{missing_values[missing_values > 0]}")
            logger.info("Filling missing values...")
            
            # Fill missing values
            # For numerical columns, use forward fill and then backward fill
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            # If any missing values remain, fill with column mean
            if df_processed.isnull().sum().sum() > 0:
                for col in numeric_cols:
                    if df_processed[col].isnull().sum() > 0:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        # Extract and keep the date column
        date_col = df_processed['Date'].copy()
        
        # Select only numerical columns for scaling
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Initialize the scaler
        scaler = MinMaxScaler()
        
        # Apply Min-Max scaling to transform values between 0 and 1
        df_scaled = df_processed.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        
        # Add back the date column
        df_scaled['Date'] = date_col
        
        # Save the scaler if a path is provided
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save the preprocessed data if an output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_scaled.to_csv(output_path, index=False)
            logger.info(f"Preprocessed data saved to {output_path}")
        
        return df_scaled
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Preprocess cryptocurrency data')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Directory containing raw cryptocurrency data files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory to save preprocessed data')
    parser.add_argument('--scaler-dir', type=str, default='models/scalers',
                        help='Directory to save scaler objects')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.scaler_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {args.input_dir}")
        return
    
    # Process each file
    for file_path in tqdm(csv_files, desc="Preprocessing files"):
        try:
            # Extract cryptocurrency name from filename
            crypto_name = os.path.basename(file_path).split('_')[0]
            
            logger.info(f"Processing {crypto_name} data from {file_path}")
            
            # Load data
            df = load_data(file_path)
            
            if df is not None:
                # Define output and scaler paths
                output_path = os.path.join(args.output_dir, f"{crypto_name}_processed.csv")
                scaler_path = os.path.join(args.scaler_dir, f"{crypto_name}_scaler.pkl")
                
                # Preprocess data
                df_processed = preprocess_data(df, output_path, scaler_path)
                
                if df_processed is not None:
                    logger.info(f"Successfully preprocessed {crypto_name} data")
                    
                    # Show sample of the preprocessed data
                    logger.info(f"Sample of preprocessed {crypto_name} data:")
                    logger.info(df_processed.head())
                else:
                    logger.error(f"Failed to preprocess {crypto_name} data")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
