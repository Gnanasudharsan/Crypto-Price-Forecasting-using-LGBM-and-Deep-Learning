#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for splitting preprocessed cryptocurrency data into training and testing sets.
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
import glob
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_splitting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sequences(data, sequence_length):
    """
    Create sequences from time series data for deep learning models.
    
    Args:
        data: NumPy array of features
        sequence_length: Length of each sequence
        
    Returns:
        X: Sequences (inputs)
        y: Targets (outputs)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, target_col='Price', 
               sequence_length=60, feature_cols=None):
    """
    Split time series data into training, validation, and test sets.
    
    Args:
        df: DataFrame with preprocessed data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        target_col: Target column for prediction
        sequence_length: Length of each sequence
        feature_cols: List of feature columns to use
        
    Returns:
        Dictionary with training, validation, and test data
    """
    try:
        # Validate ratios
        if train_ratio + val_ratio + test_ratio != 1.0:
            logger.warning("Ratios do not sum to 1.0. Normalizing...")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Use all numeric columns if feature_cols is not specified
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column from feature_cols if it's there
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # Ensure the target column exists
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame")
            return None
        
        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Determine split indices
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        X_train, y_train = create_sequences(features[:train_end], sequence_length)
        X_val, y_val = create_sequences(features[train_end:val_end], sequence_length)
        X_test, y_test = create_sequences(features[val_end:], sequence_length)
        
        # Create a dictionary with the split data
        split_data = {
            'train': {
                'X': X_train,
                'y': y_train[:, features_cols.index(target_col) if target_col in features_cols else 0]
            },
            'val': {
                'X': X_val,
                'y': y_val[:, features_cols.index(target_col) if target_col in features_cols else 0]
            },
            'test': {
                'X': X_test,
                'y': y_test[:, features_cols.index(target_col) if target_col in features_cols else 0]
            },
            'feature_cols': feature_cols,
            'target_col': target_col,
            'sequence_length': sequence_length
        }
        
        logger.info(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        return split_data
    
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Split cryptocurrency data into train, validation, and test sets')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                        help='Directory containing preprocessed cryptocurrency data files')
    parser.add_argument('--output-dir', type=str, default='data/split',
                        help='Directory to save split data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of data for testing')
    parser.add_argument('--target-col', type=str, default='Price',
                        help='Target column for prediction')
    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Length of each sequence for LSTM/CNN models')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        logger.warning(f"Ratios (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}) "
                     f"do not sum to 1.0 (sum={total_ratio}). Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(args.input_dir, "*_processed.csv"))
    
    if not csv_files:
        logger.error(f"No processed CSV files found in {args.input_dir}")
        return
    
    # Process each file
    for file_path in tqdm(csv_files, desc="Splitting data"):
        try:
            # Extract cryptocurrency name from filename
            crypto_name = os.path.basename(file_path).split('_')[0]
            
            logger.info(f"Splitting {crypto_name} data from {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Split data
            split_result = split_data(
                df, 
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                target_col=args.target_col,
                sequence_length=args.sequence_length
            )
            
            if split_result is not None:
                # Create a directory for this cryptocurrency
                crypto_dir = os.path.join(args.output_dir, crypto_name)
                os.makedirs(crypto_dir, exist_ok=True)
                
                # Save the split data
                np.save(os.path.join(crypto_dir, "X_train.npy"), split_result['train']['X'])
                np.save(os.path.join(crypto_dir, "y_train.npy"), split_result['train']['y'])
                np.save(os.path.join(crypto_dir, "X_val.npy"), split_result['val']['X'])
                np.save(os.path.join(crypto_dir, "y_val.npy"), split_result['val']['y'])
                np.save(os.path.join(crypto_dir, "X_test.npy"), split_result['test']['X'])
                np.save(os.path.join(crypto_dir, "y_test.npy"), split_result['test']['y'])
                
                # Save metadata
                metadata = {
                    'feature_cols': split_result['feature_cols'],
                    'target_col': split_result['target_col'],
                    'sequence_length': split_result['sequence_length'],
                    'train_samples': len(split_result['train']['X']),
                    'val_samples': len(split_result['val']['X']),
                    'test_samples': len(split_result['test']['X'])
                }
                
                pd.DataFrame([metadata]).to_csv(os.path.join(crypto_dir, "metadata.csv"), index=False)
                
                logger.info(f"Split data saved to {crypto_dir}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
