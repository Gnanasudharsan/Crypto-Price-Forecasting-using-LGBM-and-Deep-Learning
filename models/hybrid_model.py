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
            logger.info("Training CNN
