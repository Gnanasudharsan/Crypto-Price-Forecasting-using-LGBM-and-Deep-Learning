#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for cryptocurrency price prediction models.
Provides comprehensive evaluation metrics to assess model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_basic_metrics(y_true, y_pred, prefix=''):
    """
    Calculate basic regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (useful when tracking multiple models)
        
    Returns:
        Dictionary with calculated metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Handle any zero values in y_true to avoid division by zero
    # Replace zeros with a small value
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    
    # Calculate percentage errors
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        # Manual calculation as fallback
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Calculate median absolute percentage error
    mdape = np.median(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Create metrics dictionary with optional prefix
    prefix = f"{prefix}_" if prefix else ""
    metrics = {
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}MAPE': mape,
        f'{prefix}MDAPE': mdape
    }
    
    return metrics

def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with directional accuracy metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate actual and predicted price movements
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate directional accuracy
    directional_accuracy = np.mean(true_direction == pred_direction) * 100
    
    # Calculate confusion matrix elements
    true_positives = np.sum((true_direction == True) & (pred_direction == True))
    false_positives = np.sum((true_direction == False) & (pred_direction == True))
    true_negatives = np.sum((true_direction == False) & (pred_direction == False))
    false_negatives = np.sum((true_direction == True) & (pred_direction == False))
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return metrics
    return {
        'Directional_Accuracy': directional_accuracy,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1_Score': f1_score * 100,
        'True_Positives': true_positives,
        'False_Positives': false_positives,
        'True_Negatives': true_negatives,
        'False_Negatives': false_negatives
    }

def calculate_threshold_accuracy(y_true, y_pred, thresholds=[0.01, 0.02, 0.05, 0.1]):
    """
    Calculate accuracy within various percentage thresholds.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        thresholds: List of threshold values (as decimal percentages)
        
    Returns:
        Dictionary with accuracy metrics at different thresholds
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle any zero values in y_true to avoid division by zero
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    
    # Calculate percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true_safe)
    
    # Calculate accuracy within each threshold
    threshold_metrics = {}
    for threshold in thresholds:
        within_threshold = np.mean(percentage_errors <= threshold) * 100
        threshold_metrics[f'Within_{int(threshold*100)}%'] = within_threshold
    
    return threshold_metrics

def calculate_trading_metrics(y_true, y_pred, initial_capital=10000, transaction_fee_pct=0.001):
    """
    Calculate trading strategy metrics based on model predictions.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        initial_capital: Initial investment amount
        transaction_fee_pct: Transaction fee as a percentage of trade amount
        
    Returns:
        Dictionary with trading metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate price changes
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    
    # Initialize variables for backtest
    capital = initial_capital
    position = 0  # 0: no position, 1: long position
    trades = 0
    profitable_trades = 0
    
    # Simple trading strategy: Buy when predicted change is positive, sell when negative
    capitals = [initial_capital]
    
    for i in range(len(true_changes)):
        pred_change = pred_changes[i]
        true_change = true_changes[i]
        
        # Trading logic
        if pred_change > 0 and position == 0:  # Buy signal
            position = 1
            trades += 1
            # Apply transaction fee
            capital *= (1 - transaction_fee_pct)
        elif pred_change <= 0 and position == 1:  # Sell signal
            position = 0
            trades += 1
            # Check if trade was profitable
            if true_change > 0:
                profitable_trades += 1
            # Apply transaction fee
            capital *= (1 - transaction_fee_pct)
        
        # Update capital based on position and actual price change
        if position == 1:
            # Update capital based on price change
            capital *= (1 + true_change / y_true[i])
        
        capitals.append(capital)
    
    # Calculate trading metrics
    final_capital = capitals[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    avg_return_per_trade = (final_capital / initial_capital) ** (1 / max(1, trades)) - 1
    
    # Calculate buy-and-hold return
    buy_and_hold_return = (y_true[-1] / y_true[0] - 1) * 100
    
    # Calculate Sharpe ratio
    daily_returns = np.diff(capitals) / np.array(capitals[:-1])
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Assuming 252 trading days per year
    
    # Calculate max drawdown
    running_max = np.maximum.accumulate(capitals)
    drawdowns = (running_max - capitals) / running_max
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    
    # Calculate win rate
    win_rate = profitable_trades / max(1, trades) * 100
    
    return {
        'Initial_Capital': initial_capital,
        'Final_Capital': final_capital,
        'Total_Return_Pct': total_return,
        'Buy_and_Hold_Return_Pct': buy_and_hold_return,
        'Outperformance_Pct': total_return - buy_and_hold_return,
        'Total_Trades': trades,
        'Win_Rate_Pct': win_rate,
        'Sharpe_Ratio': sharpe_ratio,
        'Annualized_Sharpe': annualized_sharpe,
        'Max_Drawdown_Pct': max_drawdown
    }

def calculate_time_horizon_metrics(y_true, y_pred, horizons=[1, 3, 5, 7, 14, 30]):
    """
    Calculate prediction accuracy for different time horizons.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        horizons: List of time horizons to evaluate
        
    Returns:
        Dictionary with metrics for different time horizons
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    horizon_metrics = {}
    
    for horizon in horizons:
        if len(y_true) <= horizon:
            logger.warning(f"Time horizon {horizon} exceeds data length. Skipping.")
            continue
        
        # Calculate errors for this horizon
        errors = []
        
        for i in range(len(y_true) - horizon):
            # Actual value after horizon steps
            future_true = y_true[i + horizon]
            
            # Current prediction
            current_pred = y_pred[i]
            
            # Current actual value
            current_true = y_true[i]
            
            # Predicted change (as percentage)
            pred_change_pct = (current_pred / current_true - 1) * 100
            
            # Actual change (as percentage)
            true_change_pct = (future_true / current_true - 1) * 100
            
            # Error in predicting change
            error = abs(pred_change_pct - true_change_pct)
            errors.append(error)
        
        # Calculate metrics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        
        # Store metrics
        horizon_metrics[f'Horizon_{horizon}_Mean_Error'] = mean_error
        horizon_metrics[f'Horizon_{horizon}_Median_Error'] = median_error
    
    return horizon_metrics

def evaluate_model(y_true, y_pred, model_name='Model', decimal_places=4):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model for logging
        decimal_places: Number of decimal places to round metrics
        
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Collect all metrics
    metrics = {}
    
    # Add basic metrics
    metrics.update(calculate_basic_metrics(y_true, y_pred))
    
    # Add directional accuracy metrics
    metrics.update(calculate_directional_accuracy(y_true, y_pred))
    
    # Add threshold accuracy metrics
    metrics.update(calculate_threshold_accuracy(y_true, y_pred))
    
    # Add trading metrics
    metrics.update(calculate_trading_metrics(y_true, y_pred))
    
    # Add time horizon metrics
    metrics.update(calculate_time_horizon_metrics(y_true, y_pred))
    
    # Round all metrics to specified decimal places
    for key in metrics:
        if isinstance(metrics[key], (int, float)):
            metrics[key] = round(metrics[key], decimal_places)
    
    # Log key metrics
    logger.info(f"{model_name} Evaluation Results:")
    logger.info(f"RMSE: {metrics['RMSE']}")
    logger.info(f"MAE: {metrics['MAE']}")
    logger.info(f"MAPE: {metrics['MAPE']}%")
    logger.info(f"R²: {metrics['R2']}")
    logger.info(f"Directional Accuracy: {metrics['Directional_Accuracy']}%")
    logger.info(f"Within 1%: {metrics['Within_1%']}%")
    logger.info(f"Within 5%: {metrics['Within_5%']}%")
    logger.info(f"Trading Return: {metrics['Total_Return_Pct']}%")
    logger.info(f"Buy-and-Hold Return: {metrics['Buy_and_Hold_Return_Pct']}%")
    
    return metrics

def compare_models(y_true, model_predictions, model_names, output_path=None):
    """
    Compare multiple models using various evaluation metrics.
    
    Args:
        y_true: Actual values
        model_predictions: List of predicted values from different models
        model_names: List of model names
        output_path: Path to save the comparison results
        
    Returns:
        DataFrame with comparison results
    """
    if len(model_predictions) != len(model_names):
        logger.error("Number of model predictions must match number of model names")
        return None
    
    # Create a list to store metrics for each model
    all_metrics = []
    
    # Evaluate each model
    for i, (y_pred, model_name) in enumerate(zip(model_predictions, model_names)):
        metrics = evaluate_model(y_true, y_pred, model_name)
        all_metrics.append(metrics)
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(all_metrics, index=model_names)
    
    # Save results if output path is provided
    if output_path:
        comparison_df.to_csv(output_path)
        logger.info(f"Comparison results saved to {output_path}")
    
    return comparison_df

def generate_classification_report(y_true, y_pred):
    """
    Generate a classification report for directional prediction.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with classification metrics
    """
    # Calculate actual and predicted price movements
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate confusion matrix elements
    tp = np.sum((true_direction == True) & (pred_direction == True))
    fp = np.sum((true_direction == False) & (pred_direction == True))
    tn = np.sum((true_direction == False) & (pred_direction == False))
    fn = np.sum((true_direction == True) & (pred_direction == False))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Up movement metrics
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_up = 2 * precision_up * recall_up / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0
    
    # Down movement metrics
    precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_down = 2 * precision_down * recall_down / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0
    
    # Weighted metrics
    total = tp + tn + fp + fn
    weighted_precision = (precision_up * (tp + fn) + precision_down * (tn + fp)) / total if total > 0 else 0
    weighted_recall = (recall_up * (tp + fn) + recall_down * (tn + fp)) / total if total > 0 else 0
    weighted_f1 = (f1_up * (tp + fn) + f1_down * (tn + fp)) / total if total > 0 else 0
    
    # Create report dictionary
    report = {
        'Accuracy': accuracy * 100,
        'Up_Movement': {
            'Precision': precision_up * 100,
            'Recall': recall_up * 100,
            'F1': f1_up * 100,
            'Support': tp + fn
        },
        'Down_Movement': {
            'Precision': precision_down * 100,
            'Recall': recall_down * 100,
            'F1': f1_down * 100,
            'Support': tn + fp
        },
        'Weighted_Avg': {
            'Precision': weighted_precision * 100,
            'Recall': weighted_recall * 100,
            'F1': weighted_f1 * 100,
            'Support': total
        },
        'Confusion_Matrix': {
            'True_Positive': tp,
            'False_Positive': fp,
            'True_Negative': tn,
            'False_Negative': fn
        }
    }
    
    return report

if __name__ == "__main__":
    # Example usage
    # This would typically be imported by other scripts
    
    # Example data
    y_true = np.array([100, 105, 103, 107, 109, 108, 110, 112, 115, 113])
    y_pred = np.array([102, 104, 105, 106, 108, 109, 111, 113, 114, 112])
    
    # Evaluate model
    metrics = evaluate_model(y_true, y_pred, model_name='Example Model')
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
