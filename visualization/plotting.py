#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for cryptocurrency price prediction models.
Provides functions for creating various plots to analyze model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import logging
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("plotting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("deep")

def set_plotting_style(style='seaborn-v0_8-whitegrid', context='talk', palette='deep'):
    """
    Set the plotting style for consistent visualizations.
    
    Args:
        style: Matplotlib style to use
        context: Seaborn context (paper, notebook, talk, poster)
        palette: Color palette to use
    """
    plt.style.use(style)
    sns.set_context(context)
    sns.set_palette(palette)
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20
    
    # Set figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Set line width
    plt.rcParams['lines.linewidth'] = 2
    
    # Set marker size
    plt.rcParams['lines.markersize'] = 8
    
    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette(palette))
    
    logger.info(f"Plot style set to {style}, context set to {context}")

def plot_price_prediction(dates, y_true, y_pred, title=None, output_path=None, zoom_last_n=None):
    """
    Plot actual vs predicted cryptocurrency prices.
    
    Args:
        dates: Array of dates
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
        output_path: Path to save the plot
        zoom_last_n: Number of most recent points to focus on
    """
    # Make sure dates is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # If zoom_last_n is provided, focus on the most recent points
    if zoom_last_n and zoom_last_n < len(dates):
        dates = dates[-zoom_last_n:]
        y_true = y_true[-zoom_last_n:]
        y_pred = y_pred[-zoom_last_n:]
    
    # Plot actual prices
    plt.plot(dates, y_true, label='Actual', marker='o', linestyle='-', alpha=0.7)
    
    # Plot predicted prices
    plt.plot(dates, y_pred, label='Predicted', marker='s', linestyle='-', alpha=0.7)
    
    # Format x-axis date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title('Cryptocurrency Price: Actual vs Predicted')
    
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Price prediction plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_prediction_error(dates, y_true, y_pred, title=None, output_path=None, error_type='absolute'):
    """
    Plot prediction error over time.
    
    Args:
        dates: Array of dates
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
        output_path: Path to save the plot
        error_type: Type of error to plot ('absolute', 'percentage')
    """
    # Make sure dates is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Calculate error
    if error_type == 'absolute':
        error = y_true - y_pred
        ylabel = 'Absolute Error (USD)'
    elif error_type == 'percentage':
        error = (y_true - y_pred) / y_true * 100
        ylabel = 'Percentage Error (%)'
    else:
        logger.error(f"Invalid error_type: {error_type}. Using 'absolute' instead.")
        error = y_true - y_pred
        ylabel = 'Absolute Error (USD)'
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot error
    plt.bar(dates, error, alpha=0.7, width=0.8)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Format x-axis date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Cryptocurrency Price Prediction Error ({error_type.capitalize()})')
    
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add statistics in a text box
    mean_error = np.mean(error)
    std_error = np.std(error)
    median_error = np.median(error)
    max_error = np.max(error)
    min_error = np.min(error)
    
    stats_text = (
        f"Mean Error: {mean_error:.2f}\n"
        f"Median Error: {median_error:.2f}\n"
        f"Std Dev: {std_error:.2f}\n"
        f"Max Error: {max_error:.2f}\n"
        f"Min Error: {min_error:.2f}"
    )
    
    plt.annotate(
        stats_text,
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=10
    )
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction error plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_error_distribution(y_true, y_pred, title=None, output_path=None, error_type='absolute'):
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
        output_path: Path to save the plot
        error_type: Type of error to plot ('absolute', 'percentage')
    """
    # Calculate error
    if error_type == 'absolute':
        error = y_true - y_pred
        xlabel = 'Absolute Error (USD)'
    elif error_type == 'percentage':
        error = (y_true - y_pred) / y_true * 100
        xlabel = 'Percentage Error (%)'
    else:
        logger.error(f"Invalid error_type: {error_type}. Using 'absolute' instead.")
        error = y_true - y_pred
        xlabel = 'Absolute Error (USD)'
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot error distribution
    sns.histplot(error, kde=True, bins=30)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Add mean and median lines
    mean_error = np.mean(error)
    median_error = np.median(error)
    
    plt.axvline(x=mean_error, color='g', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.2f}')
    plt.axvline(x=median_error, color='b', linestyle='--', alpha=0.7, label=f'Median: {median_error:.2f}')
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of Prediction Errors ({error_type.capitalize()})')
    
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error distribution plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_directional_accuracy(y_true, y_pred, title=None, output_path=None):
    """
    Plot confusion matrix for directional prediction accuracy.
    
    Args:
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
        output_path: Path to save the plot
    """
    # Calculate actual and predicted price movements
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Convert boolean to integers (0 for down, 1 for up)
    true_direction = true_direction.astype(int)
    pred_direction = pred_direction.astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_direction, pred_direction)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(cmap='Blues', values_format='d')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Directional Accuracy Confusion Matrix')
    
    plt.tight_layout()
    
    # Calculate directional accuracy
    accuracy = np.mean(true_direction == pred_direction) * 100
    
    # Add accuracy as text
    plt.figtext(
        0.5, 0.01,
        f"Overall Directional Accuracy: {accuracy:.2f}%",
        ha="center",
        fontsize=12,
        bbox={"facecolor":"white", "alpha":0.8, "pad":5}
    )
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Directional accuracy plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_model_comparison(dates, y_true, model_predictions, model_names, title=None, output_path=None, zoom_last_n=None):
    """
    Plot comparison of predictions from multiple models.
    
    Args:
        dates: Array of dates
        y_true: Actual prices
        model_predictions: List of predicted prices from different models
        model_names: List of model names
        title: Plot title
        output_path: Path to save the plot
        zoom_last_n: Number of most recent points to focus on
    """
    # Make sure dates is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Check if model_predictions and model_names have the same length
    if len(model_predictions) != len(model_names):
        logger.error("Length of model_predictions and model_names must be the same")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # If zoom_last_n is provided, focus on the most recent points
    if zoom_last_n and zoom_last_n < len(dates):
        plot_dates = dates[-zoom_last_n:]
        plot_y_true = y_true[-zoom_last_n:]
        plot_model_predictions = [pred[-zoom_last_n:] for pred in model_predictions]
    else:
        plot_dates = dates
        plot_y_true = y_true
        plot_model_predictions = model_predictions
    
    # Plot actual prices
    plt.plot(plot_dates, plot_y_true, label='Actual', marker='o', linestyle='-', linewidth=2, alpha=0.7)
    
    # Plot predicted prices for each model
    for i, (pred, name) in enumerate(zip(plot_model_predictions, model_names)):
        plt.plot(plot_dates, pred, label=name, marker='s', linestyle='--', alpha=0.7)
    
    # Format x-axis date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title('Model Comparison: Cryptocurrency Price Predictions')
    
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_trading_strategy(dates, y_true, y_pred, title=None, output_path=None, initial_capital=10000, transaction_fee_pct=0.001):
    """
    Plot the performance of a trading strategy based on model predictions.
    
    Args:
        dates: Array of dates
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
        output_path: Path to save the plot
        initial_capital: Initial investment amount
        transaction_fee_pct: Transaction fee as a percentage of trade amount
    """
    # Make sure dates is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Calculate price changes
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    
    # Skip the first date (since we need at least one change)
    strategy_dates = dates[1:]
    
    # Initialize variables for backtest
    capital = initial_capital
    position = 0  # 0: no position, 1: long position
    trades = 0
    profitable_trades = 0
    
    # Track capital over time
    capitals = [initial_capital]
    positions = [0]  # 0: no position, 1: long position
    
    # Buy-and-hold capital
    buy_and_hold_capital = [initial_capital]
    buy_and_hold_pos = 0
    
    # Simple trading strategy: Buy when predicted change is positive, sell when negative
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
        positions.append(position)
        
        # Update buy-and-hold strategy
        if buy_and_hold_pos == 0:
            buy_and_hold_pos = 1
            # Apply transaction fee once
            buy_and_hold_capital.append(buy_and_hold_capital[-1] * (1 - transaction_fee_pct))
        else:
            # Update capital based on price change
            buy_and_hold_capital.append(buy_and_hold_capital[-1] * (1 + true_change / y_true[i]))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot capital over time
    ax1.plot(strategy_dates, capitals[1:], label='Strategy Capital', color='blue', linewidth=2)
    ax1.plot(strategy_dates, buy_and_hold_capital[1:], label='Buy-and-Hold Capital', color='green', linestyle='--', linewidth=2)
    
    # Format x-axis date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Format y-axis with dollar signs
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Set title and labels for first subplot
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Trading Strategy Performance')
    
    ax1.set_xlabel('')  # No x-label for first subplot
    ax1.set_ylabel('Capital (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot actual cryptocurrency price
    ax2.plot(strategy_dates, y_true[1:], label='Cryptocurrency Price', color='orange', linewidth=2)
    
    # Overlay buy/sell points
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(positions)):
        if positions[i] == 1 and positions[i-1] == 0:  # Buy signal
            buy_signals.append(i-1)
        elif positions[i] == 0 and positions[i-1] == 1:  # Sell signal
            sell_signals.append(i-1)
    
    # Plot buy signals
    if buy_signals:
        ax2.scatter(
            [strategy_dates[i] for i in buy_signals],
            [y_true[i+1] for i in buy_signals],
            marker='^',
            color='green',
            s=100,
            label='Buy Signal'
        )
    
    # Plot sell signals
    if sell_signals:
        ax2.scatter(
            [strategy_dates[i] for i in sell_signals],
            [y_true[i+1] for i in sell_signals],
            marker='v',
            color='red',
            s=100,
            label='Sell Signal'
        )
    
    # Format x-axis date ticks for second subplot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Format y-axis with dollar signs for second subplot
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Set labels for second subplot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format date ticks
    fig.autofmt_xdate()
    
    # Add statistics in a text box
    final_capital = capitals[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    buy_and_hold_return = (buy_and_hold_capital[-1] / initial_capital - 1) * 100
    win_rate = profitable_trades / max(1, trades) * 100
    
    stats_text = (
        f"Initial Capital: ${initial_capital:,.2f}\n"
        f"Final Capital: ${final_capital:,.2f}\n"
        f"Total Return: {total_return:.2f}%\n"
        f"Buy-and-Hold Return: {buy_and_hold_return:.2f}%\n"
        f"Outperformance: {total_return - buy_and_hold_return:.2f}%\n"
        f"Total Trades: {trades}\n"
        f"Win Rate: {win_rate:.2f}%"
    )
    
    ax1.annotate(
        stats_text,
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trading strategy plot saved to {output_path}")
    
    # Show plot
    plt.show()
    
    # Return trading metrics
    return {
        'Initial_Capital': initial_capital,
        'Final_Capital': final_capital,
        'Total_Return_Pct': total_return,
        'Buy_and_Hold_Return_Pct': buy_and_hold_return,
        'Outperformance_Pct': total_return - buy_and_hold_return,
        'Total_Trades': trades,
        'Win_Rate_Pct': win_rate
    }

def plot_feature_importance(feature_names, importances, title=None, output_path=None, top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importances
        title: Plot title
        output_path: Path to save the plot
        top_n: Number of top features to display
    """
    # Create DataFrame for sorting
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_df = feature_df.sort_values('Importance', ascending=False)
    
    # Limit to top_n features
    if top_n and len(feature_df) > top_n:
        feature_df = feature_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot horizontal bar chart
    ax = sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    
    # Add values to bars
    for i, v in enumerate(feature_df['Importance']):
        ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Top {len(feature_df)} Feature Importances')
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_metric_comparison(metrics_df, title=None, output_path=None):
    """
    Create bar charts comparing metrics across models.
    
    Args:
        metrics_df: DataFrame with metrics (rows are models, columns are metrics)
        title: Plot title
        output_path: Path to save the plot
    """
    # Define key metrics to plot
    key_metrics = [
        'RMSE', 'MAE', 'MAPE', 'R2', 
        'Directional_Accuracy', 'Within_1%', 'Within_5%', 
        'Total_Return_Pct', 'Buy_and_Hold_Return_Pct'
    ]
    
    # Filter available metrics
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if not available_metrics:
        logger.error("No key metrics found in the DataFrame")
        return
    
    # Calculate the number of rows and columns for the subplots
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Flatten axes array for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Create bar chart for the metric
            metrics_df[metric].plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(metrics_df)))
            
            # Add values on top of bars
            for j, v in enumerate(metrics_df[metric]):
                ax.text(j, v + 0.01 * max(metrics_df[metric]), f"{v:.2f}", ha='center')
            
            # Set title and labels
            ax.set_title(metric.replace('_', ' '))
            ax.set_xlabel('')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Model Performance Comparison', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for main title
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metric comparison plot saved to {output_path}")
    
    # Show plot
    plt.show()

def plot_correlation_matrix(df, title=None, output_path=None):
    """
    Plot correlation matrix for cryptocurrency data features.
    
    Args:
        df: DataFrame with cryptocurrency data
        title: Plot title
        output_path: Path to save the plot
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=False,
        fmt='.2f',
        cbar_kws={'shrink': 0.8}
    )
    
    # Set title
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Feature Correlation Matrix', fontsize=16)
    
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix plot saved to {output_path}")
    
    # Show plot
    plt.show()

def create_visualization_report(dates, y_true, y_pred, model_name, output_dir='visualizations'):
    """
    Create a comprehensive visualization report for a model.
    
    Args:
        dates: Array of dates
        y_true: Actual prices
        y_pred: Predicted prices
        model_name: Name of the model
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a sanitized model name for file names
    safe_model_name = model_name.lower().replace(' ', '_')
    
    # Generate plots
    
    # 1. Price prediction plot
    plot_price_prediction(
        dates,
        y_true,
        y_pred,
        title=f"{model_name}: Actual vs Predicted Prices",
        output_path=os.path.join(output_dir, f"{safe_model_name}_price_prediction.png")
    )
    
    # 2. Price prediction plot (last 90 days)
    plot_price_prediction(
        dates,
        y_true,
        y_pred,
        title=f"{model_name}: Actual vs Predicted Prices (Last 90 Days)",
        output_path=os.path.join(output_dir, f"{safe_model_name}_price_prediction_recent.png"),
        zoom_last_n=90
    )
    
    # 3. Absolute error plot
    plot_prediction_error(
        dates,
        y_true,
        y_pred,
        title=f"{model_name}: Absolute Prediction Error",
        output_path=os.path.join(output_dir, f"{safe_model_name}_absolute_error.png"),
        error_type='absolute'
    )
    
    # 4. Percentage error plot
    plot_prediction_error(
        dates,
        y_true,
        y_pred,
        title=f"{model_name}: Percentage Prediction Error",
        output_path=os.path.join(output_dir, f"{safe_model_name}_percentage_error.png"),
        error_type='percentage'
    )
    
    # 5. Error distribution plot
    plot_error_distribution(
        y_true,
        y_pred,
        title=f"{model_name}: Distribution of Absolute Errors",
        output_path=os.path.join(output_dir, f"{safe_model_name}_error_distribution.png"),
        error_type='absolute'
    )
    
    # 6. Percentage error distribution plot
    plot_error_distribution(
        y_true,
        y_pred,
        title=f"{model_name}: Distribution of Percentage Errors",
        output_path=os.path.join(output_dir, f"{safe_model_name}_percentage_error_distribution.png"),
        error_type='percentage'
    )
    
    # 7. Directional accuracy plot
    plot_directional_accuracy(
        y_true,
        y_pred,
        title=f"{model_name}: Directional Accuracy",
        output_path=os.path.join(output_dir, f"{safe_model_name}_directional_accuracy.png")
    )
    
    # 8. Trading strategy plot
    plot_trading_strategy(
        dates,
        y_true,
        y_pred,
        title=f"{model_name}: Trading Strategy Performance",
        output_path=os.path.join(output_dir, f"{safe_model_name}_trading_strategy.png")
    )
    
    logger.info(f"Visualization report for {model_name} created in {output_dir}")

if __name__ == "__main__":
    # Example usage
    # This would typically be imported by other scripts
    
    # Set style
    set_plotting_style()
    
    # Example data
    dates = pd.date_range(start='2023-01-01', periods=100)
    y_true = np.linspace(100, 150, 100) + np.random.normal(0, 5, 100)
    y_pred = y_true + np.random.normal(0, 3, 100)
    
    # Example plot
    plot_price_prediction(dates, y_true, y_pred, title="Example Price Prediction")
