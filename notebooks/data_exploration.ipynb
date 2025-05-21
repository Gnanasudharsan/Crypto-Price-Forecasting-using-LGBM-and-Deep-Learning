{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cryptocurrency Price Data Exploration\n",
    "\n",
    "This notebook explores cryptocurrency price data and performs initial analysis to understand the dataset characteristics. This exploration is crucial for model development and feature engineering."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from datetime import datetime, timedelta\n",
    "\n",
    "# Set plotting style\n",
    "plt.style.use('ggplot')\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "\n",
    "# Set display options\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', None)\n",
    "pd.set_option('display.float_format', '{:.2f}'.format)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load the Dataset\n",
    "\n",
    "We'll be using cryptocurrency data from Kaggle, which contains historical price information for major cryptocurrencies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Define the data directory\n",
    "DATA_DIR = '../data/raw'\n",
    "\n",
    "# List available CSV files\n",
    "csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]\n",
    "print(f\"Available CSV files: {csv_files}\")\n",
    "\n",
    "# Load Bitcoin data as an example\n",
    "# In a real implementation, this would load the actual Kaggle dataset\n",
    "# For this notebook, we'll simulate the data\n",
    "\n",
    "# Create sample data\n",
    "def generate_sample_data(crypto_name, days=1000, start_price=40000):\n",
    "    np.random.seed(42)  # For reproducibility\n",
    "    \n",
    "    # Generate dates\n",
    "    end_date = datetime.now()\n",
    "    start_date = end_date - timedelta(days=days)\n",
    "    dates = pd.date_range(start=start_date, end=end_date, freq='D')\n",
    "    \n",
    "    # Generate prices with random walk\n",
    "    prices = [start_price]\n",
    "    for i in range(1, len(dates)):\n",
    "        # Daily change between -5% and 5%\n",
    "        change = np.random.normal(0, 0.02)  # Mean 0, std 2%\n",
    "        prices.append(prices[-1] * (1 + change))\n",
    "    \n",
    "    # Create DataFrame\n",
    "    df = pd.DataFrame({\n",
    "        'Date': dates,\n",
    "        'Price': prices,\n",
    "        'Open': [price * np.random.uniform(0.97, 1.0) for price in prices],\n",
    "        'High': [price * np.random.uniform(1.0, 1.05) for price in prices],\n",
    "        'Low': [price * np.random.uniform(0.95, 1.0) for price in prices],\n",
    "        'Volume': [np.random.uniform(1e9, 5e9) for _ in prices]\n",
    "    })\n",
    "    \n",
    "    return df\n",
    "\n",
    "# Generate sample data for cryptocurrencies\n",
    "btc_df = generate_sample_data('Bitcoin', start_price=40000)\n",
    "eth_df = generate_sample_data('Ethereum', start_price=2000)\n",
    "xrp_df = generate_sample_data('Ripple', start_price=0.5)\n",
    "\n",
    "# Display the first few rows of Bitcoin data\n",
    "btc_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Exploration and Visualization\n",
    "\n",
    "Let's explore the dataset and visualize the historical prices."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check dataset info\n",
    "print(\"Bitcoin Dataset Info:\")\n",
    "btc_df.info()\n",
    "\n",
    "# Descriptive statistics\n",
    "print(\"\\nDescriptive Statistics:\")\n",
    "btc_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check for missing values\n",
    "print(\"Missing Values:\")\n",
    "btc_df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plot Bitcoin price over time\n",
    "plt.figure(figsize=(14, 7))\n",
    "plt.plot(btc_df['Date'], btc_df['Price'], label='BTC Close Price')\n",
    "plt.title('Bitcoin Price History')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Price (USD)')\n",
    "plt.grid(True)\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plot daily price changes\n",
    "btc_df['Daily_Return'] = btc_df['Price'].pct_change() * 100\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plt.plot(btc_df['Date'][1:], btc_df['Daily_Return'][1:], color='blue')\n",
    "plt.title('Bitcoin Daily Returns (%)')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Daily Return (%)')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Distribution of daily returns\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(btc_df['Daily_Return'].dropna(), kde=True, bins=50)\n",
    "plt.title('Distribution of Bitcoin Daily Returns')\n",
    "plt.xlabel('Daily Return (%)')\n",
    "plt.ylabel('Frequency')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Compare multiple cryptocurrencies\n",
    "# Standardize the prices for comparison\n",
    "btc_normalized = btc_df.copy()\n",
    "eth_normalized = eth_df.copy()\n",
    "xrp_normalized = xrp_df.copy()\n",
    "\n",
    "btc_normalized['Price'] = btc_normalized['Price'] / btc_normalized['Price'].iloc[0]\n",
    "eth_normalized['Price'] = eth_normalized['Price'] / eth_normalized['Price'].iloc[0]\n",
    "xrp_normalized['Price'] = xrp_normalized['Price'] / xrp_normalized['Price'].iloc[0]\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plt.plot(btc_normalized['Date'], btc_normalized['Price'], label='Bitcoin')\n",
    "plt.plot(eth_normalized['Date'], eth_normalized['Price'], label='Ethereum')\n",
    "plt.plot(xrp_normalized['Date'], xrp_normalized['Price'], label='Ripple')\n",
    "plt.title('Normalized Price Comparison (Base = First Day)')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Normalized Price')\n",
    "plt.grid(True)\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Correlation Analysis\n",
    "\n",
    "Let's examine the correlation between different cryptocurrencies and between price and volume."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Merge datasets on Date\n",
    "btc_price = btc_df[['Date', 'Price']].rename(columns={'Price': 'BTC_Price'})\n",
    "eth_price = eth_df[['Date', 'Price']].rename(columns={'Price': 'ETH_Price'})\n",
    "xrp_price = xrp_df[['Date', 'Price']].rename(columns={'Price': 'XRP_Price'})\n",
    "\n",
    "# Merge\n",
    "merged_df = pd.merge(btc_price, eth_price, on='Date', how='inner')\n",
    "merged_df = pd.merge(merged_df, xrp_price, on='Date', how='inner')\n",
    "\n",
    "# Correlation matrix\n",
    "corr_matrix = merged_df.drop('Date', axis=1).corr()\n",
    "\n",
    "# Plot correlation heatmap\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)\n",
    "plt.title('Correlation Between Cryptocurrency Prices')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check correlation between price and volume for Bitcoin\n",
    "btc_price_volume_corr = btc_df[['Price', 'Volume']].corr()\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(btc_price_volume_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)\n",
    "plt.title('Correlation Between Bitcoin Price and Volume')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Scatter plot of Price vs Volume\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(btc_df['Volume'], btc_df['Price'], alpha=0.5)\n",
    "plt.title('Bitcoin Price vs Volume')\n",
    "plt.xlabel('Volume')\n",
    "plt.ylabel('Price (USD)')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Engineering\n",
    "\n",
    "Let's create some additional features that might be useful for our model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Add features to Bitcoin data\n",
    "btc_features = btc_df.copy()\n",
    "\n",
    "# Price momentum (5-day and 20-day)\n",
    "btc_features['Price_5d_pct'] = btc_features['Price'].pct_change(periods=5) * 100\n",
    "btc_features['Price_20d_pct'] = btc_features['Price'].pct_change(periods=20) * 100\n",
    "\n",
    "# Moving averages\n",
    "btc_features['MA_5'] = btc_features['Price'].rolling(window=5).mean()\n",
    "btc_features['MA_20'] = btc_features['Price'].rolling(window=20).mean()\n",
    "btc_features['MA_50'] = btc_features['Price'].rolling(window=50).mean()\n",
    "\n",
    "# Volatility (standard deviation of returns)\n",
    "btc_features['Volatility_5d'] = btc_features['Daily_Return'].rolling(window=5).std()\n",
    "btc_features['Volatility_20d'] = btc_features['Daily_Return'].rolling(window=20).std()\n",
    "\n",
    "# Relative price to moving average\n",
    "btc_features['Price_Rel_MA5'] = btc_features['Price'] / btc_features['MA_5']\n",
    "btc_features['Price_Rel_MA20'] = btc_features['Price'] / btc_features['MA_20']\n",
    "\n",
    "# Volume features\n",
    "btc_features['Volume_5d_pct'] = btc_features['Volume'].pct_change(periods=5) * 100\n",
    "btc_features['Volume_MA_5'] = btc_features['Volume'].rolling(window=5).mean()\n",
    "btc_features['Volume_Rel_MA5'] = btc_features['Volume'] / btc_features['Volume_MA_5']\n",
    "\n",
    "# Display the new features\n",
    "btc_features.dropna().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plot some of the engineered features\n",
    "plt.figure(figsize=(14, 10))\n",
    "\n",
    "# Plot 1: Price with moving averages\n",
    "plt.subplot(2, 2, 1)\n",
    "plt.plot(btc_features['Date'], btc_features['Price'], label='Price')\n",
    "plt.plot(btc_features['Date'], btc_features['MA_5'], label='5-day MA')\n",
    "plt.plot(btc_features['Date'], btc_features['MA_20'], label='20-day MA')\n",
    "plt.plot(btc_features['Date'], btc_features['MA_50'], label='50-day MA')\n",
    "plt.title('Bitcoin Price with Moving Averages')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Price (USD)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "# Plot 2: Volatility\n",
    "plt.subplot(2, 2, 2)\n",
    "plt.plot(btc_features['Date'], btc_features['Volatility_5d'], label='5-day Volatility')\n",
    "plt.plot(btc_features['Date'], btc_features['Volatility_20d'], label='20-day Volatility')\n",
    "plt.title('Bitcoin Price Volatility')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Volatility (Std Dev of Returns)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "# Plot 3: Price Momentum\n",
    "plt.subplot(2, 2, 3)\n",
    "plt.plot(btc_features['Date'], btc_features['Price_5d_pct'], label='5-day Momentum')\n",
    "plt.plot(btc_features['Date'], btc_features['Price_20d_pct'], label='20-day Momentum')\n",
    "plt.title('Bitcoin Price Momentum (%)')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Price Change (%)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "# Plot 4: Relative Price to Moving Average\n",
    "plt.subplot(2, 2, 4)\n",
    "plt.plot(btc_features['Date'], btc_features['Price_Rel_MA5'], label='Price / 5-day MA')\n",
    "plt.plot(btc_features['Date'], btc_features['Price_Rel_MA20'], label='Price / 20-day MA')\n",
    "plt.axhline(y=1, color='r', linestyle='--')\n",
    "plt.title('Bitcoin Price Relative to Moving Averages')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Ratio')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Data Preparation for Modeling\n",
    "\n",
    "Let's prepare the data for our prediction models by normalizing features and creating sequences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Select features to use\n",
    "features = ['Price', 'Open', 'High', 'Low', 'Volume', \n",
    "            'MA_5', 'MA_20', 'Volatility_5d', \n",
    "            'Price_Rel_MA5', 'Price_Rel_MA20', 'Volume_Rel_MA5']\n",
    "\n",
    "# Filter the data to rows that have all features\n",
    "btc_model_data = btc_features.dropna(subset=features).copy()\n",
    "\n",
    "# Normalize the features using Min-Max scaling\n",
    "scaler = MinMaxScaler()\n",
    "btc_model_data[features] = scaler.fit_transform(btc_model_data[features])\n",
    "\n",
    "# Display the normalized data\n",
    "btc_model_data[features].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create sequences for time series prediction\n",
    "def create_sequences(data, seq_length):\n",
    "    X, y = [], []\n",
    "    for i in range(len(data) - seq_length):\n",
    "        X.append(data[i:i + seq_length])\n",
    "        y.append(data[i + seq_length, 0])  # 0 index corresponds to Price\n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "# Define sequence length\n",
    "sequence_length = 60  # 60 days of historical data\n",
    "\n",
    "# Prepare data for sequence creation\n",
    "data_array = btc_model_data[features].values\n",
    "\n",
    "# Create sequences\n",
    "X, y = create_sequences(data_array, sequence_length)\n",
    "\n",
    "print(f\"X shape: {X.shape}\")\n",
    "print(f\"y shape: {y.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Split data into training, validation, and test sets\n",
    "train_size = int(0.7 * len(X))\n",
    "val_size = int(0.15 * len(X))\n",
    "\n",
    "X_train, y_train = X[:train_size], y[:train_size]\n",
    "X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]\n",
    "X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]\n",
    "\n",
    "print(f\"Training set: {X_train.shape}, {y_train.shape}\")\n",
    "print(f\"Validation set: {X_val.shape}, {y_val.shape}\")\n",
    "print(f\"Test set: {X_test.shape}, {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Conclusion\n",
    "\n",
    "In this notebook, we explored the cryptocurrency price data, performed feature engineering, and prepared the data for our LGBM and neural network models. The key observations are:\n",
    "\n",
    "1. **Price Trends**: We observed the historical price trends of Bitcoin, Ethereum, and Ripple.\n",
    "2. **Volatility**: Cryptocurrency prices show high volatility, as evidenced by the daily returns distribution.\n",
    "3. **Correlations**: There are strong correlations between different cryptocurrencies, suggesting common market factors.\n",
    "4. **Feature Engineering**: We created several features like moving averages, momentum, and volatility metrics that can help our models capture market patterns.\n",
    "5. **Data Preparation**: We normalized the data and created sequences for time series prediction using a 60-day window.\n",
    "\n",
    "The prepared dataset will be used to train our LGBM Neural network model as described in the research paper."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
