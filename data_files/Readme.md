# Data Directory

This directory contains all cryptocurrency data used for training and evaluation of price prediction models.

## Directory Structure

```
data_files/
├── raw/                # Raw cryptocurrency data
├── processed/          # Processed and normalized data
├── features/           # Feature-engineered data
└── split/              # Train/validation/test split data
```

## Data Pipeline

The data flows through the following pipeline:

1. **Raw Data Collection**: Historical cryptocurrency price data is collected from sources like Investing.com, Kaggle datasets, or cryptocurrency APIs (e.g., CryptoCompare, CoinGecko). Data is stored in CSV format in the `raw/` directory.

2. **Data Preprocessing**: Raw data is preprocessed using the `data_preprocessing.py` script. This includes handling missing values, normalizing numerical features using Min-Max scaling, and standardizing column names. Processed data is stored in the `processed/` directory.

3. **Feature Engineering**: Processed data is enhanced with additional technical indicators and features using the `feature_engineering.py` script. This includes moving averages, momentum indicators, volatility measures, and more. Feature-engineered data is stored in the `features/` directory.

4. **Data Splitting**: Feature-engineered data is split into training, validation, and test sets using the `data_splitting.py` script. This also prepares sequences for time series prediction. Split data is stored in the `split/` directory.

## Sample Data

The repository includes small sample datasets for Bitcoin and Ethereum to demonstrate the expected data format. For actual model training, you should download or generate larger datasets.

## Data Format

### Raw Data

Raw data should be in CSV format with the following columns:
- `Date`: Date of the price record
- `Price` (or `Close`): Closing price for the given date
- `Open`: Opening price
- `High`: Highest price during the period
- `Low`: Lowest price during the period
- `Volume`: Trading volume

### Processed Data

Processed data maintains the same column structure as raw data, but all numerical values are normalized to the range [0, 1] using Min-Max scaling.

### Feature-Engineered Data

Feature-engineered data includes all columns from the processed data plus additional technical indicators such as:
- Moving averages (SMA, EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Momentum indicators
- Volume-based indicators
- Cyclical time features

### Split Data

Split data is stored as NumPy arrays in cryptocurrency-specific subdirectories:
- `X_train.npy`, `y_train.npy`: Training data (features and targets)
- `X_val.npy`, `y_val.npy`: Validation data
- `X_test.npy`, `y_test.npy`: Test data
- `metadata.csv`: Information about the split (feature names, split ratios, etc.)

## Data Processing Scripts

To run the data processing pipeline:

1. **Preprocessing**:
```bash
python src/data/data_preprocessing.py --input-dir data_files/raw --output-dir data/processed
```

2. **Feature Engineering**:
```bash
python src/data/feature_engineering.py --input-dir data_files/processed --output-dir data/features
```

3. **Data Splitting**:
```bash
python src/data/data_splitting.py --input-dir data_files/features --output-dir data/split
```
