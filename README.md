# Cryptocurrency Price Forecasting with LGBM Neural Network

This repository contains the implementation of a hybrid deep learning model that uses LGBM (Light Gradient Boosted Machine) for cryptocurrency price prediction as described in the research paper "Blockchain and Deep Learning-Based Models for Crypto Price Forecasting".

## Overview

The cryptocurrency market is known for its volatility and unpredictability. This project implements a Python-based LGBM Neural network model that can forecast market changes in real-time and help reduce investor losses. The model achieves approximately 97% accuracy by combining convolutional layers for short-term pattern recognition and LGBM layers for effective long-term relationship modeling.

![image](https://github.com/user-attachments/assets/0edf8918-42ec-42a3-b943-9c341c98c218)


## Features

- **Data Collection**: Automatically collect cryptocurrency price data from various sources
- **Data Preprocessing**: Normalize and prepare data for model training
- **Model Training**: Train the hybrid LGBM-CNN model on cryptocurrency data
- **Price Prediction**: Make real-time price predictions for major cryptocurrencies
- **Visualization**: Interactive visualizations of historical prices and predictions
- **Web Interface**: Flask-based dashboard for easy access to predictions

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/crypto-price-forecasting.git
cd crypto-price-forecasting
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection and Preprocessing

To collect and preprocess cryptocurrency data:

```bash
python src/data/data_collection.py --cryptocurrencies bitcoin,ethereum,ripple,monero,tether,iota
python src/data/data_preprocessing.py
```

### Model Training

To train the LGBM Neural network model:

```bash
python notebooks/model_training.py
```

### Price Prediction

To make predictions with the trained model:

```bash
python notebooks/price_prediction.py --cryptocurrency bitcoin
```

### Running the Web Application

To start the Flask web application:

```bash
cd app
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Model Architecture

The model used in this project is a hybrid architecture that combines:

1. **Convolutional Layers**: To capture short-term patterns in cryptocurrency price data
2. **LGBM Layers**: For efficient handling of long-term relationships and processing large volumes of data
3. **Integration Layer**: Combines information from both models to produce the final forecast

## Performance Evaluation

The model achieves approximately 97% accuracy on test data. The evaluation metrics include:

- Accuracy
- Precision
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Supported Cryptocurrencies

This implementation supports major cryptocurrencies including:

- Bitcoin (BTC)
- Ethereum (ETH)
- Ripple (XRP)
- Litecoin (LTC)
- Monero (XMR)
- Tether (USDT)
- IOTA (MIOTA)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The LGBM implementation is based on [Microsoft's LightGBM](https://github.com/microsoft/LightGBM)
- Cryptocurrency data is collected from [Investing.com](https://www.investing.com)
- This project implements the model described in the research paper "Blockchain and Deep Learning-Based Models for Crypto Price Forecasting" by Sai Monika S, et al.
