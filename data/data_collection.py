#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for collecting cryptocurrency data from Investing.com or Kaggle datasets.
This script supports downloading historical data for various cryptocurrencies.
"""

import os
import pandas as pd
import requests
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
import time
import logging
from tqdm import tqdm
import kaggle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cryptocurrency ID mappings for Investing.com
CRYPTO_IDS = {
    'bitcoin': '1057391',
    'ethereum': '1061443',
    'ripple': '1057392',
    'litecoin': '1061445',
    'monero': '1061453',
    'tether': '1061453',
    'iota': '1118146'
}

def download_from_investing(crypto_name, start_date, end_date, output_path):
    """
    Download cryptocurrency data from Investing.com using web scraping.
    
    Args:
        crypto_name: Name of the cryptocurrency (lowercase)
        start_date: Start date for historical data in YYYY-MM-DD format
        end_date: End date for historical data in YYYY-MM-DD format
        output_path: Path to save the downloaded data
    
    Returns:
        DataFrame with the downloaded data
    """
    try:
        if crypto_name.lower() not in CRYPTO_IDS:
            logger.error(f"Cryptocurrency {crypto_name} not found in the supported list.")
            return None
        
        crypto_id = CRYPTO_IDS[crypto_name.lower()]
        
        # Convert dates to required format
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Investing.com format
        st_date = f"{start_date_obj.month}/{start_date_obj.day}/{start_date_obj.year}"
        end_date = f"{end_date_obj.month}/{end_date_obj.day}/{end_date_obj.year}"
        
        # Note: The actual implementation would require more sophisticated handling
        # of web scraping with headers, sessions, etc. This is a simplified version.
        logger.info(f"Downloading {crypto_name} data from Investing.com")
        logger.info("In a real implementation, this would use proper web scraping techniques.")
        logger.info("For this demo, we'll simulate the download by loading sample data.")
        
        # Simulate downloaded data (in a real implementation, this would be web-scraped data)
        df = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date),
            'Price': [0] * (end_date_obj - start_date_obj).days,
            'Open': [0] * (end_date_obj - start_date_obj).days,
            'High': [0] * (end_date_obj - start_date_obj).days,
            'Low': [0] * (end_date_obj - start_date_obj).days,
            'Vol.': [0] * (end_date_obj - start_date_obj).days,
            'Change %': [0] * (end_date_obj - start_date_obj).days
        })
        
        # Save the data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return None

def download_from_kaggle(dataset_name, output_path):
    """
    Download cryptocurrency data from Kaggle datasets.
    
    Args:
        dataset_name: Name of the Kaggle dataset
        output_path: Path to save the downloaded data
    
    Returns:
        List of file paths of downloaded files
    """
    try:
        logger.info(f"Downloading {dataset_name} from Kaggle")
        
        # Note: This requires setting up Kaggle API credentials
        # See: https://github.com/Kaggle/kaggle-api
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name,
            path=os.path.dirname(output_path),
            unzip=True
        )
        
        logger.info(f"Kaggle dataset downloaded to {os.path.dirname(output_path)}")
        return [f for f in os.listdir(os.path.dirname(output_path)) if f.endswith('.csv')]
    
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Download cryptocurrency historical data')
    parser.add_argument('--source', choices=['investing', 'kaggle'], default='kaggle',
                        help='Source to download data from (investing or kaggle)')
    parser.add_argument('--cryptocurrencies', type=str, default='bitcoin,ethereum',
                        help='Comma-separated list of cryptocurrencies to download')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-01-01',
                        help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Directory to save downloaded data')
    parser.add_argument('--kaggle-dataset', type=str, 
                        default='mczielinski/bitcoin-historical-data',
                        help='Kaggle dataset to download (if source is kaggle)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cryptocurrencies = args.cryptocurrencies.split(',')
    
    if args.source == 'investing':
        for crypto in tqdm(cryptocurrencies, desc="Downloading cryptocurrencies"):
            output_path = os.path.join(args.output_dir, f"{crypto}_data.csv")
            download_from_investing(
                crypto,
                args.start_date,
                args.end_date,
                output_path
            )
    else:  # Kaggle
        download_from_kaggle(
            args.kaggle_dataset,
            args.output_dir
        )
        logger.info("Kaggle data downloaded. You may need to rename or organize the files.")

if __name__ == "__main__":
    main()
