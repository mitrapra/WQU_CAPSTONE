try:
    import yfinance as yf # Try to import the yfinance package, used for fetching financial data.
except ImportError:
    # If yfinance is not installed, raise an ImportError with instructions.
    raise ImportError("Cannot start without 'yfinance' package.\n Install it before running the code again.")

import pandas as pd  # Import the pandas library for data manipulation.
import numpy as np   # Import the numpy library for numerical computations.
import math  		 # Import the math module for mathematical functions.
import os  			 # Import the os module for interacting with the operating system.

# Import module aliases for project-specific paths and utilities.
import champion.config._paths as paths
import champion.source.util._util as utils

class MarketDataEngineer:
    """A class for managing market data, including downloading, processing, and exporting."""

    def __init__(self):
        """Initialize the MarketDataEngineer object."""
        
        # Read the ETL (Extract, Transform, Load) configuration from JSON files using utility functions.
        etl_config 					= utils.read_json(paths.etl_config_path)

        # Initialize data loading, processing, and train-test split configurations.
        self.data_loading_config    = etl_config.get("data_loading")
        self.data_processing_config = etl_config.get("data_processing")
        self.train_test_config 		= etl_config.get("train_test_split")

        # Extract parameters related to data loading.
        self.vix_ticker 		= self.data_loading_config.get("VIX_Ticker_Name", "^VIX")
        self.eq_ticker 			= self.data_loading_config.get("EQ_Ticker_Name", "V00")
        self.all_tickers 		= " ".join([self.vix_ticker, self.eq_ticker])
        self.start_date 		= self.data_loading_config.get("start_date")
        self.end_date 			= self.data_loading_config.get("end_date")
        self.eq_ticker_stress 	= self.data_loading_config.get("EQ_Ticker_Name_Stress", "SPY")
        self.all_tickers_stress = " ".join([self.vix_ticker, self.eq_ticker_stress])
        self.stress_start_date  = self.data_loading_config.get("stress_start_date")
        self.stress_end_date 	= self.data_loading_config.get("stress_end_date")
        self.interval 			= self.data_loading_config.get("download_interval")
        self.col_name 			= self.data_loading_config.get("col_name")

        # Extract parameters related to data processing.
        self.window 			= self.data_processing_config.get("window_period")
        self.benchmark 			= self.data_processing_config.get("target_benchmark")
        self.lag 				= self.data_processing_config.get("lags", 12)

        # Extract parameters related to train-test split.
        self.train_size 		= self.train_test_config.get("train_size")

    def market_data_download(self) -> pd.DataFrame:
        """Download market data using Yahoo Finance API."""
        return yf.download(tickers=self.all_tickers,
                           start=self.start_date,
                           end=self.end_date,
                           interval=self.interval)[self.col_name].dropna()

    def market_data_download_stress_period(self) -> pd.DataFrame:
        """Download market data for a stress test period."""
        return yf.download(tickers=self.all_tickers_stress,
                           start=self.stress_start_date,
                           end=self.stress_end_date,
                           interval=self.interval)[self.col_name].dropna()

    def market_data_generate_returns(self, market_data: pd.DataFrame) -> pd:
        """Generate log returns from market data."""
        return np.log(market_data).diff() * 100

    def generate_features(self, market_data: pd.DataFrame, backtest : str = "recent") -> pd:
        """Generate features including Relative Strength Index (RSI) and lagged variables."""
        
        # Determine which equity ticker to use based on the backtest period.
        if backtest == "recent":
            eq_tkr = self.eq_ticker
        else:
            eq_tkr = self.eq_ticker_stress

        # Copy the market data to avoid modification of the original DataFrame.
        vix_rsi    = market_data.copy(deep=True)
        
        # Calculate Relative Strength Index (RSI).
        vix_rsi["Gain"] 	= vix_rsi[self.vix_ticker].apply(lambda x: x if x > 0 else 0)
        vix_rsi["Loss"] 	= vix_rsi[self.vix_ticker].apply(lambda x: -x if x < 0 else 0)
        vix_rsi["Avg_Gain"] = vix_rsi["Gain"].rolling(self.window).apply(lambda x: x[x != 0].mean())
        vix_rsi["Avg_Loss"] = vix_rsi["Loss"].rolling(self.window).apply(lambda x: x[x != 0].mean())
        vix_rsi["RSI"] 		= vix_rsi["Avg_Gain"] / vix_rsi["Avg_Loss"]
        vix_rsi["RSI"] 		= vix_rsi["RSI"].shift(1)

        # Generate lagged variables for feature engineering.
        for i in range(1, self.lag + 1):
            vix_rsi[f"lag_{i}"] = vix_rsi[self.vix_ticker].shift(i)

        # Generate binary target variable based on the benchmark.
        vix_rsi["target"] = vix_rsi[eq_tkr].apply(lambda x: int(x > self.benchmark))

        # Remove intermediary columns and rows with NaN values.
        processed_market_data = vix_rsi.drop(columns=["Gain", "Loss", "Avg_Gain", "Avg_Loss"]).dropna()

        return processed_market_data

    def export_market_data(self, market_data: pd.DataFrame):
        """Export processed market data to a CSV file."""
        file_name = f"{self.eq_ticker}_{self.start_date}_{self.end_date}.csv"
        market_data.to_csv(os.path.join(paths.market_data_output_folder, file_name), index=False)

    def train_test_split(self, market_data: pd.DataFrame):
        """Split market data into training and testing sets."""
        mkt_env_size 	= market_data.shape[0]
        train_index 	= math.floor(mkt_env_size * self.train_size)
        train_df 		= market_data.iloc[:train_index]
        test_df 		= market_data.iloc[train_index:]
        return train_df, test_df

    def etl_process(self):
        """Execute the Extract, Transform, and Load (ETL) process."""
        # Download and process market data for the primary period.
        market_data 		  = self.market_data_download()
        log_returns 		  = self.market_data_generate_returns(market_data)
        processed_market_data = self.generate_features(log_returns)
        
        # Export processed market data and split it into training and testing sets.
        self.export_market_data(processed_market_data)
        train_df, test_df 	  = self.train_test_split(processed_market_data)
        train_df.to_csv(os.path.join(paths.market_data_output_folder, "train_set.csv"))
        test_df.to_csv(os.path.join(paths.market_data_output_folder, "test_set.csv"))

        # Download and process market data for the stress test period.
        market_data_stress 	  		 = self.market_data_download_stress_period()
        log_returns_stress 	  		 = self.market_data_generate_returns(market_data_stress)
        processed_market_data_stress = self.generate_features(log_returns_stress, backtest="stress")
        processed_market_data_stress.to_csv(os.path.join(paths.market_data_output_folder, "stress_period.csv"))