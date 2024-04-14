# This block of code is attempting to import the yfinance library and assigning it the alias yf. However, it includes a try-except block to handle the scenario where the yfinance library is not installed on the system.
# try:: This keyword signifies the start of the try block where the code inside it is attempted to be executed.
# import yfinance as yf: This line tries to import the yfinance library and assigns it the alias yf.
# except ImportError:: This keyword signifies the start of the except block, which catches the ImportError exception, indicating that the yfinance library is not found.
# raise ImportError("Cannot start without 'yfinance' package.\n Install it before running the code again."): In case of an ImportError, this line raises a new ImportError with a descriptive error message prompting the user to install the yfinance package before running the code again.

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Cannot start without 'yfinance' package.\n Install it before running the code again.")

import pandas as pd  					# Import pandas library with alias pd
import numpy as np  					# Import numpy library with alias np
import bs4 as bs  						# Import BeautifulSoup from bs4 library with alias bs
import requests  						# Import requests library
import datetime  						# Import datetime module
import os  								# Import os module

from typing import Tuple  				# Import Tuple type from the typing module
from pandas.core.frame import DataFrame # Import DataFrame class from pandas
from datetime import datetime		    # Import datetime type from the datetime module

class StockDataEngineer:
    """
    A class to handle stock data processing and management.

    Attributes:
        requests_module (module): 	 The requests module to handle HTTP requests. Default is requests.
        list_stocks (list): 		 List of stock tickers.
        user_spec (bool): 			 Flag indicating whether the user specifies the list of stocks. Default is True.
        start_date (str): 			 Start date for downloading stock data.
        end_date (str): 			 End date for downloading stock data.
        cleaned_prices (DataFrame):  DataFrame containing cleaned stock prices.
        cleaned_returns (DataFrame): DataFrame containing cleaned stock returns.
        raw_prices (DataFrame): 	 DataFrame containing raw stock prices.
        raw_returns (DataFrame): 	 DataFrame containing raw stock returns.
    """

    def __init__(self, requests_module=requests, list_stocks=[], user_spec=True):
        """
        Initializes the StockDataEngineer class.

        Args:
            requests_module (module, optional): The requests module to handle HTTP requests. Default is requests.
            list_stocks (list, optional): 		List of stock tickers. Default is an empty list.
            user_spec (bool, optional): 		Flag indicating whether the user specifies the list of stocks. Default is True.
        """
        # Create 'Data' directory if it doesn't exist
        if not os.path.exists('Data'):
            os.makedirs('Data')

        # Initialize class attributes
        self.requests_module = requests_module
        self.list_stocks = list_stocks
        self.user_spec = user_spec
        self.start_date = None
        self.end_date = None
        self.cleaned_prices = None
        self.cleaned_returns = None
        self.raw_prices = None
        self.raw_returns = None

        # If user doesn't specify, fetch S&P 500 tickers from Wikipedia
        if not self.user_spec:
            self._get_sp500_tickers()

    def _check_date_range(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int]) -> Tuple[datetime, datetime]:
        """
        Check if the start date is before the end date.

        Args:
            start_date (tuple): Start date in the format (year, month, day).
            end_date (tuple): 	End date in the format (year, month, day).

        Returns:
            tuple: Tuple containing start date and end date as datetime objects.
        """
        # Convert start_date and end_date to datetime objects
        start = datetime(*start_date)
        end   = datetime(*end_date)

        # Check if start_date is before end_date
        if end <= start:
            raise Exception("The start date must be before the end date!")

        return start, end

    def _get_sp500_tickers(self):
        """
        Retrieve S&P 500 tickers from Wikipedia.

        This method is used if the user does not specify the list of stocks.
        """
        # Fetch S&P 500 tickers from Wikipedia page
        response = self.requests_module.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup 	 = bs.BeautifulSoup(response.text, 'lxml')
        table    = soup.find('table', {'class': 'wikitable sortable'})
        self.list_stocks = []

        # Extract tickers from the Wikipedia table
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            self.list_stocks.append(ticker)
            self.list_stocks = [s.replace('\n', '') for s in self.list_stocks]
            self.list_stocks = self.list_stocks + ["SPY"]

    def download_prices(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int], interval='1d', column='Adj Close') -> None:
        """
        Download raw stock prices.

        Args:
            start_date (tuple): 	  Start date in the format (year, month, day).
            end_date (tuple): 		  End date in the format (year, month, day).
            interval (str, optional): Interval for data (e.g., '1d' for daily). Default is '1d'.
            column (str, optional):   Column to retrieve (e.g., 'Adj Close'). Default is 'Adj Close'.
        """
        # Check date range and assign start_date, end_date
        self.start_date, self.end_date  = self._check_date_range(start_date, end_date)
        # Download raw prices using Yahoo Finance API
        self.raw_prices 				= yf.download(self.list_stocks, start=self.start_date, end=self.end_date, interval=interval)[column]

    def _write_on_disk(self, data: DataFrame, filename: str) -> None:
        """
        Write DataFrame to a file.

        Args:
            data (DataFrame): DataFrame to write.
            filename (str):   Name of the file.
        """
        # Write DataFrame to either CSV or HDF5 file based on filename extension
        if "csv" in filename:
            data.to_csv('Data/' + filename)
        elif "h5" in filename:
            data.to_hdf('Data/' + filename, 'fixed', mode='w', complib='blosc', complevel=9)

        # Print message confirming file save
        print(f"Saved: Data/{filename}")

    def get_ticker_list(self) -> list:
        """
        Get the list of stock tickers.

        Returns:
            list: List of stock tickers.
        """
        return self.list_stocks.copy()

    def get_raw_prices(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int], interval='1d', column='Adj Close', save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get raw stock prices.

        Args:
            start_date (tuple): 		  Start date in the format (year, month, day).
            end_date (tuple): 			  End date in the format (year, month, day).
            interval (str, optional): 	  Interval for data (e.g., '1d' for daily). Default is '1d'.
            column (str, optional): 	  Column to retrieve (e.g., 'Adj Close'). Default is 'Adj Close'.
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing raw stock prices.
        """
        # Download raw prices
        self.download_prices(start_date, end_date)

        # Save raw prices to file if requested
        if save_as_csv:
            self._write_on_disk(self.raw_prices, "raw_prices.csv")
        if save_as_h5:
            self._write_on_disk(self.raw_prices, "raw_prices.h5")

        return self.raw_prices

    def get_raw_returns(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int], interval='1d', column='Adj Close', save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get raw stock returns.

        Args:
            start_date (tuple): 		  Start date in the format (year, month, day).
            end_date (tuple): 			  End date in the format (year, month, day).
            interval (str, optional): 	  Interval for data (e.g., '1d' for daily). Default is '1d'.
            column (str, optional): 	  Column to retrieve (e.g., 'Adj Close'). Default is 'Adj Close'.
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing raw stock returns.
        """
        # Get raw prices
        self.get_raw_prices(start_date, end_date)

        # Calculate raw returns
        self.raw_returns = self.raw_prices.copy()
        self.raw_returns = np.log(self.raw_returns).diff()
        self.raw_returns = self.raw_returns.iloc[1:]

        # Save raw returns to file if requested
        if save_as_csv:
            self._write_on_disk(self.raw_returns, "raw_returns.csv")
        if save_as_h5:
            self._write_on_disk(self.raw_returns, "raw_returns.h5")

        return self.raw_returns

    def get_cleaned_prices(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int], interval='1d', column='Adj Close', save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get cleaned stock prices.

        Args:
            start_date (tuple): 		  Start date in the format (year, month, day).
            end_date (tuple): 			  End date in the format (year, month, day).
            interval (str, optional): 	  Interval for data (e.g., '1d' for daily). Default is '1d'.
            column (str, optional): 	  Column to retrieve (e.g., 'Adj Close'). Default is 'Adj Close'.
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing cleaned stock prices.
        """
        # Get raw prices
        self.get_raw_prices(start_date, end_date)

        # Copy raw prices to cleaned prices DataFrame
        self.cleaned_prices = self.raw_prices.copy()

        # Drop rows and columns with missing values
        self.cleaned_prices.dropna(axis='columns', how='all', inplace=True)
        self.cleaned_prices.dropna(axis='index', how='all', inplace=True)
        self.cleaned_prices.dropna(axis='columns', how='any', inplace=True)

        # Save cleaned prices to file if requested
        if save_as_csv:
            self._write_on_disk(self.cleaned_prices, "cleaned_prices.csv")
        if save_as_h5:
            self._write_on_disk(self.cleaned_prices, "cleaned_prices.h5")

        return self.cleaned_prices

    def get_cleaned_returns(self, start_date: Tuple[int, int, int], end_date: Tuple[int, int, int], index_col: str, interval='1d', column='Adj Close', save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get cleaned stock returns.

        Args:
            start_date (tuple): 		  Start date in the format (year, month, day).
            end_date (tuple): 			  End date in the format (year, month, day).
            interval (str, optional): 	  Interval for data (e.g., '1d' for daily). Default is '1d'.
            column (str, optional): 	  Column to retrieve (e.g., 'Adj Close'). Default is 'Adj Close'.
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing cleaned stock returns.
        """
        # Get cleaned prices
        self.get_cleaned_prices(start_date, end_date)
        
        # Calculate cleaned returns
        self.cleaned_returns = self.cleaned_prices.copy()
        self.cleaned_returns = np.log(self.cleaned_returns).diff()
        self.cleaned_returns = self.cleaned_returns.iloc[1:]
        self.cleaned_returns = self.cleaned_returns.reset_index()
        self.cleaned_dates   = self.cleaned_returns[index_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        self.cleaned_returns[index_col] = self.cleaned_dates
        self.cleaned_returns = self.cleaned_returns.set_index(index_col)

        # Save cleaned returns to file if requested
        if save_as_csv:
            self._write_on_disk(self.cleaned_returns, "cleaned_returns.csv")
        if save_as_h5:
            self._write_on_disk(self.cleaned_returns, "cleaned_returns.h5")

        return self.cleaned_returns

    def get_last_raw_prices(self, save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get the last raw stock prices without redownloading.

        Args:
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing the last raw stock prices.
        """
        # Return None if raw_prices is None
        if self.raw_prices is None:
            return None

        # Save last raw prices to file if requested
        if save_as_csv:
            self._write_on_disk(self.raw_prices, "raw_prices.csv")
        if save_as_h5:
            self._write_on_disk(self.raw_prices, "raw_prices.h5")

        return self.raw_prices

    def get_last_raw_returns(self, save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get the last raw stock returns without redownloading.

        Args:
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing the last raw stock returns.
        """
        # Return None if raw_returns is None
        if self.raw_returns is None:
            return None

        # Save last raw returns to file if requested
        if save_as_csv:
            self._write_on_disk(self.raw_returns, "raw_returns.csv")
        if save_as_h5:
            self._write_on_disk(self.raw_returns, "raw_returns.h5")

        return self.raw_returns

    def get_last_cleaned_prices(self, save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get the last cleaned stock prices without redownloading.

        Args:
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing the last cleaned stock prices.
        """
        # Return None if cleaned_prices is None
        if self.cleaned_prices is None:
            return None

        # Save last cleaned prices to file if requested
        if save_as_csv:
            self._write_on_disk(self.cleaned_prices, "cleaned_prices.csv")
        if save_as_h5:
            self._write_on_disk(self.cleaned_prices, "cleaned_prices.h5")

        return self.cleaned_prices

    def get_last_cleaned_returns(self, save_as_h5=False, save_as_csv=False) -> DataFrame:
        """
        Get the last cleaned stock returns without redownloading.

        Args:
            save_as_h5 (bool, optional):  Flag to save as HDF5 format. Default is False.
            save_as_csv (bool, optional): Flag to save as CSV format. Default is False.

        Returns:
            DataFrame: DataFrame containing the last cleaned stock returns.
        """
        # Return None if cleaned_returns is None
        if self.cleaned_returns is None:
            return None

        # Save last cleaned returns to file if requested
        if save_as_csv:
            self._write_on_disk(self.cleaned_returns, "cleaned_returns.csv")
        if save_as_h5:
            self._write_on_disk(self.cleaned_returns, "cleaned_returns.h5")

        return self.cleaned_returns