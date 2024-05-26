import pandas as pd
import numpy as np
try:
    import yfinance as yf  # Importing a package to fetch financial data
except ImportError:
    raise ImportError("Cannot start without 'yfinance' package.\n Install it before running the code again.")

import pandas_market_calendars
from pandas_market_calendars import get_calendar  # Importing a package to work with market calendars

class RiskFreeRateDownloader:
    """
    A class to download and preprocess risk-free rates from Yahoo Finance.

    Attributes:
        start_date (str): Start date for downloading risk-free rates.
        end_date (str): End date for downloading risk-free rates.
    """

    def __init__(self, start_date="2000-01-01", end_date="2024-03-31"):
        """
        Initializes the RiskFreeRateDownloader class.

        Args:
            start_date (str, optional): Start date for downloading risk-free rates. Default is "2000-01-01".
            end_date (str, optional): End date for downloading risk-free rates. Default is "2024-03-31".
        """
        self.start_date = start_date  # Start date for downloading risk-free rates
        self.end_date = end_date  # End date for downloading risk-free rates

    def deannualize_rate(self, annual_rate, periods=12):
        """
        De-annualize yearly interest rates.

        Args:
            annual_rate (float): Annual interest rate.
            periods (int, optional): Number of periods in a year. Default is 12.

        Returns:
            float: De-annualized interest rate.
        """
        return (1 + annual_rate) ** (1/periods) - 1  # Formula to de-annualize the interest rate

    def get_risk_free_rate(self):
        """
        Download and preprocess 3-month US Treasury bills rates.

        Returns:
            pd.DataFrame: A DataFrame containing monthly risk-free rates.
        """
        # Download 3-month US Treasury bills rates from Yahoo Finance
        annualized_rate = yf.download("^IRX")["Adj Close"]

        # De-annualize the annualized rates to get monthly rates
        monthly_rate = annualized_rate.apply(self.deannualize_rate)

        # Create a DataFrame for monthly risk-free rates
        rates_df = pd.DataFrame({"monthly_risk_free_rate": monthly_rate})
        
        # Filter the rates for the specified start and end dates
        rates_df = rates_df[(rates_df.index >= pd.to_datetime(self.start_date)) & (rates_df.index <= pd.to_datetime(self.end_date))]
        
        # Resample the rates to get monthly averages
        rates_df.index.name = 'Date'
        rates_df.index = pd.to_datetime(rates_df.index)
        rates_df = pd.DataFrame(rates_df.resample('1M').mean() / 100)  # Convert to percentage
        
        # Format the index to a specific date-time format
        rates_df.index = rates_df.index.strftime('%Y/%m/%d %H:%M:%S')

        return rates_df

    def interpolate_monthly_data(self, calendar_choice: str, date_col=None, resample_col=None):
        """
        Interpolate missing data and filter out non-business days.

        Args:
            calendar_choice (str): The choice of calendar, e.g., 'NYSE'.
            date_col (str, optional): Column name for dates. Default is None.
            resample_col (str, optional): Column name for resampled rates. Default is None.

        Returns:
            pd.DataFrame: A DataFrame with interpolated and filtered risk-free rates.
        """
        rates_df = self.get_risk_free_rate()  # Get monthly risk-free rates
        
        rates_df = rates_df.reset_index()  # Reset index for manipulation
        rates_df = rates_df.copy()  # Create a copy to avoid modifying original DataFrame

        if date_col is None:
            date_col = rates_df.columns[0]  # Set default column name for dates if not provided

        if resample_col is None:
            resample_col = rates_df.columns[1]  # Set default column name for resampled rates if not provided

        # Convert dates to datetime format
        rates_df[date_col] = pd.to_datetime(rates_df[date_col], format="%Y-%m")
        
        # Calculate start and end of each month
        rates_df['start_of_month'] = (rates_df[date_col].dt.floor('d') + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1))
        rates_df['end_of_month'] = pd.to_datetime(rates_df['start_of_month']) + pd.offsets.MonthEnd(1)
        
        # Calculate number of days in each month
        rates_df['days_in_month'] = (rates_df['end_of_month'] - rates_df['start_of_month']).dt.days + 1
        
        # Resample rates to daily frequency and adjust for the number of days in each month
        rates_df[resample_col] = rates_df[resample_col] / rates_df['days_in_month']
        
        # Set start_of_month as index for resampling
        reindexed_df = rates_df.set_index("start_of_month")
        
        # Remove duplicate labels by grouping and selecting the first occurrence
        reindexed_df = reindexed_df.groupby(reindexed_df.index).first()
        
        # Reindex the DataFrame with a daily frequency
        reindexed_df = reindexed_df.reindex(pd.date_range(start=reindexed_df.index.min(),
                                            end=reindexed_df.end_of_month.max(),
                                            freq='1D'))
        
        # Interpolate missing values linearly
        resampled_df = reindexed_df[resample_col]
        resampled_df.replace({0:np.nan}, inplace=True)  # Replace zeros with NaNs for interpolation
        resampled_df = resampled_df.interpolate(method='linear')  # Interpolate missing values
        resampled_df = resampled_df.reset_index()
        resampled_df.rename({'index': date_col, 'monthly_risk_free_rate': 'daily_rFrate'}, axis=1, inplace=True)
        resampled_df.fillna(0, inplace=True)  # Fill remaining NaNs with zeros
        resampled_df[date_col] = resampled_df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')  # Format dates
        resampled_df = resampled_df.set_index(date_col)

        # Choose the calendar
        calendar = get_calendar(calendar_choice)  # Retrieve the specified market calendar
        
        # Get valid trading days from the calendar
        valid_days = calendar.valid_days(start_date='2000-01-04', end_date='2024-03-31').strftime('%Y-%m-%d %H:%M:%S')
        
        # Filter out non-business days from the interpolated DataFrame
        resampled_df = resampled_df[resampled_df.index.isin(valid_days)]

        return resampled_df