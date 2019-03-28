# This module creates a container for stock market data objects
# to be used to input into the model.
#
#

from pandas_datareader import data
import pandas as pd

# Function for downloading the data frame object for symbol and date range
# source can be specified but yahoo is known to work and therefore default
# weekends and holidays are removed from the data frame
def LoadStockDataForSymbol(symbol, start_date, end_date, source='yahoo'):
    # Use pandas_reader.data.DataReader to load the desired data
    panel_data = data.DataReader(symbol, source, start_date, end_date)

    # Get all of the weekdays within the range specified above
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

    # Reindex close using all_weekdays as new index
    panel_data = panel_data.reindex(all_weekdays)

    # Fill in missing dates
    panel_data = panel_data.fillna(method='ffill')

    return panel_data

# Container for a stock data object with attributes to make accessing data
# easier
class stock(object):

  def __init__(self, symbol, start_date, end_date, source='yahoo'):
    self.data_frame_ = LoadStockDataForSymbol(symbol, start_date, end_date, source)
    self.close_ = self.data_frame_['Close']
    self.open_ = self.data_frame_['Open']
    self.high_ = self.data_frame_['High']
    self.low_ = self.data_frame_['Low']
    self.volume_ = self.data_frame_['Volume']
    self.adj_close_ = self.data_frame_['Adj Close']
    self.dates_ = self.data_frame_.index.values
    self.close_mov_avg_20_ = self.close_.rolling(window=20).mean()
    self.close_mov_avg_100_ = self.close_.rolling(window=100).mean()
