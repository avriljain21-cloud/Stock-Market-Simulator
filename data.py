import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker, lookback_days):
    """
    Downloads historical stock data and calculates log returns.

    Parameters:
    - ticker (str): Stock symbol, e.g., "AAPL"
    - lookback_days (int): Number of past days to use for simulation

    Returns:
    - S0 (float): Latest closing price
    - returns (pd.Series): Log returns for the lookback period
    """
    data = yf.download(ticker, period=f"{lookback_days}d")
    if data.empty:
        raise ValueError("Ticker not found or no data available")
    prices = data['Close']
    S0 = float(prices.iloc[-1])  # ensure S0 is a float
    returns = np.log(prices / prices.shift(1)).dropna()
    return S0, returns