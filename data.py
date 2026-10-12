
import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, lookback_days):
    """
    Downloads historical stock data and calculates log returns.

    Returns:
        S0 (float): Latest closing price
        returns (pd.Series): Historical log returns
    """
    data = yf.download(
        ticker,
        period=f"{lookback_days}d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    if data.empty:
        raise ValueError("Ticker not found or no data available.")

    prices = data["Close"]

    # Handle cases where Close is still a DataFrame.
    if isinstance(prices, pd.DataFrame):
        if ticker in prices.columns:
            prices = prices[ticker]
        else:
            prices = prices.iloc[:, 0]

    prices = pd.to_numeric(prices, errors="coerce").dropna()

    if prices.empty:
        raise ValueError("No valid closing prices were returned.")

    S0 = float(prices.iloc[-1])

    returns = np.log(prices / prices.shift(1)).dropna()

    if returns.empty:
        raise ValueError("Not enough historical data to calculate returns.")

    return S0, returns
