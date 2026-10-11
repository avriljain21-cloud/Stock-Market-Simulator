# Stock Market Simulator

An interactive stock market simulator that uses Monte Carlo simulations to explore possible future stock-price outcomes.

## Overview

The Stock Market Simulator helps users explore uncertainty in the stock market by generating thousands of simulated future price paths based on historical stock data.

Rather than predicting one exact future price, the application visualizes a range of possible outcomes.

## Features

* **Monte Carlo Simulations:** Generate thousands of possible future stock-price paths.
* **Customizable Forecasts:** Choose a forecast horizon and number of simulations.
* **Interactive Visualizations:** Explore simulated price paths and statistical ranges.
* **Stock Data:** Retrieve historical market data using Yahoo Finance.
* **Financial News:** Access relevant stock news through Yahoo Finance.

## Technologies Used

* Python
* Streamlit
* NumPy
* Plotly
* yfinance

## How It Works

The application uses historical stock returns and volatility to simulate possible future prices. It then analyzes the outcomes using statistics such as the mean, median, and percentile ranges.

These simulations illustrate possible scenarios, not guaranteed future prices.

## Disclaimer

This project is for educational purposes only. Simulated outcomes are not financial advice and should not be treated as predictions of actual market performance.
