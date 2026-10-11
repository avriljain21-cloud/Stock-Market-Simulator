import numpy as np

def simulate_price_paths(current_price, mu, sigma, days, sims, news_sentiment=0):
    """
    Monte Carlo simulation with optional news sentiment adjustment.

    Parameters:
        current_price (float): starting price
        mu (float): historical mean return
        sigma (float): volatility
        days (int): forecast horizon
        sims (int): number of simulations
        news_sentiment (float): -1 = very negative, 0 = neutral, +1 = very positive
    """
    dt = 1
    paths = np.zeros((days, sims))
    paths[0] = current_price

    # Adjust drift slightly based on news sentiment
    mu_adjusted = mu + news_sentiment * 0.02  # tweak scale factor if needed

    for t in range(1, days):
        rand = np.random.normal(0, 1, sims)
        paths[t] = paths[t - 1] * (1 + mu_adjusted + sigma * rand)

    return paths

def get_price_statistics(paths):
    final_prices = paths[-1]
    return {
        "mean": np.mean(final_prices),
        "median": np.median(final_prices),
        "p5": np.percentile(final_prices, 5),
        "p95": np.percentile(final_prices, 95)
    }