
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from textblob import TextBlob
from streamlit_autorefresh import st_autorefresh

from data import get_stock_data
from monte_carlo import simulate_price_paths, get_price_statistics


# ============================================================
# CONFIGURATION
# ============================================================

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(
    page_title="Stock Market Simulator",
    page_icon="📈",
    layout="wide",
)

try:
    STREAMLIT_API_KEY = st.secrets.get("NEWSAPI_KEY", "")
except Exception:
    STREAMLIT_API_KEY = ""

API_KEY = (
    os.getenv("NEWSAPI_KEY", "").strip()
    or str(STREAMLIT_API_KEY).strip()
)

BASE_NEWS_DOMAINS = [
    "finance.yahoo.com",
    "bloomberg.com",
]

COMPANIES = {
    "AAPL": {"name": "Apple", "domain": "apple.com"},
    "MSFT": {"name": "Microsoft", "domain": "microsoft.com"},
    "NVDA": {"name": "NVIDIA", "domain": "nvidia.com"},
    "GOOGL": {"name": "Alphabet", "domain": "abc.xyz"},
    "GOOG": {"name": "Alphabet", "domain": "abc.xyz"},
    "AMZN": {"name": "Amazon", "domain": "aboutamazon.com"},
    "META": {"name": "Meta", "domain": "about.fb.com"},
    "TSLA": {"name": "Tesla", "domain": "tesla.com"},
    "NFLX": {"name": "Netflix", "domain": "netflix.com"},
    "AMD": {"name": "AMD", "domain": "amd.com"},
    "INTC": {"name": "Intel", "domain": "intel.com"},
    "ORCL": {"name": "Oracle", "domain": "oracle.com"},
    "CRM": {"name": "Salesforce", "domain": "salesforce.com"},
    "COST": {"name": "Costco", "domain": "costco.com"},
    "DIS": {"name": "Disney", "domain": "thewaltdisneycompany.com"},
    "JPM": {"name": "JPMorgan Chase", "domain": "jpmorganchase.com"},
    "V": {"name": "Visa", "domain": "visa.com"},
    "WMT": {"name": "Walmart", "domain": "corporate.walmart.com"},
    "KO": {"name": "Coca-Cola", "domain": "coca-colacompany.com"},
    "NKE": {"name": "Nike", "domain": "about.nike.com"},
}

st_autorefresh(interval=60000, key="live_refresh")


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    html, body {
        background-color: #0e1117;
        color: #f3f4f6;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 14px;
        text-align: center;
    }

    .metric-label {
        font-size: 12px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 600;
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin: 20px 0 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Stock Market Simulator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Monte Carlo • Smart Signals • Entry Analysis'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# NEWS FUNCTIONS
# ============================================================

def get_company_info(ticker):
    return COMPANIES.get(
        ticker.upper().strip(),
        {
            "name": ticker.upper().strip(),
            "domain": None,
        },
    )


def is_allowed_domain(url, allowed_domains):
    """Allow article links only from the configured domains."""
    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return False

        hostname = hostname.lower().rstrip(".")

        return any(
            hostname == domain
            or hostname.endswith("." + domain)
            for domain in allowed_domains
        )

    except (ValueError, AttributeError):
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_news(ticker, company_name, company_domain, api_key):
    """Retrieve up to five recent articles from approved domains."""

    if not api_key:
        return [], (
            "NEWSAPI_KEY was not found. "
            "Check your .env file beside app.py."
        )

    allowed_domains = BASE_NEWS_DOMAINS.copy()

    if company_domain:
        allowed_domains.append(company_domain)

    params = {
        "q": f'"{ticker}" OR "{company_name}"',
        "domains": ",".join(allowed_domains),
        "pageSize": 20,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=15,
        )

        if response.status_code != 200:
            return [], (
                f"NewsAPI error {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()

        if data.get("status") != "ok":
            return [], data.get(
                "message",
                "NewsAPI request failed.",
            )

        articles = []
        seen_urls = set()

        for article in data.get("articles", []):
            title = article.get("title")
            url = article.get("url")

            if not title or not url:
                continue

            if not is_allowed_domain(url, allowed_domains):
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "source": (
                        article.get("source", {}).get("name")
                        or "News source"
                    ),
                    "published_at": (
                        article.get("publishedAt") or ""
                    ),
                }
            )

            if len(articles) == 5:
                break

        return articles, None

    except requests.RequestException as error:
        return [], f"Could not retrieve news: {error}"

    except ValueError:
        return [], "NewsAPI returned an invalid response."


def calculate_news_sentiment(articles):
    """
    Internal model input only.
    This score is not displayed in the news section.
    """
    if not articles:
        return 0.0

    scores = [
        TextBlob(article["title"]).sentiment.polarity
        for article in articles
        if article.get("title")
    ]

    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Controls")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="NVDA",
).upper().strip()

lookback_days = st.sidebar.slider(
    "Historical lookback (days)",
    min_value=30,
    max_value=252,
    value=60,
)

future_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=5,
    max_value=180,
    value=30,
)

sims = st.sidebar.slider(
    "Number of simulations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
)

vol_shock = st.sidebar.slider(
    "Volatility multiplier",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

company = get_company_info(ticker)


# ============================================================
# LOAD STOCK DATA
# ============================================================

try:
    current_price, returns = get_stock_data(
        ticker,
        lookback_days,
    )

    current_price = float(current_price)

    returns = np.asarray(
        returns,
        dtype=float,
    ).flatten()

    returns = returns[np.isfinite(returns)]

    if current_price <= 0 or len(returns) < 2:
        st.error(
            "Not enough valid market data was returned "
            "for this ticker."
        )
        st.stop()

    mu = float(np.mean(returns))
    sigma = float(np.std(returns)) * vol_shock

except Exception as error:
    st.error(f"Stock data error: {error}")
    st.stop()


# ============================================================
# LOAD NEWS FOR THE MODEL
# ============================================================

headlines, news_error = get_news(
    ticker,
    company["name"],
    company["domain"],
    API_KEY,
)

news_sentiment = calculate_news_sentiment(headlines)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

try:
    paths = simulate_price_paths(
        current_price,
        mu,
        sigma,
        future_days,
        sims,
        news_sentiment=news_sentiment,
    )

    paths = np.asarray(paths, dtype=float)

    if (
        paths.ndim != 2
        or paths.shape[0] < 2
        or paths.shape[1] < 1
    ):
        st.error(
            "The simulation returned an unexpected data format."
        )
        st.stop()

    stats = get_price_statistics(paths)
    final_prices = paths[-1]

except Exception as error:
    st.error(f"Simulation error: {error}")

    st.info(
        "Check that simulate_price_paths() in monte_carlo.py "
        "accepts the news_sentiment argument."
    )

    st.stop()


# ============================================================
# METRICS
# ============================================================

expected_return = float(
    np.mean(
        (final_prices - current_price) / current_price
    ) * 100
)

# This is a simplified risk-scaled measure, not a conventional
# annualized Sharpe ratio.
sharpe_ratio = float(
    (np.mean(final_prices) - current_price)
    / (np.std(final_prices) + 1e-9)
)


# ============================================================
# TREND DETECTION
# ============================================================

def detect_trend(paths_array):
    median_path = np.median(paths_array, axis=1)

    midpoint = len(median_path) // 2

    if (
        median_path[midpoint] < median_path[0]
        and median_path[-1] > median_path[0]
    ):
        return "dip_buy"

    if median_path[-1] < median_path[0]:
        return "downtrend"

    if median_path[-1] > median_path[0]:
        return "uptrend"

    return "neutral"


trend = detect_trend(paths)


# ============================================================
# POSITION AND HISTORICAL RETURN ESTIMATE
# ============================================================

position = float(
    (current_price - stats["p5"])
    / (stats["p95"] - stats["p5"] + 1e-9)
)

# Annualized mean daily return estimate; not a guaranteed forecast.
long_term_return = float(np.mean(returns) * 252 * 100)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    expected_return_value,
    sharpe_value,
    sentiment,
    trend_name,
    position_value,
    long_term,
):
    score = 0

    score += (
        2 if expected_return_value > 2
        else 1 if expected_return_value > 0
        else -2
    )

    score += (
        2 if sharpe_value > 1
        else 1 if sharpe_value > 0.5
        else -1
    )

    score += (
        2 if sentiment > 0.1
        else -2 if sentiment < -0.1
        else 0
    )

    score += (
        2 if trend_name == "dip_buy"
        else -2 if trend_name == "downtrend"
        else 1 if trend_name == "uptrend"
        else 0
    )

    score += (
        2 if position_value < 0.3
        else -1 if position_value > 0.8
        else 0
    )

    score += (
        1 if long_term > 15
        else -1 if long_term < 0
        else 0
    )

    if score >= 6:
        return "🟢 STRONG BUY", score

    if score >= 3:
        return "🟢 BUY", score

    if score >= 0:
        return "🟡 HOLD", score

    if score <= -6:
        return "🔴 STRONG SELL", score

    return "🔴 SELL", score


signal, signal_score = generate_signal(
    expected_return,
    sharpe_ratio,
    news_sentiment,
    trend,
    position,
    long_term_return,
)


# ============================================================
# ENTRY ANALYSIS
# ============================================================

def entry_analysis(paths_array, price_now):
    median_path = np.median(paths_array, axis=1)

    dip_low = float(np.percentile(median_path, 10))
    dip_high = float(np.percentile(median_path, 30))

    # This estimates whether a simulated path ever exceeds
    # today's price. It is not a calibrated recovery probability.
    recoveries = np.sum(
        np.max(paths_array, axis=0) > price_now
    )

    recovery_probability = (
        float(recoveries) / paths_array.shape[1]
    )

    worst_case = float(
        np.percentile(paths_array[-1], 5)
    )

    downside_risk = float(
        (price_now - worst_case) / price_now
    )

    return (
        dip_low,
        dip_high,
        recovery_probability,
        downside_risk,
    )


(
    dip_low,
    dip_high,
    recovery_prob,
    downside_risk,
) = entry_analysis(paths, current_price)


def interpret_entry(position_value, recovery_probability, downside):
    if position_value > 0.8:
        return "⚠️ Price is near the upper end of the modeled range."

    if position_value < 0.3 and recovery_probability > 0.6:
        return "🟢 Price is in the lower part of the modeled range."

    if downside > 0.2:
        return "🔴 The model indicates elevated downside risk."

    return "🟡 The model does not indicate a clear entry zone."


entry_message = interpret_entry(
    position,
    recovery_prob,
    downside_risk,
)


# ============================================================
# METRIC CARD HELPER
# ============================================================

def card(label, value):
    return (
        "<div class='card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        "</div>"
    )


# ============================================================
# PRICE METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Price Metrics</div>',
    unsafe_allow_html=True,
)

cols = st.columns(5)

cols[0].markdown(
    card("Current Price", f"${current_price:.2f}"),
    unsafe_allow_html=True,
)

cols[1].markdown(
    card("Mean", f"${stats['mean']:.2f}"),
    unsafe_allow_html=True,
)

cols[2].markdown(
    card("Median", f"${stats['median']:.2f}"),
    unsafe_allow_html=True,
)

cols[3].markdown(
    card("5th Percentile", f"${stats['p5']:.2f}"),
    unsafe_allow_html=True,
)

cols[4].markdown(
    card("95th Percentile", f"${stats['p95']:.2f}"),
    unsafe_allow_html=True,
)


# ============================================================
# RISK METRICS
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Risk Metrics</div>',
    unsafe_allow_html=True,
)

risk_cols = st.columns(3)

risk_cols[0].markdown(
    card("Expected Return", f"{expected_return:.2f}%"),
    unsafe_allow_html=True,
)

risk_cols[1].markdown(
    card("Sharpe-like Measure", f"{sharpe_ratio:.2f}"),
    unsafe_allow_html=True,
)

risk_cols[2].markdown(
    card("Position", f"{position:.2f}"),
    unsafe_allow_html=True,
)


# ============================================================
# TRADING SIGNAL
# ============================================================

st.markdown(
    '<div class="section-title">Trading Signal</div>',
    unsafe_allow_html=True,
)

if "BUY" in signal:
    st.success(f"{signal} · Score: {signal_score}")

elif "SELL" in signal:
    st.error(f"{signal} · Score: {signal_score}")

else:
    st.info(f"{signal} · Score: {signal_score}")

st.caption(
    "Experimental rule-based signal, not a validated trading strategy."
)


# ============================================================
# ENTRY ANALYSIS DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Entry Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    **Modeled Price Zone:** ${dip_low:.2f} – ${dip_high:.2f}

    **Estimated Path Recovery:** {recovery_prob * 100:.1f}%

    **Modeled Downside:** {downside_risk * 100:.1f}%
    """
)

if "🟢" in entry_message:
    st.success(entry_message)

elif "🔴" in entry_message:
    st.error(entry_message)

elif "⚠️" in entry_message:
    st.warning(entry_message)

else:
    st.info(entry_message)


# ============================================================
# MONTE CARLO CHART
# ============================================================

median_path = np.median(paths, axis=1)

best_index = int(np.argmax(final_prices))
worst_index = int(np.argmin(final_prices))

days = np.arange(len(median_path))

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=days,
        y=median_path,
        name="Median Path",
        line=dict(width=3),
    )
)

fig.add_trace(
    go.Scatter(
        x=days,
        y=paths[:, best_index],
        name="Best Simulated Path",
        line=dict(dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=days,
        y=paths[:, worst_index],
        name="Worst Simulated Path",
        line=dict(dash="dash"),
    )
)

fig.add_hrect(
    y0=dip_low,
    y1=dip_high,
    fillcolor="yellow",
    opacity=0.10,
    line_width=0,
)

fig.add_hline(
    y=current_price,
    line_dash="dot",
    annotation_text="Current Price",
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Day",
    yaxis_title="Price ($)",
    hovermode="x unified",
    legend_title="Simulation",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# LATEST NEWS — FIVE HEADLINES, NO SENTIMENT DISPLAY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Latest News</div>',
    unsafe_allow_html=True,
)

if news_error:
    st.warning(news_error)

elif headlines:
    for article in headlines[:5]:
        st.markdown(
            f"**[{article['title']}]({article['url']})**"
        )

        details = article["source"]
        published = article.get("published_at", "")

        if published:
            details += f" · {published[:10]}"

        st.caption(details)
        st.divider()

else:
    st.info("No recent headlines found for this stock.")


# ============================================================
# MODEL NOTES
# ============================================================

with st.expander("Model Assumptions and Limitations"):
    st.markdown(
        """
        - The simulation uses historical returns and volatility
          from `data.py`.
        - Simulated paths are possible scenarios, not guaranteed forecasts.
        - Trading signals and entry analysis are experimental heuristics.
        - The best and worst paths are extremes from the simulations,
          not limits on real-world prices.
        - The modeled price zone is a rough visualization, not a
          statistically validated buying range.
        """
    )

st.caption(
    "Stock Intelligence Terminal · Educational research tool"
)