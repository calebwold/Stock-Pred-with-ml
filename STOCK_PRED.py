import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import base64
import os
import json
import requests
import datetime
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
import pandas_ta as ta
import time
from typing import Optional, Union, Dict, Any

# Read from Streamlit Secrets Manager
HF_TOKEN = st.secrets["HF_TOKEN"]
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]

# Initialize Hugging Face client

from openai import OpenAI
# Available models on Hugging Face (you can change these)
HF_MODELS = {
    "GPT-OSS-120B": "openai/gpt-oss-120b:novita",
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "CodeLlama-34B": "codellama/CodeLlama-34B-Instruct-hf"
}

# Default model to use
DEFAULT_HF_MODEL = "GPT-OSS-120B"

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Test API key function
def test_alpha_vantage_api_key():
    """Test if the Alpha Vantage API key is valid"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            return False, f"API Key Error: {data['Error Message']}"
        elif "Note" in data:
            return False, f"API Limit: {data['Note']}"
        elif "Time Series (Daily)" in data:
            return True, "API Key is valid"
        else:
            return False, f"Unexpected response: {data}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# Cache configuration for better performance
def get_alpha_vantage_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get stock data from Alpha Vantage API"""
    try:
        # Alpha Vantage TIME_SERIES_DAILY endpoint
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        
        st.info(f"Fetching data from Alpha Vantage for {ticker}...")
        
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            st.error(f"Alpha Vantage HTTP error: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
        
        # Debug: Show the response structure
        st.info(f"Alpha Vantage response keys: {list(data.keys())}")
        
        # Check for API errors
        if "Error Message" in data:
            st.error(f"Alpha Vantage error: {data['Error Message']}")
            return pd.DataFrame()
        
        if "Note" in data:
            st.warning(f"Alpha Vantage limit reached: {data['Note']}")
            return pd.DataFrame()
        
        if "Time Series (Daily)" not in data:
            st.error(f"No time series data found in Alpha Vantage response. Response: {data}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert string values to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename columns to match yfinance format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add missing columns that yfinance provides
        df['Adj Close'] = df['Close']  # Alpha Vantage doesn't provide adjusted close
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # Sort by date
        df = df.sort_index()
        
        # Reset index to make date a column (like yfinance format)
        df = df.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        
        if not df.empty:
            return df
        else:
            st.warning(f"No data found for {ticker} in the specified date range")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching Alpha Vantage data: {str(e)}")
        return pd.DataFrame()

def cached_stock_data(ticker: str, start_date: str, end_date: str, data_source: str = "Alpha Vantage (Recommended)") -> pd.DataFrame:
    """Cache stock data to avoid repeated API calls"""
    import time
    import random
    
    # Check if user wants to use Alpha Vantage
    if data_source == "Alpha Vantage (Recommended)":
        st.info("Using Alpha Vantage for stock data...")
        data = get_alpha_vantage_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            return data
        else:
            st.warning("Alpha Vantage failed, falling back to Yahoo Finance...")
    
    # Fallback to Yahoo Finance
    time.sleep(random.uniform(2, 5))  # Shorter delay for fallback
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=False
            )
            
            if data is not None and not data.empty and len(data) > 0:
                return data
            else:
                st.warning(f"Yahoo Finance attempt {attempt + 1}: No data returned for {ticker}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                wait_time = (attempt + 1) * 20
                st.warning(f"Yahoo Finance rate limited! Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Yahoo Finance error for {ticker}: {str(e)}")
                time.sleep(3)
    
    return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_fundamental_data(ticker: str, data_source: str = "Alpha Vantage (Recommended)") -> Optional[Dict[str, Any]]:
    """Cache fundamental data"""
    return get_fundamental_data(ticker, data_source)

@st.cache_data(ttl=900)  # Cache for 15 minutes
def cached_sentiment_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Cache sentiment data"""
    return get_stock_sentiment(ticker)

# Weather utility function
def get_weather(city_state: str, api_key: str) -> str:
    try:
        # Normalize the input by stripping extra spaces and capitalizing properly
        city_state = city_state.strip().title().replace(" ", ", ")
        
        # Define a dictionary for state abbreviations
        state_abbr = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
            "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
            "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
            "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
            "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
        }

        # Check if a state abbreviation or full name is provided
        for full_state, abbr in state_abbr.items():
            if full_state.lower() in city_state.lower():
                city_state = city_state.replace(full_state, abbr)
                break
        
        # Call the OpenWeatherMap API with the normalized input
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city_state}&appid={api_key}&units=imperial"
        )
        
        if response.status_code == 200:
            data = response.json()
            weather_desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            return f"{city_state}: {weather_desc}, {temp}°F"
        else:
            return "Unable to fetch weather data. Please check your entered city and state."
    except Exception as e:
        return f"Error fetching weather data: {e}"

# Functions for technical indicators and analysis
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators for stock analysis"""
    # Basic price indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Exponential moving averages
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_STD'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_STD'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_STD'])
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD_Line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    
    # VWAP (Volume Weighted Average Price)
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # ATR (Average True Range) - Volatility indicator
    high_low = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # OBV (On-Balance Volume)
    data['OBV'] = 0
    price_change = data['Close'].diff()
    data.loc[price_change > 0, 'OBV'] = data.loc[price_change > 0, 'Volume']
    data.loc[price_change < 0, 'OBV'] = -data.loc[price_change < 0, 'Volume']
    data['OBV'] = data['OBV'].cumsum()
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    numerator = data['Close'] - low_14
    denominator = high_14 - low_14
    data['%K'] = 100 * (numerator / denominator)
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    return data

def get_stock_sentiment(ticker: str, num_articles: int = 5) -> Dict[str, Any]:
    """Get sentiment analysis from news articles"""
    try:
        # Use Alpha Vantage News Sentiment API
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return {"score": 0, "articles": [], "error": "Failed to fetch news data"}
        
        news_data = response.json()
        
        if "feed" not in news_data:
            return {"score": 0, "articles": [], "error": "No news data available"}
            
        articles = []
        sentiment_scores = []
        
        # Process articles and sentiment
        for article in news_data["feed"][:num_articles]:
            title = article.get("title", "No title")
            
            # Use built-in sentiment if available or calculate with TextBlob
            if "overall_sentiment_score" in article:
                sentiment = float(article["overall_sentiment_score"])
            else:
                blob = TextBlob(title)
                sentiment = TextBlob(title).sentiment.polarity
                
            sentiment_scores.append(sentiment)
            
            articles.append({
                "title": title,
                "url": article.get("url", ""),
                "time_published": article.get("time_published", ""),
                "sentiment": sentiment
            })
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            "score": avg_sentiment,
            "articles": articles,
            "error": None
        }
        
    except Exception as e:
        return {"score": 0, "articles": [], "error": str(e)}

def get_alpha_vantage_fundamental_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get fundamental data from Alpha Vantage"""
    try:
        # Get company overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url, timeout=10)
        
        if overview_response.status_code != 200:
            return None
        
        overview_data = overview_response.json()
        
        if "Error Message" in overview_data:
            st.error(f"Alpha Vantage overview error: {overview_data['Error Message']}")
            return None
        
        if "Note" in overview_data:
            st.warning(f"Alpha Vantage limit reached: {overview_data['Note']}")
            return None
        
        # Extract fundamental data
        financials = {
            "name": overview_data.get("Name", ticker),
            "sector": overview_data.get("Sector", ""),
            "industry": overview_data.get("Industry", ""),
            "market_cap": float(overview_data.get("MarketCapitalization", 0)),
            "pe_ratio": float(overview_data.get("PERatio", 0)) if overview_data.get("PERatio") else None,
            "forward_pe": float(overview_data.get("ForwardPE", 0)) if overview_data.get("ForwardPE") else None,
            "peg_ratio": float(overview_data.get("PEGRatio", 0)) if overview_data.get("PEGRatio") else None,
            "eps": float(overview_data.get("EPS", 0)) if overview_data.get("EPS") else None,
            "dividend_yield": float(overview_data.get("DividendYield", 0)) if overview_data.get("DividendYield") else None,
            "52w_high": float(overview_data.get("52WeekHigh", 0)) if overview_data.get("52WeekHigh") else None,
            "52w_low": float(overview_data.get("52WeekLow", 0)) if overview_data.get("52WeekLow") else None,
            "price_to_book": float(overview_data.get("PriceToBookRatio", 0)) if overview_data.get("PriceToBookRatio") else None,
            "beta": float(overview_data.get("Beta", 0)) if overview_data.get("Beta") else None,
            "debt_to_equity": float(overview_data.get("DebtToEquityRatio", 0)) if overview_data.get("DebtToEquityRatio") else None,
            "return_on_equity": float(overview_data.get("ReturnOnEquityTTM", 0)) if overview_data.get("ReturnOnEquityTTM") else None,
            "profit_margins": float(overview_data.get("ProfitMargin", 0)) if overview_data.get("ProfitMargin") else None,
            "revenue_growth": float(overview_data.get("QuarterlyRevenueGrowthYOY", 0)) if overview_data.get("QuarterlyRevenueGrowthYOY") else None,
        }
        
        # Calculate financial health score
        score = 0
        count = 0
        
        if financials["pe_ratio"] and 5 < financials["pe_ratio"] < 25:
            score += 10
            count += 1
            
        if financials["peg_ratio"] and 0 < financials["peg_ratio"] < 1.5:
            score += 10
            count += 1
            
        if financials["debt_to_equity"] and financials["debt_to_equity"] < 1:
            score += 10
            count += 1
            
        if financials["return_on_equity"] and financials["return_on_equity"] > 15:
            score += 10
            count += 1
            
        if financials["profit_margins"] and financials["profit_margins"] > 10:
            score += 10
            count += 1
            
        financial_health_score = round(score / count * 10) if count > 0 else 0
        financials["financial_health_score"] = financial_health_score
        
        return financials
        
    except Exception as e:
        st.error(f"Error fetching Alpha Vantage fundamental data: {e}")
        return None

def get_fundamental_data(ticker: str, data_source: str = "Alpha Vantage (Recommended)") -> Dict[str, Any]:
    """Get fundamental financial data for a stock"""
    try:
        # Try Alpha Vantage first if selected
        if data_source == "Alpha Vantage (Recommended)":
            alpha_data = get_alpha_vantage_fundamental_data(ticker)
            if alpha_data:
                return alpha_data
            else:
                st.warning("Alpha Vantage fundamental data failed, using Yahoo Finance...")
        
        # Fallback to Yahoo Finance
        company = yf.Ticker(ticker)
        info = company.info
        
        financials = {
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
            "eps": info.get("trailingEps", None),
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") is not None else None,
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "price_to_book": info.get("priceToBook", None),
            "beta": info.get("beta", None),
            "debt_to_equity": (info.get("debtToEquity", 0) / 100) if info.get("debtToEquity") is not None else None,
            "return_on_equity": (info.get("returnOnEquity", 0) * 100) if info.get("returnOnEquity") is not None else None,
            "profit_margins": (info.get("profitMargins", 0) * 100) if info.get("profitMargins") is not None else None,
            "revenue_growth": (info.get("revenueGrowth", 0) * 100) if info.get("revenueGrowth") is not None else None,
        }
        
        # Calculate financial health score
        score = 0
        count = 0
        
        if financials["pe_ratio"] and 5 < financials["pe_ratio"] < 25:
            score += 10
            count += 1
            
        if financials["peg_ratio"] and 0 < financials["peg_ratio"] < 1.5:
            score += 10
            count += 1
            
        if financials["debt_to_equity"] and financials["debt_to_equity"] < 1:
            score += 10
            count += 1
            
        if financials["return_on_equity"] and financials["return_on_equity"] > 15:
            score += 10
            count += 1
            
        if financials["profit_margins"] and financials["profit_margins"] > 10:
            score += 10
            count += 1
            
        financial_health_score = round(score / count * 10) if count > 0 else 0
        financials["financial_health_score"] = financial_health_score
        
        return financials
        
    except Exception as e:
        st.error(f"Error fetching fundamental data: {e}")
        return {
            "name": ticker,
            "error": str(e),
            "financial_health_score": 0
        }

def get_ai_analysis(ticker: str, price_data: pd.DataFrame, forecast_data: pd.DataFrame, fundamental_data: Dict[str, Any], sentiment_data: Dict[str, Any], model_name: str = DEFAULT_HF_MODEL) -> str:
    """Get AI analysis of the stock using Hugging Face API"""
    try:
        # Current price and metrics
        current_price = price_data['Close'].iloc[-1]
        price_change_1d = ((current_price / price_data['Close'].iloc[-2]) - 1) * 100
        price_change_1w = ((current_price / price_data['Close'].iloc[-6]) - 1) * 100 if len(price_data) >= 6 else 0
        price_change_1m = ((current_price / price_data['Close'].iloc[-22]) - 1) * 100 if len(price_data) >= 22 else 0
        
        # Technical indicators
        latest_data = price_data.iloc[-1]
        rsi = latest_data.get('RSI', 0)
        macd = latest_data.get('MACD_Line', 0)
        macd_signal = latest_data.get('MACD_Signal', 0)
        
        # Forecast data
        forecast_price = forecast_data['yhat'].iloc[-1]
        forecast_change = ((forecast_price / current_price) - 1) * 100
        
        # Create prompt for Hugging Face API
        prompt = f"""
        You are a professional financial analyst. Analyze the following stock data for {ticker} and provide a concise, insightful analysis with future outlook.
        
        CURRENT DATA:
        - Current Price: ${current_price:.2f}
        - 1-Day Change: {price_change_1d:.2f}%
        - 1-Week Change: {price_change_1w:.2f}%
        - 1-Month Change: {price_change_1m:.2f}%
        
        TECHNICAL INDICATORS:
        - RSI (14): {rsi:.2f}
        - MACD: {macd:.4f}
        - MACD Signal: {macd_signal:.4f}
        
        FUNDAMENTAL DATA:
        - Sector: {fundamental_data.get('sector', 'Unknown')}
        - P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}
        - Forward P/E: {fundamental_data.get('forward_pe', 'N/A')}
        - PEG Ratio: {fundamental_data.get('peg_ratio', 'N/A')}
        - Debt-to-Equity: {fundamental_data.get('debt_to_equity', 'N/A')}
        - Return on Equity: {fundamental_data.get('return_on_equity', 'N/A')}%
        - Financial Health Score: {fundamental_data.get('financial_health_score', 0)}/100
        
        SENTIMENT:
        - News Sentiment Score: {sentiment_data.get('score', 0):.2f} (-1 to 1 scale)
        
        FORECAST:
        - Predicted Price in Future: ${forecast_price:.2f} ({forecast_change:.2f}%)
        
        Provide a 3-paragraph analysis:
        1. Technical Analysis: Interpret the current price action and technical indicators
        2. Fundamental Analysis: Evaluate company financial health and valuation
        3. Outlook: Provide a forecast based on all factors, discussing potential price movements
        
        Keep your analysis concise and focused on actionable insights.
        """
        
        # Use Hugging Face API for AI analysis
        try:
            completion = hf_client.chat.completions.create(
                model=HF_MODELS[model_name],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return completion.choices[0].message.content or "AI analysis unavailable"
            
        except Exception as api_error:
            st.warning(f"AI analysis API error: {api_error}. Using fallback analysis.")
            # Fallback to rule-based analysis if API fails
        
        sentiment_word = "positive" if sentiment_data.get('score', 0) > 0.2 else "neutral" if sentiment_data.get('score', 0) > -0.2 else "negative"
        forecast_direction = "bullish" if forecast_change > 0 else "bearish"
        rsi_condition = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        # Create different response templates based on technical conditions
        if rsi > 70 and macd < macd_signal:
            technical_analysis = f"**Technical Analysis**: {ticker} appears technically {rsi_condition} with an RSI of {rsi:.2f}, suggesting the stock may be due for a pullback. The MACD shows bearish divergence as the MACD line has crossed below the signal line, reinforcing the potential for a short-term price correction. Volume patterns indicate decreasing buying momentum."
        elif rsi < 30 and macd > macd_signal:
            technical_analysis = f"**Technical Analysis**: {ticker} is showing {rsi_condition} conditions with an RSI of {rsi:.2f}, potentially offering a buying opportunity. The MACD has formed a bullish crossover with the MACD line moving above the signal line, suggesting growing momentum. Recent price action indicates a potential bottom formation."
        else:
            technical_analysis = f"**Technical Analysis**: {ticker} is currently in a {rsi_condition} zone with an RSI of {rsi:.2f}. The MACD indicator shows {'bullish' if macd > macd_signal else 'bearish'} momentum. Price action is {'above' if current_price > latest_data.get('SMA_50', 0) else 'below'} the 50-day moving average, indicating a {'positive' if current_price > latest_data.get('SMA_50', 0) else 'negative'} intermediate-term trend."
        
        # Fundamental analysis
        if fundamental_data.get('financial_health_score', 0) > 70:
            fundamental_analysis = f"**Fundamental Analysis**: {ticker} demonstrates strong financial health with a score of {fundamental_data.get('financial_health_score', 0)}/100. The P/E ratio of {fundamental_data.get('pe_ratio', 'N/A')} suggests {'reasonable valuation' if fundamental_data.get('pe_ratio', 100) < 25 else 'premium valuation'}, while the debt-to-equity ratio of {fundamental_data.get('debt_to_equity', 'N/A')} indicates {'conservative' if fundamental_data.get('debt_to_equity', 2) < 1 else 'significant'} leverage. The company's return on equity of {fundamental_data.get('return_on_equity', 'N/A')}% reveals {'excellent' if fundamental_data.get('return_on_equity', 0) > 15 else 'adequate'} profitability relative to shareholder investments."
        else:
            fundamental_analysis = f"**Fundamental Analysis**: {ticker} shows {'moderate' if fundamental_data.get('financial_health_score', 0) > 40 else 'concerning'} financial metrics with a health score of {fundamental_data.get('financial_health_score', 0)}/100. The P/E ratio stands at {fundamental_data.get('pe_ratio', 'N/A')}, which is {'below' if fundamental_data.get('pe_ratio', 0) < 15 and fundamental_data.get('pe_ratio', 0) > 0 else 'above'} industry averages. The debt-to-equity ratio of {fundamental_data.get('debt_to_equity', 'N/A')} suggests {'manageable' if fundamental_data.get('debt_to_equity', 0) < 1.5 else 'elevated'} financial risk, while return on equity at {fundamental_data.get('return_on_equity', 'N/A')}% indicates {'reasonable' if fundamental_data.get('return_on_equity', 0) > 10 else 'suboptimal'} operational efficiency."
        
        # Outlook
        outlook = f"**Outlook**: Based on comprehensive analysis, the forecast for {ticker} appears {forecast_direction} with a target price of ${forecast_price:.2f}, representing a potential {abs(forecast_change):.2f}% {'gain' if forecast_change > 0 else 'loss'}. News sentiment is {sentiment_word} at {sentiment_data.get('score', 0):.2f}, {'supporting' if (sentiment_data.get('score', 0) > 0 and forecast_change > 0) or (sentiment_data.get('score', 0) < 0 and forecast_change < 0) else 'contradicting'} the price forecast. Investors should {'consider accumulating positions' if forecast_change > 10 else 'maintain current positions' if forecast_change > 0 else 'consider reducing exposure'}, while monitoring key resistance at ${current_price * 1.05:.2f} and support at ${current_price * 0.95:.2f}. {'Market volatility may present better entry points in the near term.' if rsi > 60 else 'Current price levels may offer an attractive entry point.' if rsi < 40 else 'Maintain a balanced approach to position sizing given current market conditions.'}"
        
        return f"{technical_analysis}\n\n{fundamental_analysis}\n\n{outlook}"
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def train_ml_models(df: pd.DataFrame, forecast_days: int = 7) -> tuple:
    """Train machine learning models for stock prediction"""
    # Check if we have enough data
    if len(df) < 30:  # Need at least 30 days of data
        # Return empty results if not enough data
        models = {}
        predictions = pd.DataFrame(index=range(forecast_days))
        predictions['day'] = range(1, forecast_days + 1)
        predictions['prediction'] = df['Close'].iloc[-1] if len(df) > 0 else 0
        predictions['mae'] = 0
        predictions['rmse'] = 0
        predictions['r2'] = 0
        return models, predictions
    
    # Prepare features and target
    df_ml = df.copy()
    
    # Add lag features
    for lag in [1, 2, 3, 5, 14, 21]:
        df_ml[f'lag_{lag}'] = df_ml['Close'].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        df_ml[f'rolling_mean_{window}'] = df_ml['Close'].rolling(window=window).mean()
        df_ml[f'rolling_std_{window}'] = df_ml['Close'].rolling(window=window).std()
    
    # Add price momentum
    for period in [1, 3, 7, 14]:
        df_ml[f'momentum_{period}'] = df_ml['Close'].pct_change(periods=period)
    
    # Add volume features
    df_ml['volume_1d_change'] = df_ml['Volume'].pct_change()
    df_ml['volume_ma_ratio'] = df_ml['Volume'] / df_ml['Volume'].rolling(window=10).mean()
    
    # Drop NaN values resulting from lagged features
    df_ml = df_ml.dropna()
    
    # Create target variables for future days
    for day in range(1, forecast_days + 1):
        df_ml[f'target_{day}d'] = df_ml['Close'].shift(-day)
    
    # Drop rows with NaN targets
    df_ml = df_ml.dropna()
    
    # Select features - exclude date and target columns
    feature_columns = [col for col in df_ml.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close'] and not col.startswith('target_')]
    
    # Initialize models and predictions dataframe
    models = {}
    predictions = pd.DataFrame(index=range(forecast_days))
    predictions['day'] = range(1, forecast_days + 1)
    
    # Train models for each forecast day
    for day in range(1, forecast_days + 1):
        target = f'target_{day}d'
        
        # Split data
        X = df_ml[feature_columns]
        y = df_ml[target]
        
        # Check if we have enough data for splitting
        if len(X) < 10:  # Need at least 10 samples
            # Use all data for training if not enough for split
            X_train, y_train = X, y
            X_test, y_test = X.iloc[-1:], y.iloc[-1:]  # Use last row as test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store model and metrics
        models[day] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Make prediction for next period
        latest_data = df_ml[feature_columns].iloc[-1:] 
        prediction = model.predict(latest_data)[0]
        predictions.loc[day-1, 'prediction'] = prediction
        predictions.loc[day-1, 'mae'] = mae
        predictions.loc[day-1, 'rmse'] = rmse
        predictions.loc[day-1, 'r2'] = r2
    
    return models, predictions

# Initialize Streamlit app
st.set_page_config(layout="wide", page_title="StockForecastX Pro")

# App title and description
st.title("StockForecastX Pro: Advanced AI Stock Forecasting")
st.markdown("""
This application combines multiple prediction techniques including technical analysis, 
fundamental analysis, sentiment analysis, and AI-driven forecasting to provide comprehensive 
stock market insights.
""")

# Weather display in the sidebar
st.sidebar.header("Weather Settings")
weather_city = st.sidebar.text_input("Enter City for Weather Info:", "Denver")
weather_info = get_weather(weather_city, OPENWEATHER_API_KEY)
st.sidebar.markdown(f"### Weather Update: {weather_info}", unsafe_allow_html=True)

# Warning notice
if "warning_shown" not in st.session_state:
    st.session_state["warning_shown"] = False

if not st.session_state["warning_shown"]:
    st.warning("""
    **Disclaimer**: This application utilizes advanced AI and machine learning for stock analysis and forecasting. 
    While it provides valuable insights based on historical data and various analytical methods, 
    financial markets are inherently unpredictable. These predictions should be used as one of many tools 
    in your investment research and should not be solely relied upon for financial decisions.
    """)
    st.session_state["warning_shown"] = True

# Main sidebar inputs
st.sidebar.header("Stock Analysis Settings")

# Stock selection with validation
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL").upper().strip()

# Validate ticker input
if ticker and not ticker.isalpha():
    st.sidebar.error("Please enter a valid stock ticker symbol (letters only)")
    ticker = ""

# Analysis timeframe
timeframe_options = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825
}
selected_timeframe = st.sidebar.selectbox("Historical Data Timeframe:", list(timeframe_options.keys()), index=0)
days_back = timeframe_options[selected_timeframe]
# Always use at least 1 year of data
min_days = 365
if days_back < min_days:
    days_back = min_days
start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Forecast period
forecast_period = st.sidebar.slider("Forecast Days:", min_value=1, max_value=60, value=14)

# AI Model selection
st.sidebar.header("AI Settings")
selected_model = st.sidebar.selectbox("AI Model for Analysis:", list(HF_MODELS.keys()), index=0)

# Rate limiting info
st.sidebar.info("""
**Primary**: Alpha Vantage API (more reliable)
Alpha Vantage provides better data quality and reliability.
""")

# API Key Status
st.sidebar.header("API Key Status")
if st.sidebar.button("Check API Keys"):
    st.sidebar.info("Click 'Test Alpha Vantage' below to check your API key status")

# Data source selection
data_source = st.sidebar.selectbox("Data Source:", ["Alpha Vantage (Recommended)"])

# Test connection buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Test Yahoo Finance"):
        st.info("Testing Yahoo Finance...")
        try:
            test_data = yf.download("AAPL", period="1d", progress=False, auto_adjust=True)
            if test_data is not None and not test_data.empty:
                st.success(f"✅ Yahoo Finance working! AAPL: ${test_data['Close'].iloc[-1]:.2f}")
            else:
                st.error("❌ No data returned")
        except Exception as e:
            st.error(f"❌ Yahoo Finance failed: {str(e)}")

with col2:
    if st.button("Test Alpha Vantage"):
        st.info("Testing Alpha Vantage API key...")
        is_valid, message = test_alpha_vantage_api_key()
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
            st.info("💡 You may need to get a new API key from https://www.alphavantage.co/support/#api-key")

# Load and process data button
if st.sidebar.button("Analyze Stock"):
    # Show progress bar for better user experience
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Fetching stock data...")
        progress_bar.progress(10)
        
        # Fetch historical stock data using cached function
        stock_data = cached_stock_data(ticker, start_date, end_date, data_source)
        
        # Check if data was retrieved successfully
        if stock_data is None or stock_data.empty:
            st.error(f"""
            No data found for ticker '{ticker}'. This could be due to:
            - Rate limiting from Yahoo Finance (try waiting a few minutes)
            - Invalid ticker symbol
            - Network connectivity issues
            
            Please try again in a few minutes or check the ticker symbol.
            """)
            st.stop()
        else:
            # Debug: Show data info
            st.info(f"Retrieved {len(stock_data)} days of data for {ticker}")
            st.info(f"Data columns: {list(stock_data.columns)}")
            st.info(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
            # Store data in session state
            st.session_state["stock_data"] = stock_data            
            status_text.text("Calculating technical indicators...")
            progress_bar.progress(25)
            
            # Calculate technical indicators
            try:
                stock_data_with_indicators = calculate_technical_indicators(stock_data)
                st.session_state["stock_data_with_indicators"] = stock_data_with_indicators
                st.info(f"Calculated technical indicators successfully")
            except Exception as e:
                st.error(f"Error calculating technical indicators: {str(e)}")
                st.stop()
            
            status_text.text("Fetching fundamental data...")
            progress_bar.progress(40)
            
            # Get fundamental data using cached function
            try:
                fundamental_data = cached_fundamental_data(ticker, data_source)
                if fundamental_data is None:
                    fundamental_data = get_fundamental_data(ticker, data_source)
                st.session_state["fundamental_data"] = fundamental_data
                st.info(f"Retrieved fundamental data successfully")
            except Exception as e:
                st.error(f"Error fetching fundamental data: {str(e)}")
                fundamental_data = {"name": ticker, "error": str(e), "financial_health_score": 0}
                st.session_state["fundamental_data"] = fundamental_data
            
            status_text.text("Analyzing market sentiment...")
            progress_bar.progress(55)
            
            # Get sentiment data using cached function
            try:
                sentiment_data = cached_sentiment_data(ticker)
                st.session_state["sentiment_data"] = sentiment_data
                st.info(f"Retrieved sentiment data successfully")
            except Exception as e:
                st.error(f"Error fetching sentiment data: {str(e)}")
                sentiment_data = {"score": 0, "articles": [], "error": str(e)}
                st.session_state["sentiment_data"] = sentiment_data
            
            status_text.text("Generating price forecasts...")
            progress_bar.progress(70)
            
            # Prophet forecast with optimized settings
            df_prophet = stock_data.reset_index()[["Date", "Close"]]
            df_prophet.columns = ["ds", "y"]
            
            # Remove any NaN values
            df_prophet = df_prophet.dropna()
            
            try:
                if len(df_prophet) > 30:  # Only run Prophet if we have enough data
                    prophet_model = Prophet(
                        daily_seasonality="auto",  # Use auto instead of False
                        yearly_seasonality="auto",
                        weekly_seasonality="auto",
                        changepoint_prior_scale=0.05,  # More flexible trend
                        seasonality_prior_scale=10.0
                    )
                    prophet_model.fit(df_prophet)
                    future = prophet_model.make_future_dataframe(periods=forecast_period)
                    forecast = prophet_model.predict(future)
                    st.session_state["prophet_forecast"] = forecast
                    st.info(f"Generated Prophet forecast successfully")
                else:
                    st.warning("Insufficient data for Prophet forecasting. Using simple trend projection.")
                    # Create a simple linear forecast
                    dates = pd.date_range(start=df_prophet['ds'].iloc[-1], periods=forecast_period+1, freq='D')[1:]
                    trend = (df_prophet['y'].iloc[-1] - df_prophet['y'].iloc[0]) / len(df_prophet)
                    forecast_values = [df_prophet['y'].iloc[-1] + trend * i for i in range(1, forecast_period+1)]
                    
                    forecast = pd.DataFrame({
                        'ds': dates,
                        'yhat': forecast_values,
                        'yhat_lower': [v * 0.95 for v in forecast_values],
                        'yhat_upper': [v * 1.05 for v in forecast_values]
                    })
                    st.session_state["prophet_forecast"] = forecast
                    st.info(f"Generated simple forecast successfully")
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                # Create a simple fallback forecast
                current_price = stock_data['Close'].iloc[-1]
                dates = pd.date_range(start=stock_data.index[-1], periods=forecast_period+1, freq='D')[1:]
                forecast = pd.DataFrame({
                    'ds': dates,
                    'yhat': [current_price] * forecast_period,
                    'yhat_lower': [current_price * 0.95] * forecast_period,
                    'yhat_upper': [current_price * 1.05] * forecast_period
                })
                st.session_state["prophet_forecast"] = forecast
            
            status_text.text("Training machine learning models...")
            progress_bar.progress(85)
            
            # Train ML models
            try:
                ml_models, ml_predictions = train_ml_models(stock_data_with_indicators, forecast_days=forecast_period)
                st.session_state["ml_models"] = ml_models
                st.session_state["ml_predictions"] = ml_predictions
                st.info(f"Trained ML models successfully")
            except Exception as e:
                st.error(f"Error training ML models: {str(e)}")
                # Create fallback ML predictions
                current_price = stock_data['Close'].iloc[-1]
                ml_predictions = pd.DataFrame({
                    'day': range(1, forecast_period + 1),
                    'prediction': [current_price] * forecast_period,
                    'mae': [0] * forecast_period,
                    'rmse': [0] * forecast_period,
                    'r2': [0] * forecast_period
                })
                st.session_state["ml_models"] = {}
                st.session_state["ml_predictions"] = ml_predictions
            
            status_text.text("Generating AI analysis...")
            progress_bar.progress(95)
            
            # Get AI analysis with selected model
            try:
                if sentiment_data is None:
                    sentiment_data = {"score": 0, "articles": [], "error": "No sentiment data available"}
                ai_analysis = get_ai_analysis(ticker, stock_data_with_indicators, forecast, fundamental_data, sentiment_data, selected_model)
                st.session_state["ai_analysis"] = ai_analysis
                st.info(f"Generated AI analysis successfully")
            except Exception as e:
                st.error(f"Error generating AI analysis: {str(e)}")
                ai_analysis = f"AI analysis unavailable due to error: {str(e)}"
                st.session_state["ai_analysis"] = ai_analysis
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            st.success(f"Successfully analyzed {ticker}!")
            
    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
# Display analysis if data is available
if "stock_data" in st.session_state:
    # Create tabs for different analysis views (always create them)
    tabs = st.tabs(["Dashboard", "Technical Analysis", "Fundamental Analysis", "Sentiment Analysis", "ML Forecasts", "AI Insights"])
    
    # Check if all required session state variables exist
    required_vars = [
        "stock_data", "stock_data_with_indicators", "fundamental_data", 
        "sentiment_data", "prophet_forecast", "ml_predictions", "ai_analysis"
    ]
    
    missing_vars = [var for var in required_vars if var not in st.session_state]
    
    if missing_vars:
        # Show error in first tab
        with tabs[0]:
            st.error(f"Analysis incomplete. Missing data: {', '.join(missing_vars)}")
            st.info("Please run the analysis again by clicking 'Analyze Stock'")
        
        # Add placeholder content for other tabs
        with tabs[1]:
            st.info("Technical Analysis will be available once the analysis is complete.")
        
        with tabs[2]:
            st.info("Fundamental Analysis will be available once the analysis is complete.")
        
        with tabs[3]:
            st.info("Sentiment Analysis will be available once the analysis is complete.")
        
        with tabs[4]:
            st.info("ML Forecasts will be available once the analysis is complete.")
        
        with tabs[5]:
            st.info("AI Insights will be available once the analysis is complete.")
    else:
        # All data is available, proceed with analysis
        stock_data = st.session_state["stock_data"]
        stock_data_with_indicators = st.session_state["stock_data_with_indicators"]
        fundamental_data = st.session_state["fundamental_data"]
        sentiment_data = st.session_state["sentiment_data"]
        prophet_forecast = st.session_state["prophet_forecast"]
        ml_predictions = st.session_state["ml_predictions"]
        ai_analysis = st.session_state["ai_analysis"]
        
        # ===== DASHBOARD TAB =====
        with tabs[0]:
            # Two-column layout for main metrics
            col1, col2 = st.columns(2)
        
        # Check if stock_data and prophet_forecast exist and are not None
        if 'stock_data' not in st.session_state or st.session_state["stock_data"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        if 'prophet_forecast' not in st.session_state or st.session_state["prophet_forecast"] is None:
            st.error("No forecast data available. Please run the analysis first.")
            st.stop()
        
        stock_data = st.session_state["stock_data"]
        prophet_forecast = st.session_state["prophet_forecast"]
        
        # Current price and forecast
        with col1:
            current_price = stock_data["Close"].iloc[-1]
            forecast_price = prophet_forecast["yhat"].iloc[-1]
            forecast_change = ((forecast_price / current_price) - 1) * 100
            
            st.metric(
                label=f"{ticker} Current Price", 
                value=f"${current_price:.2f}",
                delta=f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-2]) - 1) * 100:.2f}%"
            )
            
            st.metric(
                label=f"Forecast ({forecast_period} days)", 
                value=f"${forecast_price:.2f}",
                delta=f"{forecast_change:.2f}%"
            )
            
        # Check if fundamental_data exists and is not None
        if 'fundamental_data' not in st.session_state or st.session_state["fundamental_data"] is None:
            st.error("No fundamental data available. Please run the analysis first.")
            st.stop()
        
        fundamental_data = st.session_state["fundamental_data"]
        
        # Display company name and sector
        st.subheader(f"{fundamental_data.get('name', ticker)}")
        st.text(f"Sector: {fundamental_data.get('sector', 'N/A')} | Industry: {fundamental_data.get('industry', 'N/A')}")
        
        # Health score
        with col2:
            # Financial health score gauge
            health_score = fundamental_data.get('financial_health_score', 0)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "Financial Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': health_score
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Check if sentiment_data exists and is not None
            if 'sentiment_data' not in st.session_state or st.session_state["sentiment_data"] is None:
                st.error("No sentiment data available. Please run the analysis first.")
                st.stop()
            
            sentiment_data = st.session_state["sentiment_data"]
            
            # News sentiment score gauge
            sentiment_score = sentiment_data.get('score', 0) 
            sentiment_score_normalized = (sentiment_score + 1) * 50  # Convert -1 to 1 scale to 0 to 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score_normalized,
                title={'text': "News Sentiment Score"},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score_normalized
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Check if prophet_forecast exists and is not None
        if 'prophet_forecast' not in st.session_state or st.session_state["prophet_forecast"] is None:
            st.error("No forecast data available. Please run the analysis first.")
            st.stop()
        
        prophet_forecast = st.session_state["prophet_forecast"]
        
        # Main price chart
        st.subheader("Price History and Forecast")
        
        # Create forecast dataframe that matches stock_data format
        forecast_df = prophet_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date"})
        forecast_df = forecast_df.set_index("Date")
        # Check if stock_data exists and is not None
        if 'stock_data' not in st.session_state or st.session_state["stock_data"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        stock_data = st.session_state["stock_data"]
        forecast_df = forecast_df.loc[forecast_df.index > stock_data.index[-1]]
        
        # Create combined figure
        fig = go.Figure()
        
        # Add historical close price
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data["Close"],
            mode="lines",
            name="Historical Price",
            line=dict(color="blue", width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="orange", width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df["yhat_upper"].tolist() + forecast_df["yhat_lower"].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,165,0,0)'),
            name="Confidence Interval"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Price History and Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("Trading Volume")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data["Volume"],
            name="Volume",
            marker_color="darkblue"
        ))
        
        fig.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            margin=dict(l=20, r=20, t=50, b=20),
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if ai_analysis exists and is not None
        if 'ai_analysis' not in st.session_state or st.session_state["ai_analysis"] is None:
            st.error("No AI analysis available. Please run the analysis first.")
            st.stop()
        
        ai_analysis = st.session_state["ai_analysis"]
        
        # AI Analysis
        st.subheader("AI Analysis")
        st.markdown(ai_analysis)
        
        # Check if fundamental_data and sentiment_data exist and are not None
        if 'fundamental_data' not in st.session_state or st.session_state["fundamental_data"] is None:
            st.error("No fundamental data available. Please run the analysis first.")
            st.stop()
        
        if 'sentiment_data' not in st.session_state or st.session_state["sentiment_data"] is None:
            st.error("No sentiment data available. Please run the analysis first.")
            st.stop()
        
        fundamental_data = st.session_state["fundamental_data"]
        sentiment_data = st.session_state["sentiment_data"]
        
        # Key metrics in columns
        st.subheader("Key Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("P/E Ratio", f"{fundamental_data.get('pe_ratio', 'N/A')}")
            st.metric("Dividend Yield", f"{fundamental_data.get('dividend_yield', 0):.2f}%" if fundamental_data.get('dividend_yield') else "N/A")
        
        with metrics_col2:
            st.metric("Market Cap", f"${fundamental_data.get('market_cap', 0)/1e9:.2f}B" if fundamental_data.get('market_cap') else "N/A")
            st.metric("Beta", f"{fundamental_data.get('beta', 'N/A')}")
        
        with metrics_col3:
            st.metric("52 Week High", f"${fundamental_data.get('52w_high', 'N/A')}")
            st.metric("52 Week Low", f"${fundamental_data.get('52w_low', 'N/A')}")
        
        with metrics_col4:
            last_rsi = stock_data_with_indicators["RSI"].iloc[-1] if "RSI" in stock_data_with_indicators else None
            last_macd = stock_data_with_indicators["MACD_Line"].iloc[-1] if "MACD_Line" in stock_data_with_indicators else None
            
            st.metric("RSI (14)", f"{last_rsi:.2f}" if last_rsi is not None else "N/A")
            st.metric("MACD", f"{last_macd:.4f}" if last_macd is not None else "N/A")
        
        # Recent news
        st.subheader("Recent News")
        
        news_items = sentiment_data.get('articles', [])
        if not news_items:
            st.info("No recent news available.")
        else:
            for i, news in enumerate(news_items):
                sentiment_score = news.get('sentiment', 0)
                sentiment_color = "green" if sentiment_score > 0.2 else "red" if sentiment_score < -0.2 else "orange"
                
                st.markdown(f"""
                **{news.get('title', 'No title')}**  
                *{news.get('time_published', '')}*  
                Sentiment: <span style='color:{sentiment_color}'>{sentiment_score:.2f}</span>
                """, unsafe_allow_html=True)
                
                if i < len(news_items) - 1:
                    st.markdown("---")
        
        # Download data button
        st.subheader("Download Analysis Data")
        
        # Check if stock_data_with_indicators exists and is not None
        if 'stock_data_with_indicators' not in st.session_state or st.session_state["stock_data_with_indicators"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        stock_data_with_indicators = st.session_state["stock_data_with_indicators"]
        
        # Check if prophet_forecast exists and is not None
        if 'prophet_forecast' not in st.session_state or st.session_state["prophet_forecast"] is None:
            st.error("No forecast data available. Please run the analysis first.")
            st.stop()
        
        prophet_forecast = st.session_state["prophet_forecast"]
        forecast_df = prophet_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date"})
        forecast_df = forecast_df.set_index("Date")
        # Check if stock_data exists and is not None
        if 'stock_data' not in st.session_state or st.session_state["stock_data"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        stock_data = st.session_state["stock_data"]
        forecast_df = forecast_df.loc[forecast_df.index > stock_data.index[-1]]
        
        # Prepare download data
        data_for_download = stock_data_with_indicators.copy()
        data_for_download["Forecast"] = None
        for i, row in forecast_df.iterrows():
            if i in data_for_download.index:
                data_for_download.loc[i, "Forecast"] = row["yhat"]
            else:
                new_row = pd.Series([None] * len(data_for_download.columns), index=data_for_download.columns)
                new_row["Forecast"] = row["yhat"]
                data_for_download.loc[i] = new_row
        
        # Create download button
        csv = data_for_download.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_analysis.csv" class="btn">Download Complete Analysis (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # ===== TECHNICAL ANALYSIS TAB =====
    with tabs[1]:
        st.subheader(f"Technical Analysis for {ticker}")
        
        # Technical indicators selection
        tech_indicators = st.multiselect(
            "Select Technical Indicators",
            ["SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21", "BB_Upper", "BB_Lower", "VWAP"],
            default=["SMA_50", "EMA_21", "BB_Upper", "BB_Lower"]
        )
        
        # Moving averages and price chart
        st.subheader("Price and Selected Indicators")
        
        # Check if stock_data_with_indicators exists and is not None
        if 'stock_data_with_indicators' not in st.session_state or st.session_state["stock_data_with_indicators"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        stock_data_with_indicators = st.session_state["stock_data_with_indicators"]
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data_with_indicators.index,
            open=stock_data_with_indicators['Open'],
            high=stock_data_with_indicators['High'],
            low=stock_data_with_indicators['Low'],
            close=stock_data_with_indicators['Close'],
            name="Price"
        ))
        
        # Add selected indicators
        colors = ["orange", "green", "red", "purple", "brown", "gray", "gray", "blue"]
        for indicator, color in zip(tech_indicators, colors):
            if indicator in stock_data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=stock_data_with_indicators.index,
                    y=stock_data_with_indicators[indicator],
                    mode="lines",
                    name=indicator,
                    line=dict(color=color)
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Oscillators
        st.subheader("Technical Oscillators")
        
        # Create figure with subplots
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=("RSI", "MACD", "Stochastic Oscillator"),
                           vertical_spacing=0.1,
                           row_heights=[0.33, 0.33, 0.33])
        
        # Add RSI
        if "RSI" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=stock_data_with_indicators.index, y=stock_data_with_indicators["RSI"],
                          name="RSI", line=dict(color="blue")),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=stock_data_with_indicators.index, y=[70] * len(stock_data_with_indicators),
                      name="Overbought", line=dict(color="red", dash="dash")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=stock_data_with_indicators.index, y=[30] * len(stock_data_with_indicators),
                      name="Oversold", line=dict(color="green", dash="dash")),
            row=1, col=1
        )
        
        # Add MACD
        if "MACD_Line" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=stock_data_with_indicators.index, y=stock_data_with_indicators["MACD_Line"],
                          name="MACD Line", line=dict(color="blue")),
                row=2, col=1
            )
        
        if "MACD_Signal" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=stock_data_with_indicators.index, y=stock_data_with_indicators["MACD_Signal"],
                          name="Signal Line", line=dict(color="orange")),
                row=2, col=1
            )
        
        if "MACD_Histogram" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Bar(x=stock_data_with_indicators.index, y=stock_data_with_indicators["MACD_Histogram"],
                       name="Histogram", marker_color="gray"),
                row=2, col=1
            )
        
        # Add Stochastic Oscillator
        if "%K" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=stock_data_with_indicators.index, y=stock_data_with_indicators["%K"],
                          name="%K", line=dict(color="blue")),
                row=3, col=1
            )
        
        if "%D" in stock_data_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=stock_data_with_indicators.index, y=stock_data_with_indicators["%D"],
                      name="%D", line=dict(color="red")),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=stock_data_with_indicators.index, y=[80] * len(stock_data_with_indicators),
                      name="Overbought", line=dict(color="red", dash="dash"), showlegend=False),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=stock_data_with_indicators.index, y=[20] * len(stock_data_with_indicators),
                      name="Oversold", line=dict(color="green", dash="dash"), showlegend=False),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="%K/%D", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis signals
        st.subheader("Technical Signals")
        
        # Get latest values
        latest = stock_data_with_indicators.iloc[-1]
        
        # Create signals
        signals = []
        
        # Moving Average Signals
        if "SMA_50" in stock_data_with_indicators.columns and "SMA_200" in stock_data_with_indicators.columns:
            if latest["SMA_50"] > latest["SMA_200"]:
                signals.append(("BULLISH", "Golden Cross (SMA 50 > SMA 200)"))
            else:
                signals.append(("BEARISH", "Death Cross (SMA 50 < SMA 200)"))
        
        # RSI Signals
        if "RSI" in stock_data_with_indicators.columns:
            if latest["RSI"] > 70:
                signals.append(("BEARISH", f"Overbought (RSI = {latest['RSI']:.2f})"))
            elif latest["RSI"] < 30:
                signals.append(("BULLISH", f"Oversold (RSI = {latest['RSI']:.2f})"))
            else:
                signals.append(("NEUTRAL", f"RSI in normal range ({latest['RSI']:.2f})"))
        
        # MACD Signals
        if "MACD_Line" in stock_data_with_indicators.columns and "MACD_Signal" in stock_data_with_indicators.columns:
            prev_macd = stock_data_with_indicators["MACD_Line"].iloc[-2]
            prev_signal = stock_data_with_indicators["MACD_Signal"].iloc[-2]
            
            if latest["MACD_Line"] > latest["MACD_Signal"] and prev_macd <= prev_signal:
                signals.append(("BULLISH", "MACD Bullish Crossover"))
            elif latest["MACD_Line"] < latest["MACD_Signal"] and prev_macd >= prev_signal:
                signals.append(("BEARISH", "MACD Bearish Crossover"))
        
        # Bollinger Bands Signals
        if "BB_Upper" in stock_data_with_indicators.columns and "BB_Lower" in stock_data_with_indicators.columns:
            if latest["Close"] > latest["BB_Upper"]:
                signals.append(("BEARISH", "Price above upper Bollinger Band (Possibly overbought)"))
            elif latest["Close"] < latest["BB_Lower"]:
                signals.append(("BULLISH", "Price below lower Bollinger Band (Possibly oversold)"))
        
        # Display signals in a table
        signal_df = pd.DataFrame(signals, columns=["Signal", "Description"])
        
        # Apply styling to signals
        def color_signal(val):
            if val == "BULLISH":
                return 'background-color: #c6ecc6'
            elif val == "BEARISH":
                return 'background-color: #ffc6c6'
            else:
                return 'background-color: #f0f0f0'
        

        
        styled_signal_df = signal_df.style.map(color_signal, subset=["Signal"])
        st.table(styled_signal_df)
        
        # Support and Resistance levels
        st.subheader("Support and Resistance Levels")
        
        # Calculate pivot points
        high = stock_data_with_indicators["High"].iloc[-1]
        low = stock_data_with_indicators["Low"].iloc[-1]
        close = stock_data_with_indicators["Close"].iloc[-1]
        
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        
        # Display pivot points
        pivot_col1, pivot_col2 = st.columns(2)
        
        with pivot_col1:
            st.metric("Resistance 2", f"${r2:.2f}")
            st.metric("Resistance 1", f"${r1:.2f}")
            st.metric("Pivot Point", f"${pivot:.2f}")
        
        with pivot_col2:
            st.metric("Support 1", f"${s1:.2f}")
            st.metric("Support 2", f"${s2:.2f}")
            st.metric("Current Price", f"${close:.2f}")
    
    # ===== FUNDAMENTAL ANALYSIS TAB =====
    with tabs[2]:
        st.subheader(f"Fundamental Analysis for {ticker}")
        
        # Check if fundamental_data exists and is not None
        if 'fundamental_data' not in st.session_state or st.session_state["fundamental_data"] is None:
            st.error("No fundamental data available. Please run the analysis first.")
            st.stop()
        
        fundamental_data = st.session_state["fundamental_data"]
        
        # Company information
        company_info = f"""
        **{fundamental_data.get('name', ticker)}**  
        *{fundamental_data.get('sector', 'N/A')} | {fundamental_data.get('industry', 'N/A')}*
        
        **Market Cap**: ${fundamental_data.get('market_cap', 0)/1e9:.2f}B
        """
        st.markdown(company_info)
        
        # Create metrics row
        fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        
        with fund_col1:
            st.metric("P/E Ratio", fundamental_data.get('pe_ratio', 'N/A'))
            st.metric("Forward P/E", fundamental_data.get('forward_pe', 'N/A'))
            st.metric("PEG Ratio", fundamental_data.get('peg_ratio', 'N/A'))
        
        with fund_col2:
            st.metric("Price to Book", fundamental_data.get('price_to_book', 'N/A'))
            st.metric("EPS", f"${fundamental_data.get('eps', 'N/A')}")
            st.metric("Beta", fundamental_data.get('beta', 'N/A'))
        
        with fund_col3:
            st.metric("Dividend Yield", f"{fundamental_data.get('dividend_yield', 0):.2f}%" if fundamental_data.get('dividend_yield') else "N/A")
            st.metric("Debt to Equity", fundamental_data.get('debt_to_equity', 'N/A'))
            st.metric("Return on Equity", f"{fundamental_data.get('return_on_equity', 'N/A')}%")
        
        with fund_col4:
            st.metric("Profit Margins", f"{fundamental_data.get('profit_margins', 'N/A')}%")
            st.metric("Revenue Growth", f"{fundamental_data.get('revenue_growth', 'N/A')}%")
            st.metric("52-Week Range", f"${fundamental_data.get('52w_low', 'N/A')} - ${fundamental_data.get('52w_high', 'N/A')}")
        
        # Valuation chart
        st.subheader("Valuation Score Breakdown")
        
        # Create a radar chart for valuation scores
        # Convert metrics to scores between 0-10
        pe_ratio = fundamental_data.get('pe_ratio')
        price_to_book = fundamental_data.get('price_to_book')
        peg_ratio = fundamental_data.get('peg_ratio')
        debt_to_equity = fundamental_data.get('debt_to_equity')
        profit_margins = fundamental_data.get('profit_margins')
        
        pe_score = 7 if pe_ratio and 0 < pe_ratio < 20 else 5 if pe_ratio and 20 <= pe_ratio < 50 else 3
        pb_score = 7 if price_to_book and 0 < price_to_book < 3 else 5 if price_to_book and 3 <= price_to_book < 7 else 3
        peg_score = 8 if peg_ratio and 0 < peg_ratio < 1 else 6 if peg_ratio and 1 <= peg_ratio < 2 else 4
        de_score = 8 if debt_to_equity and 0 < debt_to_equity < 0.5 else 6 if debt_to_equity and 0.5 <= debt_to_equity < 1.5 else 4
        profit_score = 8 if profit_margins and profit_margins > 15 else 6 if profit_margins and 5 <= profit_margins < 15 else 4
        
        # Radar chart
        categories = ['P/E Ratio', 'Price to Book', 'PEG Ratio', 'Debt to Equity', 'Profit Margins']
        values = [pe_score, pb_score, peg_score, de_score, profit_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Valuation Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of scores
        st.info("""
        **Interpretation of Scores:**
        * **P/E Ratio**: Lower is typically better; scores well if below 20.
        * **Price to Book**: Lower is typically better; scores well if below 3.
        * **PEG Ratio**: Lower is better; scores well if below 1.
        * **Debt to Equity**: Lower is better; scores well if below 0.5.
        * **Profit Margins**: Higher is better; scores well if above 15%.
        """)
        
        # Financial health components
        st.subheader("Financial Health Components")
        
        # Simulate financial health components
        profit_margins = fundamental_data.get('profit_margins')
        debt_to_equity = fundamental_data.get('debt_to_equity')
        pe_ratio = fundamental_data.get('pe_ratio')
        revenue_growth = fundamental_data.get('revenue_growth')
        return_on_equity = fundamental_data.get('return_on_equity')
        
        health_components = {
            "Profitability": int(profit_margins * 5) if profit_margins else 50,
            "Capital Structure": 80 if debt_to_equity and debt_to_equity < 1 else 50,
            "Valuation": 75 if pe_ratio and pe_ratio < 25 else 50,
            "Growth": int(revenue_growth * 5) if revenue_growth else 50,
            "Efficiency": int(return_on_equity * 5) if return_on_equity else 50
        }
        
        # Create a horizontal bar chart
        fig = go.Figure()
        
        for component, score in health_components.items():
            fig.add_trace(go.Bar(
                y=[component],
                x=[score],
                orientation='h',
                marker_color='blue'
            ))
        
        fig.update_layout(
            title="Financial Health Score Components",
            xaxis_title="Score (0-100)",
            yaxis_title="Component",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Investment checklist
        st.subheader("Investment Checklist")
        
        # Create a checklist based on fundamental data
        checklist = []
        
        # P/E Ratio Check
        if fundamental_data.get('pe_ratio') and 0 < fundamental_data.get('pe_ratio') < 25:
            checklist.append(("PASS", "P/E Ratio below 25"))
        else:
            checklist.append(("FAIL", "P/E Ratio below 25"))
        
        # PEG Ratio Check
        if fundamental_data.get('peg_ratio') and 0 < fundamental_data.get('peg_ratio') < 1.5:
            checklist.append(("PASS", "PEG Ratio below 1.5"))
        else:
            checklist.append(("FAIL", "PEG Ratio below 1.5"))
        
        # Debt to Equity Check
        if fundamental_data.get('debt_to_equity') and fundamental_data.get('debt_to_equity') < 1:
            checklist.append(("PASS", "Debt to Equity below 1"))
        else:
            checklist.append(("FAIL", "Debt to Equity below 1"))
        
        # ROE Check
        if fundamental_data.get('return_on_equity') and fundamental_data.get('return_on_equity') > 10:
            checklist.append(("PASS", "Return on Equity above 10%"))
        else:
            checklist.append(("FAIL", "Return on Equity above 10%"))
        
        # Profit Margins Check
        if fundamental_data.get('profit_margins') and fundamental_data.get('profit_margins') > 10:
            checklist.append(("PASS", "Profit Margins above 10%"))
        else:
            checklist.append(("FAIL", "Profit Margins above 10%"))
        
        # Revenue Growth Check
        if fundamental_data.get('revenue_growth') and fundamental_data.get('revenue_growth') > 5:
            checklist.append(("PASS", "Revenue Growth above 5%"))
        else:
            checklist.append(("FAIL", "Revenue Growth above 5%"))
        
        # Display checklist
        checklist_df = pd.DataFrame(checklist, columns=["Status", "Criteria"])
        
        # Apply styling manually
        def style_checklist(val):
            if val == "PASS":
                return 'background-color: #c6ecc6'
            elif val == "FAIL":
                return 'background-color: #ffc6c6'
            else:
                return 'background-color: #f0f0f0'
        
        styled_checklist_df = checklist_df.style.apply(lambda x: [style_checklist(v) for v in x], subset=["Status"])
        st.table(styled_checklist_df)
    
    # ===== SENTIMENT ANALYSIS TAB =====
    with tabs[3]:
        st.subheader(f"Sentiment Analysis for {ticker}")
        
        # Check if sentiment_data exists and is not None
        if 'sentiment_data' not in st.session_state or st.session_state["sentiment_data"] is None:
            st.error("No sentiment data available. Please run the analysis first.")
            st.stop()
        
        sentiment_data = st.session_state["sentiment_data"]
        
        # Overall sentiment score
        sentiment_score = sentiment_data.get('score', 0)
        sentiment_color = "green" if sentiment_score > 0.2 else "red" if sentiment_score < -0.2 else "orange"
        
        st.markdown(f"""
        ### Overall Sentiment Score: <span style='color:{sentiment_color}'>{sentiment_score:.2f}</span>
        *(-1 = Very Negative, 0 = Neutral, 1 = Very Positive)*
        """, unsafe_allow_html=True)
        
        # Sentiment gauge
        sentiment_score_normalized = (sentiment_score + 1) * 50  # Convert -1 to 1 scale to 0 to 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score_normalized,
            title={'text': "News Sentiment"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score_normalized
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent news articles
        st.subheader("Recent News Articles")
        
        news_items = sentiment_data.get('articles', [])
        if not news_items:
            st.info("No recent news available for this stock.")
        else:
            for i, news in enumerate(news_items):
                sentiment_score = news.get('sentiment', 0)
                sentiment_color = "green" if sentiment_score > 0.2 else "red" if sentiment_score < -0.2 else "orange"
                sentiment_text = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
                
                st.markdown(f"""
                **{news.get('title', 'No title')}**  
                *Published: {news.get('time_published', 'Unknown')}*  
                **Sentiment**: <span style='color:{sentiment_color}'>{sentiment_text} ({sentiment_score:.2f})</span>
                """, unsafe_allow_html=True)
                
                if news.get('url'):
                    st.markdown(f"[Read Article]({news.get('url')})")
                
                if i < len(news_items) - 1:
                    st.markdown("---")
        
        # Sentiment trends (if we had historical data)
        st.subheader("Sentiment Analysis Insights")
        
        if sentiment_score > 0.3:
            st.success("**Positive Sentiment Detected**: Recent news coverage is generally positive, which may support upward price movement.")
        elif sentiment_score < -0.3:
            st.error("**Negative Sentiment Detected**: Recent news coverage is generally negative, which may pressure the stock price.")
        else:
            st.info("**Neutral Sentiment**: Recent news coverage is balanced, with no strong directional bias.")
        
        # Sentiment impact on price
        st.markdown("""
        **How Sentiment Affects Stock Price:**
        - **Positive sentiment** often correlates with increased buying pressure
        - **Negative sentiment** can lead to selling pressure
        - **Neutral sentiment** typically indicates stable trading conditions
        - Sentiment changes can be leading indicators of price movements
        """)
    
    # ===== ML FORECASTS TAB =====
    with tabs[4]:
        st.subheader(f"Machine Learning Forecasts for {ticker}")
        
        # Check if ml_predictions exists and is not None
        if 'ml_predictions' not in st.session_state or st.session_state["ml_predictions"] is None:
            st.error("No ML predictions available. Please run the analysis first.")
            st.stop()
        
        ml_predictions = st.session_state["ml_predictions"]
        
        # ML Model Performance
        st.subheader("Model Performance Metrics")
        
        # Create performance comparison
        performance_data = []
        for day in range(1, min(8, len(ml_predictions) + 1)):
            if day in ml_predictions.index:
                performance_data.append({
                    'Day': f"Day {day}",
                    'MAE': ml_predictions.loc[day-1, 'mae'],
                    'RMSE': ml_predictions.loc[day-1, 'rmse'],
                    'R²': ml_predictions.loc[day-1, 'r2']
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Display metrics in columns
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                avg_mae = perf_df['MAE'].mean()
                st.metric("Average MAE", f"${avg_mae:.2f}")
            
            with perf_col2:
                avg_rmse = perf_df['RMSE'].mean()
                st.metric("Average RMSE", f"${avg_rmse:.2f}")
            
            with perf_col3:
                avg_r2 = perf_df['R²'].mean()
                st.metric("Average R²", f"{avg_r2:.3f}")
            
            # Performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=perf_df['Day'],
                y=perf_df['MAE'],
                mode='lines+markers',
                name='MAE',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=perf_df['Day'],
                y=perf_df['RMSE'],
                mode='lines+markers',
                name='RMSE',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Model Performance by Forecast Day",
                xaxis_title="Forecast Day",
                yaxis_title="Error ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ML Predictions
        st.subheader("Machine Learning Predictions")
        
        if not ml_predictions.empty:
            # Check if stock_data exists and is not None
            if 'stock_data' not in st.session_state or st.session_state["stock_data"] is None:
                st.error("No stock data available. Please run the analysis first.")
                st.stop()
            
            stock_data = st.session_state["stock_data"]
            
            # Create prediction chart
            current_price = stock_data['Close'].iloc[-1]
            
            fig = go.Figure()
            
            # Add historical price
            fig.add_trace(go.Scatter(
                x=stock_data.index[-30:],  # Last 30 days
                y=stock_data['Close'][-30:],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add ML predictions
            prediction_days = list(range(1, len(ml_predictions) + 1))
            prediction_prices = ml_predictions['prediction'].values
            
            fig.add_trace(go.Scatter(
                x=prediction_days,
                y=prediction_prices,
                mode='lines+markers',
                name='ML Predictions',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="ML Model Price Predictions",
                xaxis_title="Days Ahead",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction table
            st.subheader("Detailed Predictions")
            
            prediction_table = ml_predictions.copy()
            prediction_table['Current Price'] = current_price
            prediction_table['Predicted Change'] = ((prediction_table['prediction'] / current_price) - 1) * 100
            prediction_table['Predicted Change'] = prediction_table['Predicted Change'].round(2)
            
            # Format for display
            display_table = prediction_table[['day', 'prediction', 'Current Price', 'Predicted Change', 'mae', 'rmse', 'r2']].copy()
            display_table.columns = ['Day', 'Predicted Price', 'Current Price', 'Change (%)', 'MAE', 'RMSE', 'R²']
            display_table['Predicted Price'] = display_table['Predicted Price'].round(2)
            display_table['MAE'] = display_table['MAE'].round(2)
            display_table['RMSE'] = display_table['RMSE'].round(2)
            display_table['R²'] = display_table['R²'].round(3)
            
            st.table(display_table)
            
            # Model insights
            st.subheader("ML Model Insights")
            
            # Find best and worst predictions
            best_day = prediction_table.loc[prediction_table['r2'].idxmax()]
            worst_day = prediction_table.loc[prediction_table['r2'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Best Model Performance",
                    f"Day {int(best_day['day'])}",
                    f"R² = {best_day['r2']:.3f}"
                )
            
            with col2:
                st.metric(
                    "Worst Model Performance", 
                    f"Day {int(worst_day['day'])}",
                    f"R² = {worst_day['r2']:.3f}"
                )
            
            # Overall prediction trend
            avg_prediction = prediction_table['prediction'].mean()
            overall_change = ((avg_prediction / current_price) - 1) * 100
            
            st.metric(
                "Average Prediction",
                f"${avg_prediction:.2f}",
                f"{overall_change:.2f}%"
            )
            
            # Model confidence
            avg_r2 = prediction_table['r2'].mean()
            if avg_r2 > 0.7:
                confidence = "High"
                confidence_color = "green"
            elif avg_r2 > 0.5:
                confidence = "Medium"
                confidence_color = "orange"
            else:
                confidence = "Low"
                confidence_color = "red"
            
            st.markdown(f"""
            **Model Confidence**: <span style='color:{confidence_color}'>{confidence}</span>  
            *Based on average R² score of {avg_r2:.3f}*
            """, unsafe_allow_html=True)
        
        else:
            st.warning("No ML predictions available. Please run the analysis again.")
    
    # ===== AI INSIGHTS TAB =====
    with tabs[5]:
        st.subheader(f"AI-Powered Analysis for {ticker}")
        
        # Check if ai_analysis exists and is not None
        if 'ai_analysis' not in st.session_state or st.session_state["ai_analysis"] is None:
            st.error("No AI analysis available. Please run the analysis first.")
            st.stop()
        
        ai_analysis = st.session_state["ai_analysis"]
        
        # Display the AI analysis
        st.markdown("### Comprehensive AI Analysis")
        st.markdown(ai_analysis)
        
        # AI Analysis Breakdown
        st.subheader("Analysis Components")
        
        # Check if prophet_forecast exists and is not None
        if 'prophet_forecast' not in st.session_state or st.session_state["prophet_forecast"] is None:
            st.error("No forecast data available. Please run the analysis first.")
            st.stop()
        
        prophet_forecast = st.session_state["prophet_forecast"]
        
        # Investment recommendation
        st.subheader("AI Investment Recommendation")
        
        # Check if stock_data exists and is not None
        if 'stock_data' not in st.session_state or st.session_state["stock_data"] is None:
            st.error("No stock data available. Please run the analysis first.")
            st.stop()
        
        stock_data = st.session_state["stock_data"]
        
        # Create a breakdown of the analysis factors
        analysis_factors = {
            "Technical Indicators": "RSI, MACD, Moving Averages, Bollinger Bands",
            "Fundamental Metrics": "P/E Ratio, Debt-to-Equity, ROE, Profit Margins",
            "Market Sentiment": f"News Sentiment Score: {sentiment_data.get('score', 0):.2f}",
            "Price Forecast": f"Predicted Change: {((prophet_forecast['yhat'].iloc[-1] / stock_data['Close'].iloc[-1]) - 1) * 100:.2f}%",
            "Risk Assessment": "Based on volatility and financial health"
        }
        
        for factor, description in analysis_factors.items():
            st.markdown(f"**{factor}**: {description}")
        
        # Calculate recommendation score
        current_price = stock_data['Close'].iloc[-1]
        forecast_price = prophet_forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_price / current_price) - 1) * 100
        
        # Check if stock_data_with_indicators exists and is not None
        if 'stock_data_with_indicators' not in st.session_state or st.session_state["stock_data_with_indicators"] is None:
            st.error("No technical indicators available. Please run the analysis first.")
            st.stop()
        
        stock_data_with_indicators = st.session_state["stock_data_with_indicators"]
        
        # Technical score
        latest_data = stock_data_with_indicators.iloc[-1]
        rsi = latest_data.get('RSI', 50)
        technical_score = 0
        if 30 <= rsi <= 70:
            technical_score += 25
        elif rsi < 30:
            technical_score += 35  # Oversold - potential buy
        else:
            technical_score += 15  # Overbought - potential sell
        
        # Check if fundamental_data exists and is not None
        if 'fundamental_data' not in st.session_state or st.session_state["fundamental_data"] is None:
            st.error("No fundamental data available. Please run the analysis first.")
            st.stop()
        
        fundamental_data = st.session_state["fundamental_data"]
        
        # Fundamental score
        fundamental_score = fundamental_data.get('financial_health_score', 50)
        
        # Check if sentiment_data exists and is not None
        if 'sentiment_data' not in st.session_state or st.session_state["sentiment_data"] is None:
            st.error("No sentiment data available. Please run the analysis first.")
            st.stop()
        
        sentiment_data = st.session_state["sentiment_data"]
        
        # Sentiment score
        sentiment_score = sentiment_data.get('score', 0)
        sentiment_score_normalized = (sentiment_score + 1) * 25  # Convert to 0-50 scale
        
        # Forecast score
        if forecast_change > 10:
            forecast_score = 25
        elif forecast_change > 5:
            forecast_score = 20
        elif forecast_change > 0:
            forecast_score = 15
        elif forecast_change > -5:
            forecast_score = 10
        else:
            forecast_score = 5
        
        # Total recommendation score
        total_score = technical_score + fundamental_score + sentiment_score_normalized + forecast_score
        
        # Determine recommendation
        if total_score >= 80:
            recommendation = "STRONG BUY"
            recommendation_color = "green"
            recommendation_icon = "ROCKET"
        elif total_score >= 65:
            recommendation = "BUY"
            recommendation_color = "lightgreen"
            recommendation_icon = "TRENDING UP"
        elif total_score >= 50:
            recommendation = "HOLD"
            recommendation_color = "orange"
            recommendation_icon = "PAUSE"
        elif total_score >= 35:
            recommendation = "SELL"
            recommendation_color = "red"
            recommendation_icon = "TRENDING DOWN"
        else:
            recommendation = "STRONG SELL"
            recommendation_color = "darkred"
            recommendation_icon = "WARNING"
        
        # Display recommendation
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid {recommendation_color}; border-radius: 10px;'>
            <h2 style='color: {recommendation_color};'>{recommendation}</h2>
            <h3>Score: {total_score:.1f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown
        st.subheader("Recommendation Score Breakdown")
        
        score_data = {
            'Component': ['Technical Analysis', 'Fundamental Health', 'Market Sentiment', 'Price Forecast'],
            'Score': [technical_score, fundamental_score, sentiment_score_normalized, forecast_score],
            'Max Score': [50, 50, 50, 25]
        }
        
        score_df = pd.DataFrame(score_data)
        score_df['Percentage'] = (score_df['Score'] / score_df['Max Score'] * 100).round(1)
        
        # Create score breakdown chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=score_df['Component'],
            y=score_df['Score'],
            text=score_df['Percentage'].astype(str) + '%',
            textposition='auto',
            marker_color=['blue', 'green', 'orange', 'purple']
        ))
        
        fig.update_layout(
            title="Recommendation Score Components",
            xaxis_title="Analysis Component",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.subheader("Key Risk Factors")
        
        risk_factors = []
        
        # Technical risks
        if rsi > 70:
            risk_factors.append("**Technical Risk**: Stock appears overbought (RSI > 70)")
        elif rsi < 30:
            risk_factors.append("**Technical Risk**: Stock appears oversold (RSI < 30)")
        
        # Fundamental risks
        if fundamental_data.get('debt_to_equity', 0) > 1:
            risk_factors.append("**Financial Risk**: High debt-to-equity ratio")
        
        if fundamental_data.get('pe_ratio', 0) > 50:
            risk_factors.append("**Valuation Risk**: High P/E ratio suggests overvaluation")
        
        # Sentiment risks
        if sentiment_score < -0.3:
            risk_factors.append("**Sentiment Risk**: Negative news sentiment")
        
        # Forecast risks
        if forecast_change < -10:
            risk_factors.append("**Forecast Risk**: Significant downward price prediction")
        
        if not risk_factors:
            risk_factors.append("**Low Risk**: No significant risk factors identified")
        
        for risk in risk_factors:
            st.markdown(risk)
        
        # Disclaimer
        st.subheader("Important Disclaimer")
        st.warning("""
        This AI analysis is for informational purposes only and should not be considered as financial advice. 
        Always conduct your own research and consult with a financial advisor before making investment decisions. 
        Past performance does not guarantee future results, and all investments carry risk.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>StockForecastX Pro - Advanced AI Stock Analysis Platform</p>
    <p>Built with Streamlit, Python, and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
