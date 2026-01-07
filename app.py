from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import yfinance as yf

# Load environment variables from .env file
load_dotenv()
import pandas as pd
import numpy as np
import json
import requests
import datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
import time
from typing import Optional, Dict, Any
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def clean_for_json(obj):
    """Recursively clean data for JSON serialization, replacing NaN with None"""
    if isinstance(obj, pd.DataFrame):
        # Convert to dict first, then clean NaN values
        records = obj.to_dict('records')
        return clean_for_json(records)
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        return None if pd.isna(obj) else obj
    elif isinstance(obj, (int, np.integer)):
        return int(obj) if not pd.isna(obj) else None
    else:
        return obj

# API Keys - Load from environment variables for security
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_alpha_vantage_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get stock data from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        
        if "Error Message" in data or "Note" in data or "Time Series (Daily)" not in data:
            return pd.DataFrame()
        
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Adj Close'] = df['Close']
        df.index = pd.to_datetime(df.index)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        df = df.sort_index()
        df = df.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"Error fetching Alpha Vantage data: {str(e)}")
        return pd.DataFrame()

def cached_stock_data(ticker: str, start_date: str, end_date: str, data_source: str = "Alpha Vantage (Recommended)") -> pd.DataFrame:
    """Cache stock data to avoid repeated API calls"""
    import random
    
    if data_source == "Alpha Vantage (Recommended)":
        data = get_alpha_vantage_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            return data
    
    time.sleep(random.uniform(2, 5))
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
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                # Ensure we have a single-level index
                if isinstance(data.index, pd.MultiIndex):
                    data = data.reset_index()
                    if 'Date' in data.columns:
                        data = data.set_index('Date')
                return data
            else:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    time.sleep(wait_time)
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                wait_time = (attempt + 1) * 20
                time.sleep(wait_time)
            else:
                time.sleep(3)
    
    return pd.DataFrame()

def get_weather(city_state: str, api_key: str) -> str:
    try:
        city_state = city_state.strip().title().replace(" ", ", ")
        
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

        for full_state, abbr in state_abbr.items():
            if full_state.lower() in city_state.lower():
                city_state = city_state.replace(full_state, abbr)
                break
        
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

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators for stock analysis"""
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_STD'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_STD'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_STD'])
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['MACD_Line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    high_low = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # OBV (On-Balance Volume) - Fixed to avoid multidimensional indexing issues
    price_change = data['Close'].diff()
    obv = np.where(price_change > 0, data['Volume'].values,
                   np.where(price_change < 0, -data['Volume'].values, 0))
    data['OBV'] = pd.Series(obv, index=data.index).cumsum()
    
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    numerator = data['Close'] - low_14
    denominator = high_14 - low_14
    data['%K'] = 100 * (numerator / denominator)
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    return data

def get_stock_sentiment(ticker: str, num_articles: int = 5) -> Dict[str, Any]:
    """Get sentiment analysis from news articles using OpenAI
    Note: Alpha Vantage API key is ONLY used for historical stock data and fundamental data.
    For sentiment/news, we use yfinance (free, no API key needed).
    """
    articles = []
    sentiment_scores = []
    error_message = None
    
    try:
        # Use yfinance for news articles (Alpha Vantage API key is reserved for historical stock data only)
        print(f"Fetching news articles from yfinance for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news
            
            if yf_news and len(yf_news) > 0:
                print(f"Found {len(yf_news)} articles from yfinance")
                if yf_news:
                    print(f"Sample yfinance news item keys: {list(yf_news[0].keys())}")
                    print(f"Sample yfinance news item: {str(yf_news[0])[:500]}")
                
                for idx, news_item in enumerate(yf_news[:num_articles]):
                    # yfinance news structure: usually has 'title', 'link', 'publisher', 'providerPublishTime', 'uuid'
                    # Print all keys for debugging
                    print(f"\nArticle {idx + 1} keys: {list(news_item.keys())}")
                    print(f"Article {idx + 1} data: {str(news_item)[:300]}")
                    
                    # Try multiple possible field names for title - yfinance uses 'title'
                    title = None
                    for key in ['title', 'headline', 'summary', 'text']:
                        if key in news_item and news_item[key]:
                            title = str(news_item[key]).strip()
                            if title and len(title) > 5:  # Make sure it's a real title
                                break
                    
                    if not title:
                        title = f"News Article {idx + 1}"
                    
                    # Try multiple possible field names for link - yfinance uses 'link'
                    link = None
                    for key in ['link', 'url', 'uuid']:
                        if key in news_item and news_item[key]:
                            link = str(news_item[key])
                            if link:
                                break
                    link = link or ""
                    
                    # Get publication date - yfinance uses 'providerPublishTime'
                    pub_date = None
                    for key in ['providerPublishTime', 'pubDate', 'datetime', 'publishedAt']:
                        if key in news_item and news_item[key]:
                            pub_date = news_item[key]
                            break
                    
                    # Convert timestamp to readable format
                    if pub_date:
                        try:
                            if isinstance(pub_date, (int, float)) and pub_date > 0:
                                # yfinance timestamps are usually in seconds
                                time_published = datetime.datetime.fromtimestamp(pub_date).strftime("%Y%m%dT%H%M%S")
                            elif isinstance(pub_date, str):
                                # Try parsing string dates
                                time_published = pub_date
                            else:
                                time_published = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        except:
                            time_published = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    else:
                        time_published = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    
                    # Get summary/publisher info - yfinance uses 'publisher'
                    publisher = None
                    for key in ['publisher', 'source', 'provider']:
                        if key in news_item and news_item[key]:
                            publisher = str(news_item[key])
                            if publisher:
                                break
                    publisher = publisher or "Yahoo Finance"
                    
                    # Get summary - yfinance may not have summary, but check
                    summary_text = None
                    for key in ['summary', 'description', 'text', 'content']:
                        if key in news_item and news_item[key]:
                            summary_text = str(news_item[key]).strip()
                            if summary_text and len(summary_text) > 20:
                                break
                    
                    if not summary_text or summary_text == title:
                        summary_text = None
                    
                    print(f"Extracted - title='{title[:60]}...', publisher='{publisher}', has_summary={summary_text is not None}")
                    
                    # Use OpenAI to analyze sentiment - use summary if available, otherwise title
                    sentiment = 0.0
                    text_to_analyze = summary_text if (summary_text and len(summary_text) > len(title)) else title
                    
                    # Skip if we don't have a real title
                    if not title or title == f"News Article {idx + 1}":
                        print(f"Skipping article {idx + 1} - no valid title")
                        continue
                    
                    if openai_client and text_to_analyze:
                        try:
                            prompt = f"""Analyze the sentiment of this news article about {ticker} stock. 
                            
Title: {title}
Summary: {summary_text[:500] if summary_text else 'N/A'}
Source: {publisher}

Provide a sentiment score from -1 (very negative) to +1 (very positive), where:
- -1 to -0.3: Very negative/bearish
- -0.3 to -0.1: Somewhat negative
- -0.1 to 0.1: Neutral
- 0.1 to 0.3: Somewhat positive
- 0.3 to 1: Very positive/bullish

Respond with ONLY a number between -1 and 1, nothing else."""
                            
                            completion = openai_client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a financial sentiment analyst. Always respond with only a number between -1.0 and 1.0."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.3,
                                max_tokens=20
                            )
                            
                            sentiment_text = completion.choices[0].message.content.strip()
                            print(f"OpenAI response for article {idx + 1}: '{sentiment_text}'")
                            
                            # Extract number from response
                            import re
                            sentiment_match = re.search(r'-?\d+\.?\d*', sentiment_text)
                            if sentiment_match:
                                sentiment = float(sentiment_match.group())
                                sentiment = max(-1.0, min(1.0, sentiment))
                                print(f"Parsed sentiment: {sentiment:.2f}")
                            else:
                                print(f"Could not parse sentiment from: {sentiment_text}, using TextBlob fallback")
                                blob = TextBlob(text_to_analyze)
                                sentiment = float(blob.sentiment.polarity)
                                print(f"TextBlob sentiment: {sentiment:.2f}")
                        except Exception as e:
                            print(f"OpenAI sentiment analysis error for article {idx + 1}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback to TextBlob
                            try:
                                blob = TextBlob(text_to_analyze)
                                sentiment = float(blob.sentiment.polarity)
                                print(f"TextBlob fallback sentiment: {sentiment:.2f}")
                            except:
                                sentiment = 0.0
                    else:
                        # Fallback to TextBlob if OpenAI not available or no text
                        try:
                            if text_to_analyze and text_to_analyze != f"News Article {idx + 1}":
                                blob = TextBlob(text_to_analyze)
                                sentiment = float(blob.sentiment.polarity)
                                print(f"TextBlob sentiment (no OpenAI): {sentiment:.2f}")
                            else:
                                print(f"No text to analyze for article {idx + 1}, using 0.0")
                                sentiment = 0.0
                        except:
                            sentiment = 0.0
                    
                    # Generate a brief debrief/summary using OpenAI if we have good text
                    debrief = summary_text if (summary_text and len(summary_text) > 50) else None
                    
                    # If no good summary, try to generate one with OpenAI
                    if not debrief:
                        if openai_client and title:
                            try:
                                debrief_prompt = f"""Provide a brief 2-3 sentence summary of this news article about {ticker} stock. Focus on the key points and implications.

Title: {title}
{f'Summary: {summary_text[:300]}' if summary_text and summary_text != title else ''}

Respond with only a concise summary, 2-3 sentences maximum."""
                                
                                debrief_completion = openai_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a financial news summarizer. Provide concise 2-3 sentence summaries."},
                                        {"role": "user", "content": debrief_prompt}
                                    ],
                                    temperature=0.5,
                                    max_tokens=100
                                )
                                debrief = debrief_completion.choices[0].message.content.strip()
                                print(f"Generated debrief for article {idx + 1}")
                            except Exception as e:
                                print(f"Error generating debrief: {e}")
                                debrief = summary_text if summary_text else f"News article about {ticker} from {publisher}."
                    
                    # Only add article if we have a valid title
                    if title and title != f"News Article {idx + 1}":
                        sentiment_scores.append(sentiment)
                        articles.append({
                            "title": title,
                            "url": link,
                            "time_published": time_published,
                            "sentiment": float(sentiment),
                            "summary": debrief if debrief else (summary_text if summary_text else f"News article about {ticker} from {publisher}."),
                            "publisher": publisher
                        })
                        print(f"Added article {idx + 1}: title='{title[:50]}', sentiment={sentiment:.2f}")
                    else:
                        print(f"Skipped article {idx + 1} - invalid title: '{title}'")
            else:
                print("No articles found from yfinance")
                if not error_message:
                    error_message = "No articles found from yfinance"
        except Exception as yf_error:
                print(f"Error fetching news from yfinance: {yf_error}")
                import traceback
                traceback.print_exc()
                if not error_message:
                    error_message = f"yfinance news fetch failed: {yf_error}"
        
        
        # Calculate average sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            print(f"Average sentiment from {len(sentiment_scores)} articles: {avg_sentiment:.2f}")
        else:
            # If no articles, ALWAYS use OpenAI to generate a general sentiment
            print("No articles found, trying general OpenAI sentiment...")
            avg_sentiment = 0.0
            
            if openai_client:
                try:
                    # More detailed prompt for better results
                    prompt = f"""You are a financial analyst. Analyze the current market sentiment for {ticker} stock.

Consider:
- Recent price trends
- Market conditions
- Industry outlook
- General investor sentiment

Provide a sentiment score from -1.0 (very negative/bearish) to +1.0 (very positive/bullish), where:
- -1.0 to -0.3: Very negative/bearish
- -0.3 to -0.1: Somewhat negative
- -0.1 to 0.1: Neutral
- 0.1 to 0.3: Somewhat positive
- 0.3 to 1.0: Very positive/bullish

Respond with ONLY a decimal number between -1.0 and 1.0. Example: 0.25 or -0.15"""
                    
                    print(f"Calling OpenAI for general sentiment for {ticker}...")
                    completion = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a financial sentiment analyst. Always respond with only a number between -1.0 and 1.0."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=20
                    )
                    
                    sentiment_text = completion.choices[0].message.content.strip()
                    print(f"General OpenAI sentiment raw response: '{sentiment_text}'")
                    
                    # Extract number from response - try multiple patterns
                    import re
                    # Try to find decimal number
                    sentiment_match = re.search(r'-?\d+\.?\d*', sentiment_text)
                    if sentiment_match:
                        avg_sentiment = float(sentiment_match.group())
                        avg_sentiment = max(-1.0, min(1.0, avg_sentiment))
                        print(f"Successfully parsed general sentiment: {avg_sentiment:.2f}")
                    else:
                        print(f"ERROR: Could not parse sentiment from response: '{sentiment_text}'")
                        # Try TextBlob as last resort
                        blob = TextBlob(f"{ticker} stock market sentiment")
                        avg_sentiment = float(blob.sentiment.polarity)
                        print(f"Using TextBlob fallback: {avg_sentiment:.2f}")
                        
                except Exception as e:
                    print(f"ERROR: General OpenAI sentiment call failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try TextBlob as fallback
                    try:
                        blob = TextBlob(f"{ticker} stock market sentiment")
                        avg_sentiment = float(blob.sentiment.polarity)
                        print(f"Using TextBlob fallback after OpenAI error: {avg_sentiment:.2f}")
                    except:
                        avg_sentiment = 0.0
                        print("All sentiment methods failed, using 0.0")
            else:
                print("ERROR: OpenAI client is not initialized!")
                # Try TextBlob
                try:
                    blob = TextBlob(f"{ticker} stock market sentiment")
                    avg_sentiment = float(blob.sentiment.polarity)
                    print(f"Using TextBlob fallback (no OpenAI): {avg_sentiment:.2f}")
                except:
                    avg_sentiment = 0.0
        
        # If we got general sentiment but no articles from any source, add a placeholder article
        if not articles and avg_sentiment != 0.0:
            articles.append({
                "title": f"General Market Sentiment Analysis for {ticker}",
                "url": "",
                "time_published": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
                "sentiment": float(avg_sentiment),
                "summary": f"AI-generated general market sentiment analysis for {ticker} based on current market conditions and trends.",
                "publisher": "AI Analysis"
            })
            print(f"Added general sentiment as article with score: {avg_sentiment:.2f}")
        
        # Determine final error message
        final_error = None
        if error_message:
            final_error = error_message
        elif not articles:
            final_error = "Using AI-generated general sentiment (no news articles available)"
        
        result = {
            "score": float(avg_sentiment) if not pd.isna(avg_sentiment) else 0.0,
            "articles": articles,
            "error": final_error
        }
        
        # Validate articles
        for article in result["articles"]:
            if "sentiment" not in article or article["sentiment"] is None:
                article["sentiment"] = 0.0
            else:
                article["sentiment"] = float(article["sentiment"])
        
        print(f"Final sentiment result: score={result['score']:.2f}, articles={len(result['articles'])}, error={result['error']}")
        return result
        
    except Exception as e:
        print(f"Exception in get_stock_sentiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "articles": [],
            "error": str(e) if str(e) else "Unknown error"
        }

def get_alpha_vantage_fundamental_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get fundamental data from Alpha Vantage"""
    try:
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url, timeout=10)
        
        if overview_response.status_code != 200:
            return None
        
        overview_data = overview_response.json()
        
        if "Error Message" in overview_data or "Note" in overview_data:
            return None
        
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
        
        score = 0
        max_score = 0
        
        # P/E Ratio scoring (0-20 points)
        if financials["pe_ratio"] is not None and financials["pe_ratio"] > 0:
            max_score += 20
            if 5 < financials["pe_ratio"] < 25:
                score += 20
            elif 0 < financials["pe_ratio"] <= 5:
                score += 15
            elif 25 <= financials["pe_ratio"] < 40:
                score += 10
            else:
                score += 5
            
        # PEG Ratio scoring (0-20 points)
        if financials["peg_ratio"] is not None and financials["peg_ratio"] > 0:
            max_score += 20
            if 0 < financials["peg_ratio"] < 1.5:
                score += 20
            elif 1.5 <= financials["peg_ratio"] < 2.5:
                score += 15
            elif 2.5 <= financials["peg_ratio"] < 3.5:
                score += 10
            else:
                score += 5
            
        # Debt-to-Equity scoring (0-20 points)
        if financials["debt_to_equity"] is not None and financials["debt_to_equity"] >= 0:
            max_score += 20
            if financials["debt_to_equity"] < 0.5:
                score += 20
            elif 0.5 <= financials["debt_to_equity"] < 1:
                score += 15
            elif 1 <= financials["debt_to_equity"] < 2:
                score += 10
            else:
                score += 5
            
        # Return on Equity scoring (0-20 points)
        if financials["return_on_equity"] is not None:
            max_score += 20
            if financials["return_on_equity"] > 20:
                score += 20
            elif financials["return_on_equity"] > 15:
                score += 15
            elif financials["return_on_equity"] > 10:
                score += 10
            elif financials["return_on_equity"] > 5:
                score += 5
            
        # Profit Margins scoring (0-20 points)
        if financials["profit_margins"] is not None:
            max_score += 20
            if financials["profit_margins"] > 20:
                score += 20
            elif financials["profit_margins"] > 15:
                score += 15
            elif financials["profit_margins"] > 10:
                score += 10
            elif financials["profit_margins"] > 5:
                score += 5
        
        # Calculate percentage score (0-100)
        if max_score > 0:
            financial_health_score = round((score / max_score) * 100)
        else:
            financial_health_score = 0
            
        financials["financial_health_score"] = max(0, min(100, financial_health_score))
        
        return financials
    except Exception as e:
        print(f"Error fetching Alpha Vantage fundamental data: {e}")
        return None

def get_fundamental_data(ticker: str, data_source: str = "Alpha Vantage (Recommended)") -> Dict[str, Any]:
    """Get fundamental financial data for a stock"""
    try:
        if data_source == "Alpha Vantage (Recommended)":
            alpha_data = get_alpha_vantage_fundamental_data(ticker)
            if alpha_data:
                return alpha_data
        
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
        
        score = 0
        max_score = 0
        
        # P/E Ratio scoring (0-20 points)
        if financials["pe_ratio"] is not None and financials["pe_ratio"] > 0:
            max_score += 20
            if 5 < financials["pe_ratio"] < 25:
                score += 20
            elif 0 < financials["pe_ratio"] <= 5:
                score += 15
            elif 25 <= financials["pe_ratio"] < 40:
                score += 10
            else:
                score += 5
            
        # PEG Ratio scoring (0-20 points)
        if financials["peg_ratio"] is not None and financials["peg_ratio"] > 0:
            max_score += 20
            if 0 < financials["peg_ratio"] < 1.5:
                score += 20
            elif 1.5 <= financials["peg_ratio"] < 2.5:
                score += 15
            elif 2.5 <= financials["peg_ratio"] < 3.5:
                score += 10
            else:
                score += 5
            
        # Debt-to-Equity scoring (0-20 points)
        if financials["debt_to_equity"] is not None and financials["debt_to_equity"] >= 0:
            max_score += 20
            if financials["debt_to_equity"] < 0.5:
                score += 20
            elif 0.5 <= financials["debt_to_equity"] < 1:
                score += 15
            elif 1 <= financials["debt_to_equity"] < 2:
                score += 10
            else:
                score += 5
            
        # Return on Equity scoring (0-20 points)
        if financials["return_on_equity"] is not None:
            max_score += 20
            if financials["return_on_equity"] > 20:
                score += 20
            elif financials["return_on_equity"] > 15:
                score += 15
            elif financials["return_on_equity"] > 10:
                score += 10
            elif financials["return_on_equity"] > 5:
                score += 5
            
        # Profit Margins scoring (0-20 points)
        if financials["profit_margins"] is not None:
            max_score += 20
            if financials["profit_margins"] > 20:
                score += 20
            elif financials["profit_margins"] > 15:
                score += 15
            elif financials["profit_margins"] > 10:
                score += 10
            elif financials["profit_margins"] > 5:
                score += 5
        
        # Calculate percentage score (0-100)
        if max_score > 0:
            financial_health_score = round((score / max_score) * 100)
        else:
            financial_health_score = 0
            
        financials["financial_health_score"] = max(0, min(100, financial_health_score))
        
        return financials
    except Exception as e:
        return {
            "name": ticker,
            "error": str(e),
            "financial_health_score": 0
        }

def get_ai_analysis(ticker: str, price_data: pd.DataFrame, forecast_data: pd.DataFrame, fundamental_data: Dict[str, Any], sentiment_data: Dict[str, Any]) -> str:
    """Get AI analysis of the stock using OpenAI API"""
    try:
        current_price = price_data['Close'].iloc[-1]
        price_change_1d = ((current_price / price_data['Close'].iloc[-2]) - 1) * 100 if len(price_data) >= 2 else 0
        price_change_1w = ((current_price / price_data['Close'].iloc[-6]) - 1) * 100 if len(price_data) >= 6 else 0
        price_change_1m = ((current_price / price_data['Close'].iloc[-22]) - 1) * 100 if len(price_data) >= 22 else 0
        
        latest_data = price_data.iloc[-1]
        rsi = latest_data.get('RSI', 0) if 'RSI' in price_data.columns else 0
        macd = latest_data.get('MACD_Line', 0) if 'MACD_Line' in price_data.columns else 0
        macd_signal = latest_data.get('MACD_Signal', 0) if 'MACD_Signal' in price_data.columns else 0
        
        forecast_price = forecast_data['yhat'].iloc[-1]
        forecast_change = ((forecast_price / current_price) - 1) * 100
        
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
        
        if openai_client:
            try:
                completion = openai_client.chat.completions.create(
                    model="gpt-4",
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
                print(f"AI analysis API error: {api_error}")
        
        # Fallback to rule-based analysis (same as original)
        sentiment_word = "positive" if sentiment_data.get('score', 0) > 0.2 else "neutral" if sentiment_data.get('score', 0) > -0.2 else "negative"
        forecast_direction = "bullish" if forecast_change > 0 else "bearish"
        rsi_condition = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        if rsi > 70 and macd < macd_signal:
            technical_analysis = f"**Technical Analysis**: {ticker} appears technically {rsi_condition} with an RSI of {rsi:.2f}, suggesting the stock may be due for a pullback. The MACD shows bearish divergence as the MACD line has crossed below the signal line, reinforcing the potential for a short-term price correction. Volume patterns indicate decreasing buying momentum."
        elif rsi < 30 and macd > macd_signal:
            technical_analysis = f"**Technical Analysis**: {ticker} is showing {rsi_condition} conditions with an RSI of {rsi:.2f}, potentially offering a buying opportunity. The MACD has formed a bullish crossover with the MACD line moving above the signal line, suggesting growing momentum. Recent price action indicates a potential bottom formation."
        else:
            technical_analysis = f"**Technical Analysis**: {ticker} is currently in a {rsi_condition} zone with an RSI of {rsi:.2f}. The MACD indicator shows {'bullish' if macd > macd_signal else 'bearish'} momentum. Price action is {'above' if current_price > latest_data.get('SMA_50', 0) else 'below'} the 50-day moving average, indicating a {'positive' if current_price > latest_data.get('SMA_50', 0) else 'negative'} intermediate-term trend."
        
        if fundamental_data.get('financial_health_score', 0) > 70:
            fundamental_analysis = f"**Fundamental Analysis**: {ticker} demonstrates strong financial health with a score of {fundamental_data.get('financial_health_score', 0)}/100. The P/E ratio of {fundamental_data.get('pe_ratio', 'N/A')} suggests {'reasonable valuation' if fundamental_data.get('pe_ratio', 100) < 25 else 'premium valuation'}, while the debt-to-equity ratio of {fundamental_data.get('debt_to_equity', 'N/A')} indicates {'conservative' if fundamental_data.get('debt_to_equity', 2) < 1 else 'significant'} leverage. The company's return on equity of {fundamental_data.get('return_on_equity', 'N/A')}% reveals {'excellent' if fundamental_data.get('return_on_equity', 0) > 15 else 'adequate'} profitability relative to shareholder investments."
        else:
            fundamental_analysis = f"**Fundamental Analysis**: {ticker} shows {'moderate' if fundamental_data.get('financial_health_score', 0) > 40 else 'concerning'} financial metrics with a health score of {fundamental_data.get('financial_health_score', 0)}/100. The P/E ratio stands at {fundamental_data.get('pe_ratio', 'N/A')}, which is {'below' if fundamental_data.get('pe_ratio', 0) < 15 and fundamental_data.get('pe_ratio', 0) > 0 else 'above'} industry averages. The debt-to-equity ratio of {fundamental_data.get('debt_to_equity', 'N/A')} suggests {'manageable' if fundamental_data.get('debt_to_equity', 0) < 1.5 else 'elevated'} financial risk, while return on equity at {fundamental_data.get('return_on_equity', 'N/A')}% indicates {'reasonable' if fundamental_data.get('return_on_equity', 0) > 10 else 'suboptimal'} operational efficiency."
        
        outlook = f"**Outlook**: Based on comprehensive analysis, the forecast for {ticker} appears {forecast_direction} with a target price of ${forecast_price:.2f}, representing a potential {abs(forecast_change):.2f}% {'gain' if forecast_change > 0 else 'loss'}. News sentiment is {sentiment_word} at {sentiment_data.get('score', 0):.2f}, {'supporting' if (sentiment_data.get('score', 0) > 0 and forecast_change > 0) or (sentiment_data.get('score', 0) < 0 and forecast_change < 0) else 'contradicting'} the price forecast. Investors should {'consider accumulating positions' if forecast_change > 10 else 'maintain current positions' if forecast_change > 0 else 'consider reducing exposure'}, while monitoring key resistance at ${current_price * 1.05:.2f} and support at ${current_price * 0.95:.2f}. {'Market volatility may present better entry points in the near term.' if rsi > 60 else 'Current price levels may offer an attractive entry point.' if rsi < 40 else 'Maintain a balanced approach to position sizing given current market conditions.'}"
        
        return f"{technical_analysis}\n\n{fundamental_analysis}\n\n{outlook}"
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def train_ml_models(df: pd.DataFrame, forecast_days: int = 7) -> tuple:
    """Train machine learning models for stock prediction"""
    if len(df) < 30:
        models = {}
        current_price = float(df['Close'].iloc[-1]) if len(df) > 0 and 'Close' in df.columns else 100.0
        predictions = pd.DataFrame()
        predictions['day'] = range(1, forecast_days + 1)
        predictions['prediction'] = [current_price] * forecast_days  # Use current price as baseline
        predictions['mae'] = [0.0] * forecast_days
        predictions['rmse'] = [0.0] * forecast_days
        predictions['r2'] = [0.0] * forecast_days
        return models, predictions
    
    df_ml = df.copy()
    
    for lag in [1, 2, 3, 5, 14, 21]:
        df_ml[f'lag_{lag}'] = df_ml['Close'].shift(lag)
    
    for window in [7, 14, 30]:
        df_ml[f'rolling_mean_{window}'] = df_ml['Close'].rolling(window=window).mean()
        df_ml[f'rolling_std_{window}'] = df_ml['Close'].rolling(window=window).std()
    
    for period in [1, 3, 7, 14]:
        df_ml[f'momentum_{period}'] = df_ml['Close'].pct_change(periods=period)
    
    df_ml['volume_1d_change'] = df_ml['Volume'].pct_change()
    df_ml['volume_ma_ratio'] = df_ml['Volume'] / df_ml['Volume'].rolling(window=10).mean()
    
    df_ml = df_ml.dropna()
    
    for day in range(1, forecast_days + 1):
        df_ml[f'target_{day}d'] = df_ml['Close'].shift(-day)
    
    df_ml = df_ml.dropna()
    
    feature_columns = [col for col in df_ml.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close'] and not col.startswith('target_')]
    
    models = {}
    predictions = pd.DataFrame(index=range(forecast_days))
    predictions['day'] = range(1, forecast_days + 1)
    
    for day in range(1, forecast_days + 1):
        target = f'target_{day}d'
        
        X = df_ml[feature_columns]
        y = df_ml[target]
        
        if len(X) < 10:
            X_train, y_train = X, y
            X_test, y_test = X.iloc[-1:], y.iloc[-1:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        models[day] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        latest_data = df_ml[feature_columns].iloc[-1:]
        try:
            prediction = float(model.predict(latest_data)[0])
            # Ensure prediction is valid (positive and reasonable)
            if pd.isna(prediction) or prediction <= 0:
                prediction = float(df_ml['Close'].iloc[-1]) if len(df_ml) > 0 else 100.0
        except Exception as pred_error:
            print(f"Error making prediction for day {day}: {pred_error}")
            prediction = float(df_ml['Close'].iloc[-1]) if len(df_ml) > 0 else 100.0
        
        predictions.loc[day-1, 'prediction'] = float(prediction)
        predictions.loc[day-1, 'mae'] = float(mae) if not pd.isna(mae) else 0.0
        predictions.loc[day-1, 'rmse'] = float(rmse) if not pd.isna(rmse) else 0.0
        predictions.loc[day-1, 'r2'] = float(r2) if not pd.isna(r2) else 0.0
    
    # Final validation - ensure all predictions are valid numbers
    for idx in predictions.index:
        if pd.isna(predictions.loc[idx, 'prediction']) or predictions.loc[idx, 'prediction'] <= 0:
            predictions.loc[idx, 'prediction'] = float(df_ml['Close'].iloc[-1]) if len(df_ml) > 0 else 100.0
    
    return models, predictions

def get_llm_price_prediction(
    ticker: str,
    current_price: float,
    price_data: pd.DataFrame,
    technical_indicators: pd.DataFrame,
    fundamental_data: Dict[str, Any],
    sentiment_data: Dict[str, Any],
    ml_predictions: pd.DataFrame,
    prophet_forecast: pd.DataFrame,
    forecast_period: int
) -> Dict[str, Any]:
    """
    Get professional stock price prediction using LLM with reasoning.
    Acts as an experienced stock analyst who provides realistic, conservative predictions.
    """
    try:
        if not openai_client:
            print("OpenAI client not available for LLM price prediction")
            return {
                "predictions": [],
                "reasoning": "OpenAI API not available",
                "error": "OpenAI client not configured"
            }
        
        # Calculate key metrics
        latest_data = technical_indicators.iloc[-1]
        rsi = float(latest_data.get('RSI', 0)) if 'RSI' in technical_indicators.columns else 0.0
        macd = float(latest_data.get('MACD_Line', 0)) if 'MACD_Line' in technical_indicators.columns else 0.0
        macd_signal = float(latest_data.get('MACD_Signal', 0)) if 'MACD_Signal' in technical_indicators.columns else 0.0
        sma_50 = float(latest_data.get('SMA_50', current_price)) if 'SMA_50' in technical_indicators.columns else current_price
        sma_200 = float(latest_data.get('SMA_200', current_price)) if 'SMA_200' in technical_indicators.columns else current_price
        volume_avg = float(technical_indicators['Volume'].tail(20).mean()) if 'Volume' in technical_indicators.columns else 0
        
        # Price trends
        price_change_1d = ((current_price / price_data['Close'].iloc[-2]) - 1) * 100 if len(price_data) >= 2 else 0
        price_change_1w = ((current_price / price_data['Close'].iloc[-6]) - 1) * 100 if len(price_data) >= 6 else 0
        price_change_1m = ((current_price / price_data['Close'].iloc[-22]) - 1) * 100 if len(price_data) >= 22 else 0
        price_change_3m = ((current_price / price_data['Close'].iloc[-66]) - 1) * 100 if len(price_data) >= 66 else 0
        
        # Get ML and Prophet predictions for key days
        ml_pred_7d = float(ml_predictions[ml_predictions['day'] == 7]['prediction'].iloc[0]) if len(ml_predictions[ml_predictions['day'] == 7]) > 0 else current_price
        ml_pred_14d = float(ml_predictions[ml_predictions['day'] == forecast_period]['prediction'].iloc[0]) if len(ml_predictions[ml_predictions['day'] == forecast_period]) > 0 else current_price
        
        prophet_pred_7d = float(prophet_forecast.iloc[6]['yhat']) if len(prophet_forecast) > 6 else current_price
        prophet_pred_final = float(prophet_forecast.iloc[-1]['yhat']) if len(prophet_forecast) > 0 else current_price
        
        # Calculate realistic price movement limits based on current price
        # Most stocks move ±1-3% per day, ±3-7% per week, ±5-15% per month under normal conditions
        realistic_daily_change_pct = 2.0  # Max 2% per day unless exceptional
        realistic_weekly_change_pct = 5.0  # Max 5% per week unless exceptional
        realistic_biweekly_change_pct = 8.0  # Max 8% over 14 days unless exceptional
        
        # Calculate absolute dollar limits
        realistic_daily_change_dollars = current_price * (realistic_daily_change_pct / 100)
        realistic_weekly_change_dollars = current_price * (realistic_weekly_change_pct / 100)
        realistic_biweekly_change_dollars = current_price * (realistic_biweekly_change_pct / 100)
        
        # Prepare comprehensive prompt for professional analysis
        prompt = f"""You are an experienced professional stock analyst with 20+ years of experience. Your task is to predict REALISTIC stock prices for {ticker} over the next {forecast_period} days.

CRITICAL REALISTIC PRICE MOVEMENT CONSTRAINTS:
- Current Price: ${current_price:.2f}
- REALISTIC daily moves: ±1-3% per day = ±${realistic_daily_change_dollars:.2f} per day MAX (unless exceptional circumstances)
- REALISTIC weekly moves: ±3-7% per week = ±${realistic_weekly_change_dollars:.2f} per week MAX (unless exceptional circumstances)
- REALISTIC 14-day moves: ±5-15% = ±${realistic_biweekly_change_dollars:.2f} over 14 days MAX (unless exceptional circumstances)

STRICT RULES:
1. A $25 price increase in ONE day would require a {(25/current_price*100):.1f}% move - this is EXTREMELY RARE and would require MAJOR news (earnings beat, FDA approval, merger announcement, etc.). DO NOT predict this unless you have SOLID EVIDENCE.
2. A $25 price increase over 14 days would require a {(25/current_price*100):.1f}% move - still VERY UNLIKELY without strong evidence (strong earnings, major product launch, etc.).
3. Most stocks move ±${realistic_daily_change_dollars:.2f} per day and ±${realistic_weekly_change_dollars:.2f} per week under NORMAL market conditions.
4. If you predict a move >5% in 1 day or >10% in 14 days, you MUST provide EXCEPTIONAL reasoning with SPECIFIC evidence.
5. BE EXTREMELY CONSERVATIVE. It's better to slightly underestimate than overestimate price movements.
6. Stock prices rarely move in straight lines. Factor in normal volatility and pullbacks.

PREDICTION GUIDELINES:
- Day 3: Should be within ±{(realistic_daily_change_dollars*3):.2f} (${{realistic_daily_change_dollars:.2f}} × 3 days) of current price = ${current_price - realistic_daily_change_dollars*3:.2f} to ${current_price + realistic_daily_change_dollars*3:.2f}
- Day 7: Should be within ±${realistic_weekly_change_dollars:.2f} of current price = ${current_price - realistic_weekly_change_dollars:.2f} to ${current_price + realistic_weekly_change_dollars:.2f}
- Day 14: Should be within ±${realistic_biweekly_change_dollars:.2f} of current price = ${current_price - realistic_biweekly_change_dollars:.2f} to ${current_price + realistic_biweekly_change_dollars:.2f}

IMPORTANT: If you predict a price outside these ranges, you MUST justify it with STRONG, SPECIFIC evidence in your reasoning. Generic explanations are NOT acceptable for extreme predictions.

CURRENT STOCK DATA:
- Current Price: ${current_price:.2f}
- 1-Day Change: {price_change_1d:.2f}%
- 1-Week Change: {price_change_1w:.2f}%
- 1-Month Change: {price_change_1m:.2f}%
- 3-Month Change: {price_change_3m:.2f}%

TECHNICAL INDICATORS:
- RSI (14): {rsi:.2f} {'(Overbought >70)' if rsi > 70 else '(Oversold <30)' if rsi < 30 else '(Neutral)'}
- MACD: {macd:.4f}
- MACD Signal: {macd_signal:.4f}
- 50-Day SMA: ${sma_50:.2f} (Price is {'above' if current_price > sma_50 else 'below'} this level)
- 200-Day SMA: ${sma_200:.2f} (Price is {'above' if current_price > sma_200 else 'below'} this level)

FUNDAMENTAL DATA:
- Sector: {fundamental_data.get('sector', 'Unknown')}
- P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}
- Forward P/E: {fundamental_data.get('forward_pe', 'N/A')}
- PEG Ratio: {fundamental_data.get('peg_ratio', 'N/A')}
- Debt-to-Equity: {fundamental_data.get('debt_to_equity', 'N/A')}
- Return on Equity: {fundamental_data.get('return_on_equity', 'N/A')}%
- Financial Health Score: {fundamental_data.get('financial_health_score', 0)}/100

MARKET SENTIMENT:
- News Sentiment Score: {sentiment_data.get('score', 0):.2f} (range: -1 to +1, where +1 is very bullish, -1 is very bearish)

MACHINE LEARNING PREDICTIONS:
- ML Model 7-Day Prediction: ${ml_pred_7d:.2f} ({((ml_pred_7d/current_price - 1) * 100):.2f}%)
- ML Model {forecast_period}-Day Prediction: ${ml_pred_14d:.2f} ({((ml_pred_14d/current_price - 1) * 100):.2f}%)

PROPHET FORECAST:
- Prophet 7-Day Forecast: ${prophet_pred_7d:.2f} ({((prophet_pred_7d/current_price - 1) * 100):.2f}%)
- Prophet {forecast_period}-Day Forecast: ${prophet_pred_final:.2f} ({((prophet_pred_final/current_price - 1) * 100):.2f}%)

YOUR TASK:
Provide price predictions for days 3, 7, 14, and {forecast_period} (if different from 14).

For each prediction, provide:
1. Predicted price (be realistic and conservative)
2. Brief reasoning (2-3 sentences explaining your logic)

Format your response EXACTLY as JSON:
{{
  "predictions": [
    {{"day": 3, "price": 123.45, "reasoning": "Your reasoning here"}},
    {{"day": 7, "price": 124.50, "reasoning": "Your reasoning here"}},
    {{"day": 14, "price": 125.75, "reasoning": "Your reasoning here"}},
    {{"day": {forecast_period}, "price": 126.00, "reasoning": "Your reasoning here"}}
  ],
  "overall_reasoning": "A 2-3 sentence summary of your overall price outlook"
}}

FINAL REMINDER: Be REALISTIC and CONSERVATIVE. A $25 move in a day or even 14 days is EXTREMELY RARE for most stocks. Only predict such moves if you have EXCEPTIONAL, SPECIFIC evidence. When in doubt, predict smaller movements. Professional analysts prefer being slightly conservative than making unrealistic predictions."""

        print(f"Calling OpenAI for LLM price prediction for {ticker}...")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional stock analyst with 20+ years of experience. You provide realistic, conservative price predictions based on comprehensive analysis. You never over-calculate or make unrealistic predictions. You understand market volatility and uncertainty."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent, conservative predictions
            max_tokens=1500
        )
        
        response_text = completion.choices[0].message.content.strip()
        print(f"OpenAI LLM prediction response: {response_text[:500]}...")
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
                
                # Validate and clean predictions with strict realism checks
                validated_predictions = []
                for pred in result.get("predictions", []):
                    day = int(pred.get("day", 0))
                    price = float(pred.get("price", current_price))
                    reasoning = str(pred.get("reasoning", "No reasoning provided"))
                    
                    # Store original price for potential adjustments
                    original_price = price
                    
                    # Define realistic limits based on days
                    if day <= 3:
                        max_change_pct = 6.0  # Max 6% over 3 days (2% per day)
                    elif day <= 7:
                        max_change_pct = 7.0  # Max 7% over 7 days
                    elif day <= 14:
                        max_change_pct = 12.0  # Max 12% over 14 days
                    else:
                        max_change_pct = 15.0  # Max 15% for longer periods
                    
                    max_change_dollars = current_price * (max_change_pct / 100)
                    
                    # Calculate change
                    price_change_dollars = abs(price - current_price)
                    price_change_pct = abs(((price / current_price) - 1) * 100)
                    
                    # Check if prediction is unrealistic
                    adjustment_made = False
                    
                    if price_change_pct > max_change_pct or price_change_dollars > max_change_dollars:
                        print(f"WARNING: LLM prediction for day {day} is UNREALISTIC:")
                        print(f"  Predicted: ${price:.2f} (change: ${price_change_dollars:.2f}, {price_change_pct:.2f}%)")
                        print(f"  Realistic limit: ±${max_change_dollars:.2f} (±{max_change_pct:.1f}%)")
                        print(f"  Adjusting to realistic range...")
                        
                        # Calculate direction of prediction
                        direction = 1 if price > current_price else -1
                        
                        # Adjust to realistic limit (but stay conservative - use 80% of max)
                        adjusted_change_pct = max_change_pct * 0.8 * direction
                        price = current_price * (1 + adjusted_change_pct / 100)
                        adjustment_made = True
                        
                        # Update reasoning to reflect adjustment
                        reasoning = reasoning + f" [Note: Original prediction adjusted from ${original_price:.2f} to ${price:.2f} to reflect realistic market volatility limits.]"
                    
                    if adjustment_made:
                        # Recalculate after first adjustment
                        price_change_dollars = abs(price - current_price)
                        price_change_pct = ((price / current_price) - 1) * 100
                    
                    # Additional sanity check: reject if change is >$25 for typical stocks (unless stock is >$500)
                    if current_price < 500 and price_change_dollars > 25:
                        old_price = price
                        print(f"WARNING: Price change of ${price_change_dollars:.2f} is unrealistic for a ${current_price:.2f} stock over {day} days")
                        # Cap at $25 change or 10% of current price, whichever is smaller
                        max_absolute_change = min(25.0, current_price * 0.10)
                        direction = 1 if price > current_price else -1
                        price = current_price + (max_absolute_change * direction)
                        adjustment_made = True
                        reasoning = reasoning + f" [Further adjusted: ${price_change_dollars:.2f} absolute change exceeds realistic limits for this price level. Adjusted from ${old_price:.2f} to ${price:.2f}.]"
                        # Recalculate after second adjustment
                        price_change_dollars = abs(price - current_price)
                        price_change_pct = ((price / current_price) - 1) * 100
                    
                    if adjustment_made:
                        print(f"  Final adjusted price: ${price:.2f} (change: ${price_change_dollars:.2f}, {price_change_pct:.2f}%)")
                    
                    validated_predictions.append({
                        "day": day,
                        "price": round(price, 2),
                        "price_change_pct": round(price_change_pct, 2),
                        "price_change_dollars": round(price_change_dollars, 2),
                        "reasoning": reasoning
                    })
                
                # Sort by day
                validated_predictions.sort(key=lambda x: x["day"])
                
                return {
                    "predictions": validated_predictions,
                    "overall_reasoning": result.get("overall_reasoning", "Professional analysis based on comprehensive data review."),
                    "current_price": round(current_price, 2)
                }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM JSON response: {e}")
                print(f"Response was: {response_text}")
        
        # Fallback: try to extract prices from text
        print("Attempting to extract predictions from text...")
        fallback_predictions = []
        for day in [3, 7, 14, forecast_period]:
            if day <= forecast_period:
                # Use weighted average of ML and Prophet as fallback
                if day <= 7:
                    ml_pred = float(ml_predictions[ml_predictions['day'] == day]['prediction'].iloc[0]) if len(ml_predictions[ml_predictions['day'] == day]) > 0 else current_price
                    prophet_pred = float(prophet_forecast.iloc[day-1]['yhat']) if len(prophet_forecast) >= day else current_price
                    predicted_price = (ml_pred * 0.4 + prophet_pred * 0.6)  # Weight Prophet more
                else:
                    ml_pred = float(ml_predictions[ml_predictions['day'] == day]['prediction'].iloc[0]) if len(ml_predictions[ml_predictions['day'] == day]) > 0 else current_price
                    prophet_pred = float(prophet_forecast.iloc[day-1]['yhat']) if len(prophet_forecast) >= day else current_price
                    predicted_price = (ml_pred * 0.4 + prophet_pred * 0.6)
                
                # Apply strict conservative adjustment based on realistic limits
                change_pct = ((predicted_price / current_price) - 1) * 100
                change_dollars = abs(predicted_price - current_price)
                
                # Define realistic limits
                if day <= 3:
                    max_change_pct = 6.0  # Max 6% over 3 days
                elif day <= 7:
                    max_change_pct = 7.0  # Max 7% over 7 days
                elif day <= 14:
                    max_change_pct = 12.0  # Max 12% over 14 days
                else:
                    max_change_pct = 15.0  # Max 15% for longer
                
                # Cap absolute dollar change for lower-priced stocks
                max_change_dollars = min(25.0, current_price * 0.12) if current_price < 500 else current_price * 0.12
                
                # Apply conservative adjustments
                if abs(change_pct) > max_change_pct:
                    direction = 1 if change_pct > 0 else -1
                    predicted_price = current_price * (1 + (max_change_pct * 0.8 * direction / 100))  # Use 80% of max
                    print(f"Fallback: Adjusted day {day} prediction from {change_pct:.2f}% to {(max_change_pct * 0.8):.2f}% (realistic limit)")
                
                if change_dollars > max_change_dollars:
                    direction = 1 if predicted_price > current_price else -1
                    predicted_price = current_price + (max_change_dollars * 0.8 * direction)  # Use 80% of max
                    print(f"Fallback: Adjusted day {day} absolute change from ${change_dollars:.2f} to ${max_change_dollars * 0.8:.2f} (realistic limit)")
                
                final_change_pct = ((predicted_price / current_price) - 1) * 100
                final_change_dollars = abs(predicted_price - current_price)
                
                fallback_predictions.append({
                    "day": day,
                    "price": round(predicted_price, 2),
                    "price_change_pct": round(final_change_pct, 2),
                    "price_change_dollars": round(final_change_dollars, 2),
                    "reasoning": f"Conservative prediction based on ML and Prophet model consensus, adjusted to realistic market volatility limits (max {abs(final_change_pct):.1f}% change over {day} days)."
                })
        
        return {
            "predictions": fallback_predictions,
            "overall_reasoning": "Prediction based on weighted average of ML and Prophet models, with conservative adjustments for market uncertainty.",
            "current_price": round(current_price, 2),
            "error": "Could not parse LLM response, using fallback method"
        }
        
    except Exception as e:
        print(f"Error in LLM price prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Return very conservative fallback
        fallback_predictions = []
        for day in [3, 7, 14, forecast_period]:
            if day <= forecast_period:
                # Very conservative: assume minimal change (0.1-0.2% per day)
                daily_change_pct = 0.15  # Very conservative 0.15% per day
                predicted_price = current_price * (1 + (daily_change_pct * day / 100))
                
                # Ensure it's still within realistic bounds
                max_change_pct = min(12.0, day * 1.0)  # Max 1% per day, cap at 12% for 14 days
                if abs(((predicted_price / current_price) - 1) * 100) > max_change_pct:
                    direction = 1 if predicted_price > current_price else -1
                    predicted_price = current_price * (1 + (max_change_pct * direction / 100))
                
                final_change_pct = ((predicted_price / current_price) - 1) * 100
                final_change_dollars = abs(predicted_price - current_price)
                
                fallback_predictions.append({
                    "day": day,
                    "price": round(predicted_price, 2),
                    "price_change_pct": round(final_change_pct, 2),
                    "price_change_dollars": round(final_change_dollars, 2),
                    "reasoning": f"Very conservative estimate assuming minimal price movement ({abs(final_change_pct):.2f}% over {day} days) due to prediction system limitations."
                })
        
        return {
            "predictions": fallback_predictions,
            "overall_reasoning": "Conservative prediction due to system limitations.",
            "current_price": round(current_price, 2),
            "error": str(e)
        }

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper().strip()
        days_back = data.get('days_back', 365)
        forecast_period = data.get('forecast_period', 14)
        data_source = data.get('data_source', 'Alpha Vantage (Recommended)')
        
        # Always use at least 1 year of data
        min_days = 365
        if days_back < min_days:
            days_back = min_days
        
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Fetch stock data
        stock_data = cached_stock_data(ticker, start_date, end_date, data_source)
        if stock_data.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'"}), 400
        
        # Calculate technical indicators
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # Get fundamental data
        fundamental_data = get_fundamental_data(ticker, data_source)
        
        # Get sentiment data
        sentiment_data = get_stock_sentiment(ticker)
        
        # Prophet forecast
        df_prophet = stock_data.reset_index()
        if 'Date' not in df_prophet.columns:
            df_prophet = df_prophet.reset_index()
        if 'Date' in df_prophet.columns:
            df_prophet = df_prophet[["Date", "Close"]]
        else:
            df_prophet = df_prophet[["Close"]]
            df_prophet['Date'] = stock_data.index
        df_prophet.columns = ["ds", "y"]
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) > 30:
            prophet_model = Prophet(
                daily_seasonality="auto",
                yearly_seasonality="auto",
                weekly_seasonality="auto",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            prophet_model.fit(df_prophet)
            future = prophet_model.make_future_dataframe(periods=forecast_period)
            forecast_full = prophet_model.predict(future)
            # Only use future forecast values, not historical
            last_historical_date = df_prophet['ds'].max()
            forecast = forecast_full[forecast_full['ds'] > last_historical_date].copy()
        else:
            dates = pd.date_range(start=df_prophet['ds'].iloc[-1], periods=forecast_period+1, freq='D')[1:]
            trend = (df_prophet['y'].iloc[-1] - df_prophet['y'].iloc[0]) / len(df_prophet) if len(df_prophet) > 0 else 0
            forecast_values = [df_prophet['y'].iloc[-1] + trend * i for i in range(1, forecast_period+1)]
            forecast = pd.DataFrame({
                'ds': dates,
                'yhat': forecast_values,
                'yhat_lower': [v * 0.95 for v in forecast_values],
                'yhat_upper': [v * 1.05 for v in forecast_values]
            })
        
        # Train ML models
        ml_models, ml_predictions = train_ml_models(stock_data_with_indicators, forecast_days=forecast_period)
        
        # Get LLM price prediction with reasoning
        current_price = float(stock_data_with_indicators['Close'].iloc[-1])
        llm_prediction = get_llm_price_prediction(
            ticker=ticker,
            current_price=current_price,
            price_data=stock_data,
            technical_indicators=stock_data_with_indicators,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            ml_predictions=ml_predictions,
            prophet_forecast=forecast,
            forecast_period=forecast_period
        )
        
        # Get AI analysis
        ai_analysis = get_ai_analysis(ticker, stock_data_with_indicators, forecast, fundamental_data, sentiment_data)
        
        # Prepare response - convert dates to strings for JSON serialization
        stock_data_reset = stock_data.reset_index()
        if 'Date' not in stock_data_reset.columns:
            stock_data_reset = stock_data_reset.reset_index()
        stock_data_reset['Date'] = stock_data_reset['Date'].astype(str)
        
        stock_data_with_indicators_reset = stock_data_with_indicators.reset_index()
        if 'Date' not in stock_data_with_indicators_reset.columns:
            stock_data_with_indicators_reset = stock_data_with_indicators_reset.reset_index()
        stock_data_with_indicators_reset['Date'] = stock_data_with_indicators_reset['Date'].astype(str)
        
        forecast_reset = forecast.copy()
        forecast_reset['ds'] = forecast_reset['ds'].astype(str)
        
        # Ensure financial_health_score is always a number
        if fundamental_data.get('financial_health_score') is None:
            fundamental_data['financial_health_score'] = 0
        else:
            fundamental_data['financial_health_score'] = float(fundamental_data['financial_health_score'])
        
        # Ensure sentiment score is always a number
        if sentiment_data.get('score') is None:
            sentiment_data['score'] = 0.0
        else:
            sentiment_data['score'] = float(sentiment_data['score'])
        
        # Clean ML predictions - ensure no NaN values and proper structure
        ml_predictions_clean = ml_predictions.copy()
        for col in ml_predictions_clean.columns:
            ml_predictions_clean[col] = ml_predictions_clean[col].apply(lambda x: 0.0 if pd.isna(x) else float(x))
        
        # Ensure all required columns exist
        required_cols = ['day', 'prediction', 'mae', 'rmse', 'r2']
        for col in required_cols:
            if col not in ml_predictions_clean.columns:
                if col == 'day':
                    ml_predictions_clean[col] = range(1, len(ml_predictions_clean) + 1)
                else:
                    ml_predictions_clean[col] = 0.0
        
        # Ensure prediction values are reasonable (not NaN or None)
        if 'prediction' in ml_predictions_clean.columns:
            current_close = stock_data_with_indicators['Close'].iloc[-1] if len(stock_data_with_indicators) > 0 else 100
            ml_predictions_clean['prediction'] = ml_predictions_clean['prediction'].apply(
                lambda x: float(x) if not pd.isna(x) and x is not None and x > 0 else current_close
            )
        
        response = {
            "ticker": ticker,
            "stock_data": stock_data_reset.to_dict('records'),
            "stock_data_with_indicators": stock_data_with_indicators_reset.to_dict('records'),
            "fundamental_data": fundamental_data,
            "sentiment_data": sentiment_data,
            "prophet_forecast": forecast_reset.to_dict('records'),
            "ml_predictions": ml_predictions_clean.to_dict('records'),
            "llm_price_prediction": llm_prediction,
            "ai_analysis": ai_analysis
        }
        
        # Clean the entire response to handle any NaN values (convert to None for JSON)
        response = clean_for_json(response)
        
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather', methods=['POST'])
def get_weather_api():
    try:
        data = request.json
        city = data.get('city', 'Denver')
        weather_info = get_weather(city, OPENWEATHER_API_KEY)
        return jsonify({"weather": weather_info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({"status": "API is working"})


# Serve static files
@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.endswith('.css'):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path, mimetype='text/css')
    elif path.endswith('.js'):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path, mimetype='application/javascript')
    else:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), path)

if __name__ == '__main__':
    app.run(debug=True, port=5004)

