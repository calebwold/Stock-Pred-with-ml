# StockForecastX Pro - Advanced AI Stock Analysis Platform

A comprehensive stock analysis and forecasting web application built with Flask, featuring AI-powered insights, technical analysis, fundamental analysis, sentiment analysis, and machine learning predictions.

## 🚀 Features

### **Comprehensive Analysis**
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, OBV, and more
- **Fundamental Analysis**: P/E ratios, financial health scores, valuation metrics
- **Sentiment Analysis**: Real-time news sentiment scoring with AI-generated debriefs
- **Machine Learning**: Random Forest models with performance metrics
- **Prophet Forecasting**: Time series forecasting for price predictions
- **LLM Price Predictions**: AI-powered conservative price forecasts with detailed reasoning

### **Key Capabilities**
- Real-time stock data fetching via Alpha Vantage and yfinance
- Advanced technical indicators calculation
- Prophet time series forecasting
- News sentiment analysis via OpenAI
- Interactive charts and visualizations using Plotly.js
- Responsive design for mobile and desktop
- Professional, sleek UI design

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- API Keys (see setup instructions below)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Stock_DEV
```

### 2. Create Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the project root directory:

```bash
# Alpha Vantage API Key (for historical stock data and fundamentals)
# Get your free key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# OpenAI API Key (for sentiment analysis and LLM price predictions)
# Get your key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# OpenWeather API Key (optional, for weather data)
# Get your free key at: https://openweathermap.org/api
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

**Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

### 5. Run the Application
```bash
python3 app.py
```

The server will start on `http://localhost:5004` (or the port specified in `app.py`).

### 6. Open in Browser
Navigate to `http://localhost:5004` in your web browser.

## 📖 Usage Guide

### Basic Analysis
1. Enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `TSLA`, `GOOGL`)
2. Select your desired timeframe (1 Year, 2 Years, or 5 Years)
3. Adjust forecast days using the slider (1-60 days)
4. Click "Analyze Stock"
5. Wait for analysis to complete (30-60 seconds)

### Analysis Tabs

#### **Dashboard**
- Overview of current price and forecast
- Financial health and sentiment gauges
- Main price chart with forecast
- Volume analysis
- Key metrics at a glance

#### **Technical Analysis**
- Interactive candlestick charts
- Technical indicators overlay
- RSI, MACD, Bollinger Bands
- Support/resistance levels

#### **Fundamental Analysis**
- Company financial metrics
- Financial health score (0-100)
- Valuation metrics
- P/E, PEG, Debt-to-Equity ratios

#### **Sentiment Analysis**
- News sentiment scoring (-1 to 1 scale)
- Recent news articles with AI-generated debriefs
- Sentiment impact analysis
- Publisher information

#### **ML Forecasts**
- Machine learning model performance
- Prediction accuracy metrics
- Detailed price predictions
- LLM-based price predictions with reasoning
- Prophet forecasts

#### **AI Insights**
- Comprehensive AI analysis
- Investment recommendations
- Risk factor assessment

## 🔑 API Keys Setup

### **Alpha Vantage API Key**
1. Visit [Alpha Vantage](https://www.alphavantage.co/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add to your `.env` file as `ALPHA_VANTAGE_API_KEY`

**Note**: Alpha Vantage is used **only** for historical stock data and fundamental data, not for sentiment analysis.

### **OpenAI API Key**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and add billing information
3. Go to API Keys section
4. Create a new API key
5. Add to your `.env` file as `OPENAI_API_KEY`

**Note**: OpenAI is used for sentiment analysis of news articles and LLM-based price predictions.

### **OpenWeather API Key** (Optional)
1. Visit [OpenWeatherMap](https://openweathermap.org/)
2. Sign up for a free account
3. Get your API key
4. Add to your `.env` file as `OPENWEATHER_API_KEY`

## 🏗️ Project Structure

```
Stock_DEV/
├── app.py              # Flask backend application
├── index.html          # Frontend HTML
├── styles.css          # CSS styling
├── script.js           # Frontend JavaScript
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not in git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔒 Security Notes

- **Never commit API keys** to version control
- The `.env` file is automatically ignored by git
- API keys are loaded from environment variables
- Use `.env.example` as a template (create your own `.env` file)

## 🐛 Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Check if the ticker symbol is correct
   - Ensure you have internet connection
   - Verify your Alpha Vantage API key is set correctly

2. **API errors**
   - Verify your API keys are correct in `.env` file
   - Check if you've exceeded API rate limits
   - The app will use fallback methods when possible

3. **Port already in use**
   - Change the port in `app.py`: `app.run(port=5004)`
   - Update `API_BASE_URL` in `script.js` to match

4. **Module not found errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

5. **Sentiment analysis not working**
   - Verify OpenAI API key is set correctly
   - Check your OpenAI account has available credits
   - News articles are fetched from yfinance (free, no API key needed)

## 📊 Features in Detail

### **Data Sources**
- **Alpha Vantage**: Historical stock data and fundamental data
- **yfinance**: News articles (free, no API key needed)
- **OpenAI**: Sentiment analysis and LLM price predictions

### **ML Models**
- **Random Forest Regressor**: For price predictions
- **Prophet**: Time series forecasting
- **LLM (GPT-4)**: Conservative price predictions with reasoning

### **Technical Indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- OBV (On-Balance Volume)
- And more...

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. The analysis and predictions provided should **not** be considered as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions. Past performance does not guarantee future results, and all investments carry risk.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing performance

## 📧 Support

For support or questions:
- Check the troubleshooting section above
- Review the documentation
- Open an issue on the repository

---

**Built with ❤️ using Flask, Python, and AI**
