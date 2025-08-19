# StockForecastX Pro - Advanced AI Stock Analysis Platform

A comprehensive stock analysis and forecasting application built with Streamlit, featuring AI-powered insights, technical analysis, fundamental analysis, sentiment analysis, and machine learning predictions.

## Features

### **Comprehensive Analysis**
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Fundamental Analysis**: P/E ratios, financial health scores, valuation metrics
- **Sentiment Analysis**: Real-time news sentiment scoring
- **Machine Learning**: Random Forest models with performance metrics
- **AI Insights**: Hugging Face-powered analysis with multiple model options

### **Key Capabilities**
- Real-time stock data fetching via Yahoo Finance
- Advanced technical indicators calculation
- Prophet time series forecasting
- News sentiment analysis via Alpha Vantage
- Weather integration for market context
- Interactive charts and visualizations
- Downloadable analysis reports

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd Stock_DEV
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys (Optional but Recommended)**
   
   Create a `.env` file in the project directory:
   ```bash
   # Hugging Face API Token (for AI analysis)
   HF_TOKEN=your_huggingface_token_here
   
   # Alpha Vantage API Key (for news sentiment)
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   
   # OpenWeather API Key (for weather data)
   OPENWEATHER_API_KEY=your_openweather_key_here
   
   # Finnhub API Key (alternative data source)
   FINNHUB_API_KEY=your_finnhub_key_here
   ```

   **Note**: The application will work without API keys, but some features will be limited.

4. **Run the application**
   ```bash
   streamlit run PART.PY
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## Usage Guide

### 1. **Basic Analysis**
- Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- Select your desired timeframe (1 month to 5 years)
- Choose forecast period (1-60 days)
- Click "Analyze Stock"

### 2. **AI Model Selection**
- Choose from available Hugging Face models:
  - **GPT-OSS-120B**: Large, powerful model for detailed analysis
  - **Llama-3.1-8B**: Fast, efficient model
  - **Mistral-7B**: Balanced performance and speed
  - **CodeLlama-34B**: Specialized for technical analysis

### 3. **Analysis Tabs**

#### **Dashboard**
- Overview of current price and forecast
- Financial health and sentiment gauges
- Main price chart with forecast
- Volume analysis
- AI-generated insights

#### **Technical Analysis**
- Interactive candlestick charts
- Technical indicators overlay
- Oscillator charts (RSI, MACD, Stochastic)
- Technical signals and pivot points
- Support/resistance levels

#### **Fundamental Analysis**
- Company financial metrics
- Valuation score breakdown
- Financial health components
- Investment checklist
- Radar charts for metrics

#### **Sentiment Analysis**
- News sentiment scoring
- Recent news articles with sentiment
- Sentiment impact analysis
- Market mood indicators

#### **ML Forecasts**
- Machine learning model performance
- Prediction accuracy metrics
- Detailed price predictions
- Model confidence levels

#### **AI Insights**
- Comprehensive AI analysis
- Investment recommendations
- Risk factor assessment
- Score breakdown by component

## API Keys Setup

### **Hugging Face Token**
1. Visit [Hugging Face](https://huggingface.co/)
2. Create an account and log in
3. Go to Settings → Access Tokens
4. Create a new token with read permissions
5. Add to your `.env` file as `HF_TOKEN`

### **Alpha Vantage API Key**
1. Visit [Alpha Vantage](https://www.alphavantage.co/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add to your `.env` file as `ALPHA_VANTAGE_API_KEY`

### **OpenWeather API Key**
1. Visit [OpenWeatherMap](https://openweathermap.org/)
2. Sign up for a free account
3. Get your API key
4. Add to your `.env` file as `OPENWEATHER_API_KEY`

## Features in Detail

### **Caching System**
- Stock data cached for 1 hour
- Fundamental data cached for 30 minutes
- Sentiment data cached for 15 minutes
- Improves performance for repeated analyses

### **Error Handling**
- Graceful handling of API failures
- Fallback analysis when AI models are unavailable
- Input validation for ticker symbols
- Progress tracking for long operations

### **Performance Optimizations**
- Optimized Prophet model settings
- Efficient data processing
- Memory management
- Responsive UI with progress indicators

## Troubleshooting

### **Common Issues**

1. **"No data found for ticker"**
   - Check if the ticker symbol is correct
   - Ensure you have internet connection
   - Try a different ticker symbol

2. **API errors**
   - Verify your API keys are correct
   - Check if you've exceeded API limits
   - The app will use fallback analysis if APIs fail

3. **Slow performance**
   - First analysis may be slow due to data fetching
   - Subsequent analyses will be faster due to caching
   - Reduce forecast period for faster processing

4. **Installation issues**
   - Ensure Python 3.8+ is installed
   - Try upgrading pip: `pip install --upgrade pip`
   - Install dependencies one by one if needed

### **Performance Tips**
- Use shorter timeframes for faster analysis
- Reduce forecast period for quicker results
- Close other applications to free up memory
- Use the caching system effectively

## Disclaimer

This application is for educational and informational purposes only. The analysis and predictions provided should not be considered as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions. Past performance does not guarantee future results, and all investments carry risk.

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing performance

## License

This project is open source and available under the MIT License.

## Support

For support or questions:
- Check the troubleshooting section
- Review the documentation
- Open an issue on the repository

---
