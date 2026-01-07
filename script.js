// API Configuration - automatically detects environment
// For local development: uses localhost
// For production: uses the backend URL (update after deploying backend)
const API_BASE_URL = (() => {
    // Check if we're in production (hosted on Netlify or similar)
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        // Production: Replace 'YOUR-BACKEND-URL' with your actual Render backend URL
        // Example: 'https://stockforecastx-backend.onrender.com/api'
        // You can also set this via Netlify environment variable: API_BASE_URL
        const prodUrl = window.API_BASE_URL || 'https://YOUR-BACKEND-URL.onrender.com/api';
        
        // Show helpful error if placeholder URL is still being used
        if (prodUrl.includes('YOUR-BACKEND-URL')) {
            console.error('⚠️ Backend URL not configured! Please update script.js with your actual backend URL.');
        }
        
        return prodUrl;
    } else {
        // Development: Use localhost
        return 'http://localhost:5004/api';
    }
})();

let currentData = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    const forecastSlider = document.getElementById('forecastPeriod');
    const forecastValue = document.getElementById('forecastValue');
    
    forecastSlider.addEventListener('input', function() {
        forecastValue.textContent = `${this.value} days`;
    });
});

function closeWarning() {
    document.getElementById('warningBanner').style.display = 'none';
}

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Load tab-specific content if needed
    if (tabName === 'technical' && currentData) {
        loadTechnicalAnalysis();
    } else if (tabName === 'fundamental' && currentData) {
        loadFundamentalAnalysis();
    } else if (tabName === 'sentiment' && currentData) {
        loadSentimentAnalysis();
    } else if (tabName === 'ml' && currentData) {
        loadMLForecasts();
        loadLLMPredictions();
    } else if (tabName === 'ai' && currentData) {
        loadAIAnalysis();
    }
}

async function analyzeStock() {
    const ticker = document.getElementById('ticker').value.toUpperCase().trim();
    const timeframe = parseInt(document.getElementById('timeframe').value);
    const forecastPeriod = parseInt(document.getElementById('forecastPeriod').value);
    
    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }
    
    // Show loading
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ticker: ticker,
                days_back: timeframe,
                forecast_period: forecastPeriod,
                data_source: 'Alpha Vantage (Recommended)'
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentData = data;
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze stock');
        }
    } catch (error) {
        // Check if we're in production and backend URL is not configured
        const isProduction = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
        const backendNotConfigured = API_BASE_URL.includes('YOUR-BACKEND-URL');
        
        if (isProduction && backendNotConfigured) {
            showError(`Backend not configured! Please deploy your Flask backend to Railway/Render and update the API_BASE_URL in script.js. Current URL: ${API_BASE_URL}`);
        } else if (isProduction) {
            showError(`Cannot connect to backend server at ${API_BASE_URL}. Please ensure your Flask backend is deployed and running. Error: ${error.message}`);
        } else {
            showError(`Error: ${error.message}. Make sure the backend server is running on port 5004.`);
        }
    } finally {
        document.getElementById('loadingIndicator').style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    
    // Update metrics
    const stockData = data.stock_data;
    const currentPrice = stockData[stockData.length - 1].Close;
    const previousPrice = stockData[stockData.length - 2]?.Close || currentPrice;
    const priceChange = ((currentPrice / previousPrice) - 1) * 100;
    
    document.getElementById('currentPrice').textContent = `$${currentPrice.toFixed(2)}`;
    document.getElementById('currentPriceDelta').textContent = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`;
    document.getElementById('currentPriceDelta').style.color = priceChange >= 0 ? '#4caf50' : '#f44336';
    
    const forecastData = data.prophet_forecast;
    const forecastPrice = forecastData[forecastData.length - 1].yhat;
    const forecastChange = ((forecastPrice / currentPrice) - 1) * 100;
    
    document.getElementById('forecastPrice').textContent = `$${forecastPrice.toFixed(2)}`;
    document.getElementById('forecastPriceDelta').textContent = `${forecastChange >= 0 ? '+' : ''}${forecastChange.toFixed(2)}%`;
    document.getElementById('forecastPriceDelta').style.color = forecastChange >= 0 ? '#4caf50' : '#f44336';
    
    // Create gauges
    console.log('Financial Health Score:', data.fundamental_data.financial_health_score);
    console.log('Sentiment Data:', data.sentiment_data);
    createHealthGauge(data.fundamental_data.financial_health_score);
    createSentimentGauge(data.sentiment_data.score);
    
    // Create charts
    createPriceChart(data.stock_data, data.prophet_forecast);
    createVolumeChart(data.stock_data);
    
    // Display AI analysis
    document.getElementById('aiAnalysis').textContent = data.ai_analysis;
    
    // Display key metrics
    displayKeyMetrics(data.fundamental_data, data.stock_data_with_indicators);
    
    // Display news
    console.log('Sentiment data for news:', data.sentiment_data);
    displayNews(data.sentiment_data.articles || [], data.sentiment_data.error);
}

function createHealthGauge(score) {
    const gaugeData = [{
        type: "indicator",
        mode: "gauge+number",
        value: score,
        title: { text: "Financial Health Score" },
        gauge: {
            axis: { range: [0, 100] },
            bar: { color: "darkblue" },
            steps: [
                { range: [0, 30], color: "red" },
                { range: [30, 70], color: "yellow" },
                { range: [70, 100], color: "green" }
            ],
            threshold: {
                line: { color: "black", width: 4 },
                thickness: 0.75,
                value: score
            }
        }
    }];
    
    const layout = {
        height: 200,
        margin: { l: 20, r: 20, t: 50, b: 20 }
    };
    
    Plotly.newPlot('healthGauge', gaugeData, layout);
}

function createSentimentGauge(score) {
    // Ensure score is a valid number
    console.log('Creating sentiment gauge with score:', score, 'Type:', typeof score);
    const sentimentScore = (score !== undefined && score !== null) ? parseFloat(score) : 0;
    console.log('Parsed sentiment score:', sentimentScore);
    const normalizedScore = (sentimentScore + 1) * 50; // Convert -1 to 1 scale to 0 to 100
    console.log('Normalized score:', normalizedScore);
    
    // Ensure normalizedScore is within valid range
    const clampedScore = Math.max(0, Math.min(100, normalizedScore));
    console.log('Clamped score:', clampedScore);
    
    const gaugeData = [{
        type: "indicator",
        mode: "gauge+number",
        value: clampedScore,
        title: { text: "News Sentiment Score" },
        number: { suffix: "%", valueformat: ".1f" },
        gauge: {
            axis: { range: [0, 100] },
            bar: { color: "darkblue" },
            steps: [
                { range: [0, 40], color: "red" },
                { range: [40, 60], color: "yellow" },
                { range: [60, 100], color: "green" }
            ],
            threshold: {
                line: { color: "black", width: 4 },
                thickness: 0.75,
                value: clampedScore
            }
        }
    }];
    
    const layout = {
        height: 200,
        margin: { l: 20, r: 20, t: 50, b: 20 }
    };
    
    try {
        Plotly.newPlot('sentimentGauge', gaugeData, layout);
        console.log('Sentiment gauge plotted successfully');
    } catch (error) {
        console.error('Error plotting sentiment gauge:', error);
    }
}

function createPriceChart(stockData, forecastData) {
    const historicalDates = stockData.map(d => d.Date);
    const historicalPrices = stockData.map(d => d.Close);
    
    // Filter forecast to only show future dates
    const lastHistoricalDate = new Date(historicalDates[historicalDates.length - 1]);
    const futureForecast = forecastData.filter(d => new Date(d.ds) > lastHistoricalDate);
    const futureDates = futureForecast.map(d => d.ds);
    const futurePrices = futureForecast.map(d => d.yhat);
    const futureUpper = futureForecast.map(d => d.yhat_upper);
    const futureLower = futureForecast.map(d => d.yhat_lower);
    
    const trace1 = {
        x: historicalDates,
        y: historicalPrices,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Price',
        line: { color: 'blue', width: 2 }
    };
    
    const trace2 = {
        x: futureDates,
        y: futurePrices,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecast',
        line: { color: 'orange', width: 2 }
    };
    
    const trace3 = {
        x: [...futureDates, ...futureDates.slice().reverse()],
        y: [...futureUpper, ...futureLower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(255,165,0,0.2)',
        line: { color: 'rgba(255,165,0,0)' },
        name: 'Confidence Interval',
        showlegend: false
    };
    
    const layout = {
        title: 'Price History and Forecast',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        height: 400,
        hovermode: 'closest'
    };
    
    Plotly.newPlot('priceChart', [trace1, trace2, trace3], layout);
}

function createVolumeChart(stockData) {
    const dates = stockData.map(d => d.Date);
    const volumes = stockData.map(d => d.Volume);
    
    const trace = {
        x: dates,
        y: volumes,
        type: 'bar',
        name: 'Volume',
        marker: { color: 'darkblue' }
    };
    
    const layout = {
        title: 'Trading Volume',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Volume' },
        height: 250
    };
    
    Plotly.newPlot('volumeChart', [trace], layout);
}

function displayKeyMetrics(fundamentalData, stockDataWithIndicators) {
    const metricsContainer = document.getElementById('keyMetrics');
    const latestData = stockDataWithIndicators[stockDataWithIndicators.length - 1];
    
    const metrics = [
        { label: 'P/E Ratio', value: fundamentalData.pe_ratio || 'N/A' },
        { label: 'Dividend Yield', value: fundamentalData.dividend_yield ? `${fundamentalData.dividend_yield.toFixed(2)}%` : 'N/A' },
        { label: 'Market Cap', value: fundamentalData.market_cap ? `$${(fundamentalData.market_cap / 1e9).toFixed(2)}B` : 'N/A' },
        { label: 'Beta', value: fundamentalData.beta || 'N/A' },
        { label: '52 Week High', value: fundamentalData['52w_high'] ? `$${fundamentalData['52w_high']}` : 'N/A' },
        { label: '52 Week Low', value: fundamentalData['52w_low'] ? `$${fundamentalData['52w_low']}` : 'N/A' },
        { label: 'RSI (14)', value: latestData.RSI ? latestData.RSI.toFixed(2) : 'N/A' },
        { label: 'MACD', value: latestData.MACD_Line ? latestData.MACD_Line.toFixed(4) : 'N/A' }
    ];
    
    metricsContainer.innerHTML = metrics.map(metric => `
        <div class="metric-item">
            <label>${metric.label}</label>
            <value>${metric.value}</value>
        </div>
    `).join('');
}

function displayNews(articles, error) {
    const newsContainer = document.getElementById('newsItems');
    
    if (!newsContainer) {
        console.error('newsItems element not found');
        return '';
    }
    
    let html = '';
    
    if (error) {
        html += `<p class="error-message" style="margin-bottom: 16px;">Sentiment Analysis Note: ${error}</p>`;
    }
    
    if (!articles || articles.length === 0) {
        html += '<p>No recent news available.</p>';
        newsContainer.innerHTML = html;
        return html;
    }
    
    console.log(`Displaying ${articles.length} news articles`);
    
    html += articles.map(article => {
        const sentiment = (article.sentiment !== undefined && article.sentiment !== null) ? parseFloat(article.sentiment) : 0;
        const sentimentColor = sentiment > 0.2 ? '#4caf50' : sentiment < -0.2 ? '#f44336' : '#ff9800';
        const sentimentText = sentiment > 0.2 ? 'Positive' : sentiment < -0.2 ? 'Negative' : 'Neutral';
        
        console.log(`Article: ${article.title?.substring(0, 50)}... Sentiment: ${sentiment}`);
        
        return `
            <div class="news-item">
                <h4>${article.title || 'No title'}</h4>
                ${article.publisher ? `<p><strong>Source:</strong> ${article.publisher}</p>` : ''}
                <p><strong>Published:</strong> ${article.time_published || 'Unknown'}</p>
                <p><strong>Sentiment:</strong> <span style="color: ${sentimentColor}">${sentimentText} (${sentiment.toFixed(2)})</span></p>
                ${article.summary ? `<div style="margin-top: 12px; padding: 12px; background: #0a0a0a; border-radius: 6px; border-left: 3px solid ${sentimentColor};"><strong>Summary:</strong> <span style="color: #cccccc;">${article.summary}</span></div>` : ''}
                ${article.url ? `<a href="${article.url}" target="_blank" style="display: inline-block; margin-top: 12px;">Read Full Article →</a>` : ''}
            </div>
        `;
    }).join('');
    
    newsContainer.innerHTML = html;
    return html;
}

function loadTechnicalAnalysis() {
    if (!currentData) return;
    
    const stockData = currentData.stock_data_with_indicators;
    const dates = stockData.map(d => d.Date);
    
    // Create candlestick chart
    const trace1 = {
        x: dates,
        open: stockData.map(d => d.Open),
        high: stockData.map(d => d.High),
        low: stockData.map(d => d.Low),
        close: stockData.map(d => d.Close),
        type: 'candlestick',
        name: 'Price'
    };
    
    // Add technical indicators
    const traces = [trace1];
    
    if (stockData[0].SMA_50) {
        traces.push({
            x: dates,
            y: stockData.map(d => d.SMA_50),
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 50',
            line: { color: 'orange' }
        });
    }
    
    if (stockData[0].EMA_21) {
        traces.push({
            x: dates,
            y: stockData.map(d => d.EMA_21),
            type: 'scatter',
            mode: 'lines',
            name: 'EMA 21',
            line: { color: 'green' }
        });
    }
    
    const layout = {
        title: 'Price with Technical Indicators',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        height: 600
    };
    
    Plotly.newPlot('technicalChart', traces, layout);
    
    // Create oscillators chart
    createOscillatorsChart(stockData, dates);
    
    // Display technical signals
    displayTechnicalSignals(stockData);
}

function createOscillatorsChart(stockData, dates) {
    const rsi = stockData.map(d => d.RSI);
    const macdLine = stockData.map(d => d.MACD_Line);
    const macdSignal = stockData.map(d => d.MACD_Signal);
    
    const trace1 = {
        x: dates,
        y: rsi,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI',
        line: { color: 'blue' },
        yaxis: 'y'
    };
    
    const trace2 = {
        x: dates,
        y: new Array(dates.length).fill(70),
        type: 'scatter',
        mode: 'lines',
        name: 'Overbought',
        line: { color: 'red', dash: 'dash' },
        yaxis: 'y'
    };
    
    const trace3 = {
        x: dates,
        y: new Array(dates.length).fill(30),
        type: 'scatter',
        mode: 'lines',
        name: 'Oversold',
        line: { color: 'green', dash: 'dash' },
        yaxis: 'y'
    };
    
    const trace4 = {
        x: dates,
        y: macdLine,
        type: 'scatter',
        mode: 'lines',
        name: 'MACD Line',
        line: { color: 'blue' },
        yaxis: 'y2'
    };
    
    const trace5 = {
        x: dates,
        y: macdSignal,
        type: 'scatter',
        mode: 'lines',
        name: 'MACD Signal',
        line: { color: 'orange' },
        yaxis: 'y2'
    };
    
    const layout = {
        title: 'Technical Oscillators',
        height: 800,
        yaxis: { title: 'RSI', domain: [0.66, 1] },
        yaxis2: { title: 'MACD', domain: [0, 0.33], anchor: 'x', overlaying: 'y' }
    };
    
    Plotly.newPlot('oscillatorsChart', [trace1, trace2, trace3, trace4, trace5], layout);
}

function displayTechnicalSignals(stockData) {
    const latest = stockData[stockData.length - 1];
    const previous = stockData[stockData.length - 2];
    
    const signals = [];
    
    // Moving Average Signals
    if (latest.SMA_50 && latest.SMA_200) {
        if (latest.SMA_50 > latest.SMA_200) {
            signals.push({ signal: 'BULLISH', description: 'Golden Cross (SMA 50 > SMA 200)' });
        } else {
            signals.push({ signal: 'BEARISH', description: 'Death Cross (SMA 50 < SMA 200)' });
        }
    }
    
    // RSI Signals
    if (latest.RSI) {
        if (latest.RSI > 70) {
            signals.push({ signal: 'BEARISH', description: `Overbought (RSI = ${latest.RSI.toFixed(2)})` });
        } else if (latest.RSI < 30) {
            signals.push({ signal: 'BULLISH', description: `Oversold (RSI = ${latest.RSI.toFixed(2)})` });
        } else {
            signals.push({ signal: 'NEUTRAL', description: `RSI in normal range (${latest.RSI.toFixed(2)})` });
        }
    }
    
    // MACD Signals
    if (latest.MACD_Line && latest.MACD_Signal && previous) {
        if (latest.MACD_Line > latest.MACD_Signal && previous.MACD_Line <= previous.MACD_Signal) {
            signals.push({ signal: 'BULLISH', description: 'MACD Bullish Crossover' });
        } else if (latest.MACD_Line < latest.MACD_Signal && previous.MACD_Line >= previous.MACD_Signal) {
            signals.push({ signal: 'BEARISH', description: 'MACD Bearish Crossover' });
        }
    }
    
    // Display signals
    const signalsContainer = document.getElementById('technicalSignals');
    signalsContainer.innerHTML = `
        <table class="signal-table">
            <thead>
                <tr>
                    <th>Signal</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                ${signals.map(s => `
                    <tr class="signal-${s.signal.toLowerCase()}">
                        <td>${s.signal}</td>
                        <td>${s.description}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function loadFundamentalAnalysis() {
    if (!currentData) return;
    
    const fundamental = currentData.fundamental_data;
    const container = document.getElementById('fundamentalContent');
    
    container.innerHTML = `
        <div class="section">
            <h4>${fundamental.name || currentData.ticker}</h4>
            <p>${fundamental.sector || 'N/A'} | ${fundamental.industry || 'N/A'}</p>
            <p><strong>Market Cap:</strong> $${fundamental.market_cap ? (fundamental.market_cap / 1e9).toFixed(2) + 'B' : 'N/A'}</p>
        </div>
        <div class="metrics-grid">
            <div class="metric-item"><label>P/E Ratio</label><value>${fundamental.pe_ratio || 'N/A'}</value></div>
            <div class="metric-item"><label>Forward P/E</label><value>${fundamental.forward_pe || 'N/A'}</value></div>
            <div class="metric-item"><label>PEG Ratio</label><value>${fundamental.peg_ratio || 'N/A'}</value></div>
            <div class="metric-item"><label>Price to Book</label><value>${fundamental.price_to_book || 'N/A'}</value></div>
            <div class="metric-item"><label>EPS</label><value>${fundamental.eps ? '$' + fundamental.eps : 'N/A'}</value></div>
            <div class="metric-item"><label>Beta</label><value>${fundamental.beta || 'N/A'}</value></div>
            <div class="metric-item"><label>Dividend Yield</label><value>${fundamental.dividend_yield ? fundamental.dividend_yield.toFixed(2) + '%' : 'N/A'}</value></div>
            <div class="metric-item"><label>Debt to Equity</label><value>${fundamental.debt_to_equity || 'N/A'}</value></div>
            <div class="metric-item"><label>Return on Equity</label><value>${fundamental.return_on_equity ? fundamental.return_on_equity.toFixed(2) + '%' : 'N/A'}</value></div>
            <div class="metric-item"><label>Profit Margins</label><value>${fundamental.profit_margins ? fundamental.profit_margins.toFixed(2) + '%' : 'N/A'}</value></div>
            <div class="metric-item"><label>Revenue Growth</label><value>${fundamental.revenue_growth ? fundamental.revenue_growth.toFixed(2) + '%' : 'N/A'}</value></div>
            <div class="metric-item"><label>52-Week Range</label><value>$${fundamental['52w_low'] || 'N/A'} - $${fundamental['52w_high'] || 'N/A'}</value></div>
        </div>
    `;
}

function loadSentimentAnalysis() {
    if (!currentData) return;
    
    const sentiment = currentData.sentiment_data;
    const container = document.getElementById('sentimentContent');
    
    if (!sentiment) {
        container.innerHTML = '<p>No sentiment data available.</p>';
        return;
    }
    
    const sentimentScore = sentiment.score !== undefined ? sentiment.score : 0;
    const sentimentColor = sentimentScore > 0.2 ? 'green' : sentimentScore < -0.2 ? 'red' : 'orange';
    const articles = sentiment.articles || [];
    const error = sentiment.error;
    
    let html = `
        <div class="section">
            <h4>Overall Sentiment Score: <span style="color: ${sentimentColor}">${sentimentScore.toFixed(2)}</span></h4>
            <p>(-1 = Very Negative, 0 = Neutral, 1 = Very Positive)</p>
    `;
    
    if (error) {
        html += `<p style="color: orange;">Note: ${error}</p>`;
    }
    
    html += `</div>`;
    
    if (articles.length > 0) {
        // Generate news HTML
        const newsHtml = articles.map(article => {
            const sentiment = article.sentiment || 0;
            const sentimentColor = sentiment > 0.2 ? 'green' : sentiment < -0.2 ? 'red' : 'orange';
            const sentimentText = sentiment > 0.2 ? 'Positive' : sentiment < -0.2 ? 'Negative' : 'Neutral';
            
            return `
                <div class="news-item">
                    <h4>${article.title || 'No title'}</h4>
                    ${article.publisher ? `<p><strong>Source:</strong> ${article.publisher}</p>` : ''}
                    <p><strong>Published:</strong> ${article.time_published || 'Unknown'}</p>
                    <p><strong>Sentiment:</strong> <span style="color: ${sentimentColor}">${sentimentText} (${sentiment.toFixed(2)})</span></p>
                    ${article.summary ? `<div style="margin-top: 12px; padding: 12px; background: #0a0a0a; border-radius: 6px; border-left: 3px solid ${sentimentColor};"><strong>Summary:</strong> <span style="color: #cccccc;">${article.summary}</span></div>` : ''}
                    ${article.url ? `<a href="${article.url}" target="_blank" style="display: inline-block; margin-top: 12px;">Read Full Article →</a>` : ''}
                </div>
            `;
        }).join('');
        
        html += `
            <div class="section">
                <h4>Recent News Articles (${articles.length})</h4>
                <div id="sentimentNewsItems">${newsHtml}</div>
            </div>
        `;
    } else {
        html += `
            <div class="section">
                <h4>Recent News Articles</h4>
                <p>No articles available at this time.</p>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function loadMLForecasts() {
    if (!currentData) {
        console.error('No current data available');
        return;
    }
    
    const mlPredictions = currentData.ml_predictions;
    const stockData = currentData.stock_data;
    
    if (!mlPredictions || !Array.isArray(mlPredictions) || mlPredictions.length === 0) {
        const container = document.getElementById('mlContent');
        if (container) {
            container.innerHTML = '<div class="section"><h4>ML Forecasts</h4><p>No ML predictions available. The model may need more historical data.</p></div>';
        }
        console.error('ML predictions data is missing or invalid');
        return;
    }
    
    if (!stockData || !Array.isArray(stockData) || stockData.length === 0) {
        console.error('Stock data is missing or invalid');
        return;
    }
    
    const currentPrice = stockData[stockData.length - 1]?.Close;
    if (!currentPrice) {
        console.error('Current price not available');
        return;
    }
    
    const container = document.getElementById('mlContent');
    if (!container) {
        console.error('mlContent container not found');
        return;
    }
    
    try {
        // Filter out any null/undefined predictions
        const validPredictions = mlPredictions.filter(p => p && p.day !== undefined && p.prediction !== undefined && p.prediction !== null);
        
        if (validPredictions.length === 0) {
            container.innerHTML = '<div class="section"><h4>ML Forecasts</h4><p>No valid ML predictions available.</p></div>';
            return;
        }
        
        // Create ML predictions chart
        const predictionDays = validPredictions.map(p => p.day);
        const predictionPrices = validPredictions.map(p => parseFloat(p.prediction) || 0);
        
        // Get historical data for chart
        const historicalData = stockData.slice(-30);
        const historicalDates = historicalData.map(d => d.Date);
        const historicalPrices = historicalData.map(d => parseFloat(d.Close) || 0);
        
        const trace1 = {
            x: historicalDates,
            y: historicalPrices,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Price',
            line: { color: 'blue', width: 2 }
        };
        
        const trace2 = {
            x: predictionDays,
            y: predictionPrices,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'ML Predictions',
            line: { color: 'orange', width: 2 }
        };
        
        const layout = {
            title: 'ML Model Price Predictions',
            xaxis: { title: 'Days Ahead' },
            yaxis: { title: 'Price ($)' },
            height: 400
        };
        
        // Display predictions table first
        container.innerHTML = `
            <div id="mlChart"></div>
            <div class="section">
                <h4>Detailed Predictions</h4>
                <table class="signal-table">
                    <thead>
                        <tr>
                            <th>Day</th>
                            <th>Predicted Price</th>
                            <th>Current Price</th>
                            <th>Change (%)</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>R²</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${validPredictions.map(p => {
                            const predPrice = parseFloat(p.prediction) || 0;
                            const mae = parseFloat(p.mae) || 0;
                            const rmse = parseFloat(p.rmse) || 0;
                            const r2 = parseFloat(p.r2) || 0;
                            const change = ((predPrice / currentPrice) - 1) * 100;
                            return `
                                <tr>
                                    <td>${p.day || 'N/A'}</td>
                                    <td>$${predPrice.toFixed(2)}</td>
                                    <td>$${currentPrice.toFixed(2)}</td>
                                    <td>${change >= 0 ? '+' : ''}${change.toFixed(2)}%</td>
                                    <td>${mae.toFixed(2)}</td>
                                    <td>${rmse.toFixed(2)}</td>
                                    <td>${r2.toFixed(3)}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        // Plot the chart after DOM update
        setTimeout(() => {
            try {
                Plotly.newPlot('mlChart', [trace1, trace2], layout);
            } catch (plotError) {
                console.error('Error plotting ML chart:', plotError);
            }
        }, 100);
    } catch (error) {
        console.error('Error in loadMLForecasts:', error);
        if (container) {
            container.innerHTML = `<div class="section"><h4>ML Forecasts</h4><p>Error loading ML predictions: ${error.message}</p></div>`;
        }
    }
}

function loadLLMPredictions() {
    if (!currentData) {
        console.error('No current data available');
        return;
    }
    
    const llmPrediction = currentData.llm_price_prediction;
    
    if (!llmPrediction) {
        console.log('LLM prediction data not available');
        return;
    }
    
    const container = document.getElementById('mlContent');
    if (!container) {
        console.error('mlContent container not found');
        return;
    }
    
    try {
        const predictions = llmPrediction.predictions || [];
        const overallReasoning = llmPrediction.overall_reasoning || 'No overall reasoning provided.';
        const currentPrice = llmPrediction.current_price || (currentData.stock_data && currentData.stock_data[currentData.stock_data.length - 1]?.Close) || 0;
        
        if (predictions.length === 0) {
            // Add LLM section even if no predictions
            const existingContent = container.innerHTML;
            container.innerHTML = existingContent + `
                <div class="section" style="margin-top: 40px; border-top: 2px solid #333; padding-top: 30px;">
                    <h4>🤖 Professional LLM Price Predictions (with Reasoning)</h4>
                    <p>No LLM predictions available at this time.</p>
                    ${llmPrediction.error ? `<p class="error-message">Error: ${llmPrediction.error}</p>` : ''}
                </div>
            `;
            return;
        }
        
        // Create LLM predictions HTML
        const llmSection = `
            <div class="section" style="margin-top: 40px; border-top: 2px solid #333; padding-top: 30px;">
                <h4>🤖 Professional LLM Price Predictions (with Reasoning)</h4>
                <p style="margin-bottom: 20px; color: #888; font-style: italic;">${overallReasoning}</p>
                
                <div id="llmChart" style="margin-bottom: 30px;"></div>
                
                <h5 style="margin-top: 30px; margin-bottom: 15px;">Detailed Predictions with Reasoning</h5>
                <div class="llm-predictions-container">
                    ${predictions.map(pred => {
                        const price = parseFloat(pred.price) || 0;
                        const changePct = parseFloat(pred.price_change_pct) || 0;
                        const day = parseInt(pred.day) || 0;
                        const reasoning = pred.reasoning || 'No reasoning provided.';
                        const changeColor = changePct >= 0 ? '#4CAF50' : '#f44336';
                        const changeIcon = changePct >= 0 ? '📈' : '📉';
                        
                        return `
                            <div class="llm-prediction-card" style="background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h5 style="margin: 0; color: #fff;">Day ${day} Prediction</h5>
                                    <div style="text-align: right;">
                                        <div style="font-size: 1.5em; font-weight: bold; color: ${changeColor};">
                                            $${price.toFixed(2)}
                                        </div>
                                        <div style="color: ${changeColor}; font-size: 0.9em;">
                                            ${changeIcon} ${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%
                                        </div>
                                    </div>
                                </div>
                                <div style="background: #0f0f0f; padding: 15px; border-radius: 6px; border-left: 3px solid #4CAF50;">
                                    <strong style="color: #4CAF50; display: block; margin-bottom: 8px;">Professional Reasoning:</strong>
                                    <p style="margin: 0; color: #ccc; line-height: 1.6;">${reasoning}</p>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
        // Append LLM section to existing content
        container.innerHTML = container.innerHTML + llmSection;
        
        // Create chart for LLM predictions
        setTimeout(() => {
            try {
                const stockData = currentData.stock_data;
                const historicalData = stockData.slice(-30);
                const historicalDates = historicalData.map(d => d.Date);
                const historicalPrices = historicalData.map(d => parseFloat(d.Close) || 0);
                
                const predictionDays = predictions.map(p => `Day ${p.day}`);
                const predictionPrices = predictions.map(p => parseFloat(p.price) || 0);
                
                const trace1 = {
                    x: historicalDates,
                    y: historicalPrices,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical Price',
                    line: { color: '#2196F3', width: 2 }
                };
                
                const trace2 = {
                    x: predictionDays,
                    y: predictionPrices,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'LLM Professional Predictions',
                    line: { color: '#4CAF50', width: 3, dash: 'dash' },
                    marker: { size: 10, color: '#4CAF50' }
                };
                
                const trace3 = {
                    x: [`Current (Day 0)`, ...predictionDays],
                    y: [currentPrice, ...predictionPrices],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Prediction Trend',
                    line: { color: '#4CAF50', width: 2, dash: 'dot' },
                    showlegend: false
                };
                
                const layout = {
                    title: {
                        text: 'Professional LLM Price Predictions',
                        font: { color: '#fff', size: 16 }
                    },
                    xaxis: { 
                        title: 'Time Period',
                        gridcolor: '#333',
                        color: '#fff'
                    },
                    yaxis: { 
                        title: 'Price ($)',
                        gridcolor: '#333',
                        color: '#fff'
                    },
                    plot_bgcolor: '#1a1a1a',
                    paper_bgcolor: '#0f0f0f',
                    font: { color: '#fff' },
                    height: 400,
                    legend: {
                        x: 0,
                        y: 1,
                        bgcolor: 'rgba(0,0,0,0.5)',
                        bordercolor: '#333',
                        borderwidth: 1
                    }
                };
                
                Plotly.newPlot('llmChart', [trace1, trace3, trace2], layout);
            } catch (plotError) {
                console.error('Error plotting LLM chart:', plotError);
            }
        }, 200);
        
    } catch (error) {
        console.error('Error in loadLLMPredictions:', error);
        const existingContent = container.innerHTML;
        container.innerHTML = existingContent + `
            <div class="section" style="margin-top: 40px; border-top: 2px solid #333; padding-top: 30px;">
                <h4>🤖 Professional LLM Price Predictions</h4>
                <p class="error-message">Error loading LLM predictions: ${error.message}</p>
            </div>
        `;
    }
}

function loadAIAnalysis() {
    if (!currentData) return;
    
    const container = document.getElementById('aiContent');
    container.innerHTML = `
        <div class="section">
            <h4>Comprehensive AI Analysis</h4>
            <div class="analysis-text">${currentData.ai_analysis}</div>
        </div>
    `;
}

async function getWeather() {
    const city = document.getElementById('weatherCity').value || 'Denver';
    const weatherInfo = document.getElementById('weatherInfo');
    
    try {
        const response = await fetch(`${API_BASE_URL}/weather`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ city: city })
        });
        
        const data = await response.json();
        weatherInfo.textContent = data.weather || 'Unable to fetch weather data';
    } catch (error) {
        weatherInfo.textContent = `Error: ${error.message}`;
    }
}


function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

