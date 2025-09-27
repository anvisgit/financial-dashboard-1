 #Financial Analysis and Stock Prediction

## Overview
Streamlit-based application for financial analysis and stock prediction.  
Integrates Yahoo Finance, Finnhub API, Prophet, XGBoost, and sentiment analysis.  
Provides visualization, prediction, financial data, and sentiment insights.

## Features

### Home Page
- Basic introduction placeholder.

### Graphical Analysis
- Fetch historical stock data from Yahoo Finance.
- Line charts for Close price and Volume.
- Histograms for Close and Volume.
- Pie chart for volume distribution by date.
- Candlestick chart for price movements.
- 7-day Moving Average chart.
- Daily Return (%) chart.
- Option to download CSV of fetched data.

### Options and Futures
- Placeholder (under development).

### Headlines
- Fetches company news from Finnhub API.
- Displays latest headlines with source and link.

### Financial Information
- Displays financials, cash flow, and balance sheet from Yahoo Finance.

### Stock Price Prediction
- Uses Facebook Prophet model.
- Trains on past 200 days of Close price.
- Predicts next 40 days.
- Interactive prediction chart (Plotly).
- Shows predicted values with confidence intervals.

### Live Data
- Fetches real-time stock quotes from Finnhub API.
- Displays current price, daily high, and daily low.

### Sentiment Analysis
- Fetches last 60 days of news headlines from Finnhub API.
- Analyzes sentiment using TextBlob.
- Calculates average sentiment score.
- Classifies sentiment: Positive / Negative / Neutral.
- Displays histogram of sentiment score distribution.
- Provides interpretation of sentiment score.

## Dependencies
Python libraries required:
- yfinance
- streamlit
- pandas
- numpy
- datetime
- plotly
- matplotlib
- scikit-learn
- xgboost
- ta
- finnhub-python
- prophet
- textblob
