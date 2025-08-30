import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import ta
import finnhub
import prophet 
from prophet.plot import plot_plotly
from textblob import TextBlob 


finapi=finnhub.Client(api_key="d2or3v1r01qnhraoaqv0d2or3v1r01qnhraoaqvg")

st.set_page_config(page_title="Stock Stuff", page_icon="💸", layout="wide")

st.title("Financial Analysis and Stock Prediction")
tab0,tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home Page",
    "Graphical Analysis",
    "Options and Futures",
    "Headlines",
    " Financial Information",
    "Stock Price Prediction",
    "Live data",
    "Sentiment Analysis"
])

with tab0:
    st.write("bs intro")
with tab1:
    st.header("📈 Graphical Analysis")

    ticker_input = st.text_input("Enter the company's ticker symbol:").upper()
    if ticker_input:
        today = datetime.date.today()
        with st.form("form_graph"):
            col1, col2 = st.columns(2)
            with col1:
                date_inps = st.date_input("Start date:")
            with col2:
                date_inpe = st.date_input("End date:")
            submitted = st.form_submit_button("PROCEED!")

        if submitted:
            try:
                stck = yf.Ticker(ticker_input)
                df = stck.history(start=str(date_inps), end=str(date_inpe))

                if df.empty:
                    st.warning("No data found for the selected period.")
                else:
                    st.success(f"Data successfully loaded for {ticker_input}")

                    df_reset = df.reset_index()
                    csv_filename = f"{ticker_input}_data.csv"
                    df.to_csv(csv_filename)
                    with open(csv_filename, "rb") as f:
                        st.download_button("🧙🏽‍♂️ Download CSV", f, file_name=csv_filename)

                    st.subheader("Line Charts📈")
                    st.line_chart(df["Close"], use_container_width=True)
                    st.line_chart(df["Volume"], use_container_width=True)

                    st.subheader("Histogram of Close & Volume📈")
                    st.bar_chart(df["Close"])
                    st.bar_chart(df["Volume"])

                    st.subheader("Volume Distribution (Pie Chart)📈")
                    pie_fig = px.pie(df_reset, names="Date", values="Volume", title="Volume by Date")
                    st.plotly_chart(pie_fig, use_container_width=True)

                    st.subheader("Candlestick Chart📈")
                    candle_fig = go.Figure(data=[go.Candlestick(
                        x=df_reset["Date"],
                        open=df_reset["Open"],
                        high=df_reset["High"],
                        low=df_reset["Low"],
                        close=df_reset["Close"],
                    )])
                    candle_fig.update_layout(
                        title=f"{ticker_input} Candlestick Chart",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(candle_fig, use_container_width=True)

                    st.subheader("7-Day Moving Average📈")
                    df["MA7"] = df["Close"].rolling(window=7).mean()
                    st.line_chart(df[["Close", "MA7"]])

                    st.subheader("Daily Return (%)📈")
                    df["Daily Return (%)"] = df["Close"].pct_change() * 100
                    st.line_chart(df["Daily Return (%)"])

            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab2:
    st.header("📊 Options and Futures")
    st.warning("underwork")
with tab3:
    st.header("📰 Headlines")
    ticker_input = st.text_input("Enter the company's ticker symbol:", key="headline").upper()

    if ticker_input:
        news = finapi.company_news(ticker_input, _from="2025-08-20", to="2025-08-29")

        if news:
            for article in news[:5]:
                st.subheader(article["headline"])
                st.write(f"Source: {article['source']}")
                st.markdown(f"[Read more]({article['url']})")
        else:
            st.warning("No news available for this ticker.")

with tab4:
    st.header("💰 Financial Information")
    ticker_input = st.text_input("Enter the company's ticker symbol:", key="fininfo").upper()
    if ticker_input:
        today = datetime.date.today()
        with st.form("form_fin"):
            col1, col2 = st.columns(2)
            with col1:
                date_inps = st.date_input("Start Date:")
            with col2:
                date_inpe = st.date_input("End Date:")
            submitted = st.form_submit_button("PROCEED!")

        if submitted:
            stck = yf.Ticker(ticker_input)

            st.subheader("Financials")
            if not stck.financials.empty:
                st.dataframe(stck.financials)
            else:
                st.warning("No financial data available.")

            st.subheader("Cash Flow")
            if not stck.cashflow.empty:
                st.dataframe(stck.cashflow)

            st.subheader("Balance Sheet")
            if not stck.balance_sheet.empty:
                st.dataframe(stck.balance_sheet)
with tab5:
    st.header("Stock Price Prediction")
    tickerin=st.text_input("Enter the ticker symbol of the company:").upper()
    if tickerin:
        end=datetime.date.today()
        start=end-datetime.timedelta(days=200)
        stk=yf.Ticker(tickerin)
        df = yf.download(tickerin, start=start, end=end)
        data = df.reset_index()[["Date", "Close"]]
        data.columns=["ds","y"]
        model=prophet.Prophet(daily_seasonality=True)
        model.fit(data)
        predicted=model.make_future_dataframe(periods=40)
        predictedst=model.predict(predicted)
        st.subheader("Predicted data")
        fig = plot_plotly(model, predictedst)
        st.plotly_chart(fig)
        st.write(predictedst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())



with tab6:
    st.header("🧙🏽‍♂️Live data")
    ticker = st.text_input("Enter a company's ticker symbol:", "").upper()

    if ticker:
        quote = finapi.quote(ticker)
        st.metric(label=f"{ticker} Current Price (USD)", value=quote["c"])
        st.metric(label=f"{ticker} High of the day (USD)", value=quote["h"])
        st.metric(label=f"{ticker} Low of the day (USD)", value=quote["h"])

with tab7:
    st.header("Sentiment Analysis")
    ticker=st.text_input("Enter the companies ticker symbol:").upper()
    if ticker:
        end=datetime.date.today()
        start= datetime.date.today()-datetime.timedelta(days =60)
        news = finapi.company_news(ticker, _from=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"))
        score=[]
        for i in news:
            hl=i['headline']
            blob=TextBlob(hl)
            score.append(blob.sentiment.polarity)#-1to1
        if score:
            finalscore=sum(score)/len(score)
            print("Sentiment score for", ticker, ":", finalscore)

            if finalscore==0:
                st.write("Neutral")
            elif finalscore<0:
                st.write("Negative")
            else:
                st.write("Positive")
    
        fig, px = plt.subplots()
        px.hist(score, bins=10, color="blue", alpha=0.7)
        px.set_title(f"Sentiment distribution for {ticker}")
        px.set_xlabel("Sentiment score")
        px.set_ylabel("Frequency")

        st.pyplot(fig)
        st.write('''Sentiment scores measure the general tone of news, articles, or social media posts about a stock, helping investors gauge market sentiment.
        A positive sentiment score indicates optimism, suggesting that the information portrays the stock or company favorably, such as beating earnings expectations or announcing growth opportunities.
        A negative sentiment score reflects pessimism, signaling unfavorable news like missed revenue targets, legal issues, or declining performance, 
        whereas a neutral sentiment represents factual or indifferent content, such as routine announcements or data reports, which neither imply optimism nor pessimism.
        By analyzing multiple headlines or posts, these scores can be aggregated to calculate an overall sentiment or net sentiment score, 
        which helps investors quickly understand whether the market perception of a stock is generally positive, negative, or neutral. ''' )


    

            










