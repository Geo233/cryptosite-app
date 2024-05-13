from django.contrib.sites import requests
from django.shortcuts import render
import re
import praw
from django.http import JsonResponse
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from .predictii import main_prediction_function
nltk.download('vader_lexicon')
from .forms import TimePeriodForm
import yfinance as yf
import plotly.graph_objects as go
import json
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
FIRST_CALL_AFTER_RESTART = True
def home(request):
    import requests
    import json

    #Crypto Prices
    price_request = requests.get("https://min-api.cryptocompare.com"
                                 "/data/pricemultifull?fsyms="
                                 "BTC,ETH,TUSD,USDT,BUSD,USDC,XRP,BNB,SOL,ARB&tsyms=EUR")
    price = json.loads(price_request.content)

    #Crypto News
    api_request = requests.get("https://min-api.cryptocompare.com"
                               "/data/v2/news/?lang=EN")
    api = json.loads(api_request.content)


    return render(request, 'home.html', {'api': api, 'price': price})

def prices(request):
    import requests
    import json
    if request.method == 'POST':
        quote = request.POST['quote']
        quote = quote.upper()
        crypto_request = requests.get("https://min-api.cryptocompare.com/data/pricemultifull?fsyms="+quote+"&tsyms=EUR")
        news_request = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        crypto = json.loads(crypto_request.content)
        news = json.loads(news_request.content)
        filtered_news = [item for item in news['Data'] if quote in item['body'].upper() or quote in item['title'].upper()]
        return render(request, 'prices.html', {'quote': quote, 'crypto': crypto, 'news' : filtered_news })
    else:
        return render(request, 'prices.html', {})


# def predictions(request):
#
#
#
#     return render(request, 'predictions.html', context)


def predictions(request):
    global FIRST_CALL_AFTER_RESTART

    import requests
    import json

    keywords = ['BTC', 'BITCOIN', 'SATOSHI', 'ELON']
    # Get the news
    news_request = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
    news = json.loads(news_request.content)


    def contains_keyword(item):
        return any(keyword in item['title'].upper()
                   or keyword in item['body'].upper()
                   for keyword in keywords)

    filtered_news = [item for item in news['Data']
                     if contains_keyword(item)]

    sid = SentimentIntensityAnalyzer()
    rescaled_scores = []

    def classify_sentiment(score):
        compound = score['compound']
        rescaled_score = int((compound + 1) / 2 * 100)
        rescaled_scores.append(rescaled_score)

        if compound > 0.05:
            sentiment = "Positive"
        elif compound < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return f"{sentiment} ({rescaled_score})"

    scores = [(item['title'], classify_sentiment(sid.polarity_scores(item['title'])))
              for item in filtered_news]

    avg_score = sum(rescaled_scores) / len(rescaled_scores)\
        if rescaled_scores else 0

    # Make a request to the cryptocompare API for historical price data
    hist_request = requests.get("https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=EUR&limit=2000")
    hist = json.loads(hist_request.content)

    # Extract dates and prices
    dates = [datetime.utcfromtimestamp(item['time']).strftime('%Y-%m-%d') for item in hist['Data']['Data']]
    prices = [item['close'] for item in hist['Data']['Data']]
    context = {'dates': dates, 'prices': prices, 'scores': scores, 'avg_score': avg_score}

    if FIRST_CALL_AFTER_RESTART:
        price = main_prediction_function()
        predicted_price = price[0]
        diff_value = price[1]
        last_day = price[2]
        request.session['predicted_price'] = predicted_price
        request.session['diff_value'] = diff_value
        request.session['last_price'] = last_day
        FIRST_CALL_AFTER_RESTART = False
    else:
        predicted_price = request.session['predicted_price']
        diff_value = request.session['diff_value']
        last_day = request.session['last_price']

    context['price'] = predicted_price
    context['difference'] = diff_value
    context['last_price'] = last_day


    return render(request, 'predictions.html', context)


def get_data(ticker="BTC-USD", days=194):
    data = yf.download(ticker, period=f"{days}d")
    return data


def charts(request):
    days = 30  # default value
    indicators = []
    ticker = 'BTC-USD'


    if request.method == "POST":
        days = request.POST.get('days', 30)
        indicators = request.POST.get('indicators', 'MACD')
        ticker = request.POST.get('ticker', 'BTC-USD')

    data = get_data(ticker=str(ticker), days=int(days))
    rsi_data = get_data(ticker=str(ticker), days=int(days) + 1)
    william_data = get_data(ticker=str(ticker), days=int(days) + 14)
    adx_data = get_data(ticker=str(ticker), days=int(days) + 26)

    # Setăm configurațiile subplot-ului
    # row_heights = [0.5]  # default height pentru graficul de lumânări
    # for _ in indicators:
    #     row_heights.append(0.5 / len(indicators))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.02)

    # Graficul de lumânări
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlesticks"
    ), row=1, col=1)


    if "MACD" in indicators:
        data = compute_macd(data)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line'), row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)


    # Putem adăuga aici și alți indicatori în viitor, la fel ca MACD-ul
    if "RSI" in indicators:
        data = compute_rsi(rsi_data)
        display_data = rsi_data.iloc[1:]
        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['RSI'], mode='lines', name='RSI', line=dict(color='royalblue')),
                      row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)

    if "WilliamR" in indicators:
        data = william_r(william_data)
        display_data = william_data.iloc[14:]
        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['WilliamR'], mode='lines', name='William %R'), row=2, col=1)
        fig.update_yaxes(title_text="William %R", row=2, col=1)


    if "ADX" in indicators:
        data = compute_adx(adx_data)
        display_data = adx_data.iloc[26:]
        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['ADX'], mode='lines', name='ADX'), row=2, col=1)
        fig.update_yaxes(title_text="ADX", row=2, col=1)
    # Configurăm aspectul general

    fig.update_xaxes(rangeslider_visible=False)


    fig.update_layout(
        width=1200,
        height=800,
    )

    chart = fig.to_html(full_html=False)

    return render(request, 'charts.html', {'chart': chart,
                                           'indicators': indicators,
                                           "days": days,
                                           'ticker': ticker})

def compute_macd(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0)).fillna(0)
    loss = (-delta.where(delta < 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def william_r(data, window=14):
     data['Highest␣High'] = data['High'].rolling(window=window).max()
     data['Lowest␣Low'] = data['Low'].rolling(window=window).min()
     data['WilliamR'] = ((data['Highest␣High'] - data['Close']) /
     (data['Highest␣High'] - data['Lowest␣Low'])) * -100
     return data

def compute_adx(data, period=14):
    delta_high = data['High'] - data['High'].shift(1)
    delta_low = data['Low'].shift(1) - data['Low']

    TR = data['High'] - data['Low']
    TR = pd.Series(np.where((data['High'] - data['Close'].shift(1)) > TR, data['High'] - data['Close'].shift(1), TR),
                   index=data.index)
    TR = pd.Series(np.where((data['Close'].shift(1) - data['Low']) > TR, data['Close'].shift(1) - data['Low'], TR),
                   index=data.index)

    data['+DM'] = np.where(((delta_high > 0) & (delta_high > delta_low)), delta_high, 0)
    data['-DM'] = np.where(((delta_low > 0) & (delta_low > delta_high)), delta_low, 0)

    data['+DM_MA14'] = data['+DM'].rolling(window=period).sum()
    data['-DM_MA14'] = data['-DM'].rolling(window=period).sum()
    data['TR_MA14'] = TR.rolling(window=period).sum()

    data['+DI14'] = 100 * (data['+DM_MA14'] / data['TR_MA14'])
    data['-DI14'] = 100 * (data['-DM_MA14'] / data['TR_MA14'])

    data['Dx'] = 100 * abs(data['+DI14'] - data['-DI14']) / (data['+DI14'] + data['-DI14'])
    data['ADX'] = data['Dx'].rolling(window=period).mean()

    drop_columns = ['+DM', '-DM', '+DM_MA14', '-DM_MA14', 'TR_MA14', '+DI14', '-DI14', 'Dx']
    data = data.drop(columns=drop_columns, errors='ignore')

    return data