import yfinance as yf
import csv



def fetch_bitcoin_data():
    btc = yf.Ticker("BTC-EUR")
    hist = btc.history(period="max")
    return hist


def save_to_csv(hist, filename='bitcoin_prices_yahoo.csv'):
    hist.to_csv(filename)


if __name__ == '__main__':
    hist = fetch_bitcoin_data()
    save_to_csv(hist)