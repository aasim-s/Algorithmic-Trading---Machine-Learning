from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

# 1) Download/Load SP500 stocks prices data
sp500 = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")
symbols_list = sp500["Symbol"].unique().tolist()

end_date = pd.to_datetime("today")
start_date = end_date-pd.DateOffset(years=8)

df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
df.index.names = ["date", "ticker"]
df.columns = df.columns.str.lower()

# 2) Calculate features and technical indicators
# Garman-Klass Volatility
df["garman_klass_vol"] = ((np.log(df["high"])-np.log(df["low"])) **
                          2/2-(2*np.log(2)-1)*(np.log(df["adj close"])-np.log(df["open"]))**2)

# RSI
df["RSI"] = df.groupby(level=1)["adj close"].transform(
    lambda x: pandas_ta.rsi(close=x, length=20))

# Bollinger Bands
df["bb_low"] = df.groupby(level=1)["adj close"].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df["bb_mid"] = df.groupby(level=1)["adj close"].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df["bb_high"] = df.groupby(level=1)["adj close"].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])

# ATR


def compute_atr(data):
    atr = pandas_ta.atr(
        high=data["high"], low=data["low"], close=data["close"], length=14)

    return atr.sub(atr.mean()).div(atr.std())


df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)

# MACD


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]

    return macd.sub(macd.mean()).div(macd.std())


df["macd"] = df.groupby(level=1, group_keys=False)[
    "adj close"].apply(compute_macd)

# Dollar Volume
df["dollar_volume"] = df["adj close"]*df["volume"]/1e6

print(df)
