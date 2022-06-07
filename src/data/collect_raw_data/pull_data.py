import math
import os.path
import time
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client
from dateutil import parser

# API
binance_api_key = '[REDACTED]'  # Enter your own API-key here.
binance_api_secret = '[REDACTED]'  # Enter your own API-secret here.

# CONSTANTS
binsizes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440, "1w": 10080, "1M": 43200}
batch_size = 750
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
HEADER = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']


# FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data):
    old = None
    new = None
    if len(data) > 0:
        old = parser.parse(data["timestamp"].iloc[-1]) + timedelta(0, 0, 0, 0, binsizes[kline_size])
    else:
        old = datetime.strptime('01 Sep 2017', '%d %b %Y')

    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    new = datetime.strptime('01 Jun 2022 23 59 59', '%d %b %Y %H %M %S')

    print(old, new)
    return old, new


def get_all_binance(symbol, kline_size, save=False):
    filename = f'{symbol}-{kline_size}-data_trimmed.csv'
    print(filename)

    data_df = pd.DataFrame()
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)

    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])

    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'):
        print(f'Downloading all available {kline_size} data for {symbol}. Be patient..!')
    else:
        print(f'Downloading {delta_min} minutes of new data available for {symbol}, i.e. {available_data} instances of '
              f'{kline_size} data.')

    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                  newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns=HEADER)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else:
        data_df = data
    data_df.set_index('timestamp', inplace=True)

    if save:
        data_df.iloc[:, 0:5].to_csv('D:\\uncertainty_sharpe\\evidential_deep_learning\\data\\' + filename)
    print('All caught up..!')

    return data_df


symbol_set = ['BTCUSDT']
for symbol in symbol_set:
    get_all_binance(symbol, '1h', save=True)
