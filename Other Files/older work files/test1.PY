
# import pandas_datareader.data as web
# import datetime as dt

# start = dt.datetime(2020, 1, 1)
# end = dt.datetime(2020, 12, 31)

# df = web.DataReader('AAPL', 'yahoo', start, end)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import os


# aapl_df = yf.download('AAPL', 
#                       start='2019-01-01', 
#                       end='2023-02-03', 
#                       progress=False,
# )
# aapl_df.head()
# print(aapl_df.head())

# start_date='2024-05-01'
# end_date='2024-05-03'
# dates = pd.date_range(start_date, end_date, freq='D')
# print((dates))
# l1 = [3,53.1,5.23]
# l2 = [62.2,5213.234321,2]

# n1 = np.array(l1)
# n2 = np.array(l2)

cryptocoin_list = ['BTC','ETH']
for i in cryptocoin_list:
    crypto_coin = i 
    #crypto_coin = 'BTC'
    currency = 'CAD'

    # specify the path for the directory – make sure to surround it with quotation marks
    path = f'./data_modelCreation/{crypto_coin}'
    # check whether directory already exists
    if not os.path.exists(path):
        os.mkdir(path)
        print("Folder %s created!" % path)
    else:
        print("Folder %s already exists" % path)

    start = dt.datetime(2021,1,1)                           # start, end for whole dataframe for training/testing whole
    today = dt.datetime.now()
    today_plus_one = today + dt.timedelta(days=1)                        # Because, yf download excludes end date while downloading data
    end = today_plus_one

    data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)
    data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_whole.csv')