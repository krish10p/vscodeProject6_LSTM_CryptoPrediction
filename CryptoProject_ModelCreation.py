
# importing libraries
#import streamlit as st
import datareader as datareader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime, timedelta
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM # type: ignore
from keras.models import Sequential # type: ignore
from dateutil import parser
from pickle import dump
from pickle import load
from keras.models import load_model # type: ignore
import yfinance as yf
from copy import deepcopy
#np.random.seed(9)                                      # not doing anything since other seeds tf,keras needs to be fixed too to reproduce results
import os


# All utility functions:
def subtract_years(dt, years):
    """Subtract years from a date or datetime."""
    year = dt.year - years
    # if leap day and the new year is not leap, replace year and day
    # otherwise, only replace year
    if dt.month == 2 and dt.day == 29 and not isleap(year):
        return dt.replace(year=year, day=28)
    return dt.replace(year=year)







#Loading the data
global start                                            # start, end for whole dataframe for training/testing whole
global end
global crypto_coin
global currency

today = dt.datetime.now().date()
#start = subtract_years(today, years = 4).date()                         # training All models from last 4 years
start = dt.datetime(2021,1,1)    
today_plus_one = today + dt.timedelta(days=1)                        # Because, yf download excludes end date while downloading data
end = today_plus_one                       

global look_back
#global future_day                                                      # global one not useful in the current context of code
look_back = 60
future_day = 1                                                          # future day prediction =61st day prediction from learning past 60 days/look_back days:60/time_stamps_of_lstm:60 



# creates scalaed data and dumps scalar with scalar_name_model.pkl using MinMaxScalar
def custom_scaler(dataset, scaler_name):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(dataset.values.reshape(-1,1))
    #save the scaler
    dump(scaler, open(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_{scaler_name}_scaler.pkl','wb'))
    return data_scaled


# opens/loads saved scaler model in .pkl format
def scaler_opener(scaler_name):
    return load(open(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_{scaler_name}_scaler.pkl','rb'))
#train_scaler = scaler_opener('train')                                  # use function/open scaler when you need them
#test_scaler = scaler_opener('test')                                    

# #creating x_train and y_train from train_scaled using dataset_generator_lstm function

def dataset_generator_lstm(dataset, look_back=60):
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    dataX, dataY = np.array(dataX), np.array(dataY)
    return np.reshape(dataX,(dataX.shape[0],dataX.shape[1], 1)), dataY


cryptocoin_list = ['BTC','ETH','HOT1','RSR','NULS','NIM','AION','QASH','VITE','APL'] 
# excluding extra 10 coins since time to pretrain model taking too long['QRL','BCN','GBYTE','LBC','POA','PAC','ILC','BEPRO','GO','XMC']

#cryptocoin_list1 = ['BTC','ETH']
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

    today = dt.datetime.now()
    today_plus_one = today + dt.timedelta(days=1)                        # Because, yf download excludes end date while downloading data

    data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)
    data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_whole.csv')

    #prepare data
    print(data.shape)
    #print(data.head())

    #global split_percent
    split_percent = 0.8
    split = int(split_percent*len(data))
    train_data = data[:split]
    test_data = data[split:]

    train_scaled = custom_scaler(train_data['Close'],'train')
    test_scaled = custom_scaler(test_data['Close'],'test')

    # #creating x_train and y_train from train_scaled using dataset_generator_lstm function

    x_train, y_train = dataset_generator_lstm(train_scaled)
    x_test, y_test = dataset_generator_lstm(test_scaled)

    # Create the model :
    # Create Neural Network
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    lstm_model.add(Dropout(0.2))                                            #   We add Dropout layer to avoid overfitting of the model
    lstm_model.add(LSTM(units=50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=future_day))
    lstm_model.summary()

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)                 #keep epochs at least 10
    #save the lstm_model                                        # too big model for pickle to save, so use keras.models
    #dump(lstm_model, open('lstm_model.pkl','wb'))
    lstm_model.save(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                                # to save keras model and load the model

