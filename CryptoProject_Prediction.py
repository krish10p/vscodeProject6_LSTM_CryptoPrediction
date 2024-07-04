
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
#np.random.seed(9)                                       # not doing anything since other seeds tf,keras needs to be fixed too to reproduce results
import os
from calendar import isleap


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


cryptocoin_list = ['BTC','ETH','HOT1','RSR','NULS','NIM','AION','QASH','VITE','APL']
# excluding extra 10 coins since time taking to pretrain models longer, ['QRL','BCN','GBYTE','LBC','POA','PAC','ILC','BEPRO','GO','XMC']
currency_list = ['CAD']     # ['CAD','USD','GBP','EUR','INR', 'AUD','JPY','CNY']
                            # Avoiding other Currencies since there will be further more combinatios for models to train on
crypto_coin = 'BTC' 
currency = 'CAD'

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

# creating recursive dataset, which will be just 1 sample for model prediction
def recursive_dataset_generator(dataset, look_back=60):
    dataX = []
    
    dataX.append(dataset)
    dataX = np.array(dataX)
    return np.reshape(dataX,(dataX.shape[0],dataX.shape[1], 1))

# predict function to predict crypto prices between selected dates and plot the graph for actual vs predicted prices, & print actual and predicted prices
def predict(dt0, dt1):

    today = dt.datetime.now().date()
    dt0 = dt.datetime.strptime(dt0, '%Y-%m-%d').date()
    dt1 = dt.datetime.strptime(dt1, '%Y-%m-%d').date()
    dt1_plus_one = dt1 + dt.timedelta(days=1)                            # cause yf download exclude end date in downloading
    dt3 = dt0 - timedelta(days=look_back)
    
    # #if dt2 <= today :
    if (dt0 > dt1) or (dt0 > today) or (dt1 > today):
        print('The chosen recursive start date is in past than recursive end date or recursive start or end date is after today, check your dates')
        return

    custom_test_data = yf.download(f'{crypto_coin}-{currency}', start=dt3, end=dt1_plus_one, progress=False,)
    custom_test_data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_custom_test_dates.csv')

    custom_test_scaled_data = custom_scaler(custom_test_data['Close'],'custom_test')
    custom_x_test, custom_y_test = dataset_generator_lstm(custom_test_scaled_data)
    lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model
    custom_test_scaler = scaler_opener('custom_test')
    lstm_model.evaluate(custom_x_test, custom_y_test)                   # use results of evaluation if needed on custom test set
 
    prediction_custom_test = lstm_model.predict(custom_x_test,batch_size=1)
    prediction_custom_test = custom_test_scaler.inverse_transform(prediction_custom_test)
    actual_custom_test = custom_test_scaler.inverse_transform(custom_test_scaled_data)

    #print('Regualr/Non-Recursive Prediction and Actual Prices for ' + crypto_coin + ' from ' + str(dt0) + ' to ' + str(dt1) + ' as following:')
   
    # Ploting with dates: Predict the prices using model on custom test data and plot the actual vs predicted prices graph
    dates = pd.date_range(end =dt1, periods=len(custom_test_data)-look_back, freq='D')

    plt.figure(figsize=(16,8))
    plt.plot(dates, prediction_custom_test.reshape(-1,), marker = '.', color='green', label='Prediction Prices')
    plt.plot(dates, actual_custom_test[look_back:,].reshape(-1,), marker = '.', color='orange',label='Ground Truth Prices')
    plt.title(f'{crypto_coin} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'Price in {currency}')
    plt.legend(loc='upper right')
    plt.show()

    # creating df, to print actual, predicted prices for custom_test daterange:
    df = pd.DataFrame({'dates':dates,'Prediction':prediction_custom_test.reshape(-1,),'Actual':actual_custom_test[look_back:,].reshape(-1,)})
    df.set_index('dates',inplace=True)
    print('Prediction and Actual Prices for ' + crypto_coin + ' with selected date from ' + str(dt0) + ' to ' + str(dt1) + ' as following:')
    print(df)

    return True

# recursive_predict1 : predicts recursive predictions from dt4 to dt2
def recursive_predict1(dt4,dt2):

    dt2 = dt.datetime.strptime(dt2, '%Y-%m-%d').date()
    today = dt.datetime.now().date()
    today_plus_one = today + dt.timedelta(days=1)
    dt4 = dt.datetime.strptime(dt4, '%Y-%m-%d').date()
    delta = abs((dt2-dt4).days)

    if (dt2 <= dt4) or (dt4>today_plus_one):
        print('The chosen recursive start date is in past than recursive end date or recursive start date must be before or equal to today_plus_one')
        return

    today = dt.datetime.now().date()
    delta = abs((dt2-dt4).days) + 1

    # preparing recursive_custom_x_test[-1] :
    
    dt3 = dt4 - timedelta(days=look_back)
    
    recursive_custom_test_data = yf.download(f'{crypto_coin}-{currency}', start=dt3, end=dt4, progress=False,)
    recursive_custom_test_data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_recursive_custom_test_dates.csv')

    recursive_custom_test_scaled_data = custom_scaler(recursive_custom_test_data['Close'],'recursive_custom_test')
    recursive_custom_x_test = recursive_dataset_generator(recursive_custom_test_scaled_data)
    lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model
    recursive_custom_test_scaler = scaler_opener('recursive_custom_test')
 
    
    # recursive prediction code
    recursive_predictions = []
    lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model

    last_window = deepcopy(recursive_custom_x_test[-1])
    for i in range(delta):
        temp1 = np.array([])
        temp = deepcopy(last_window)
        next_prediction = lstm_model.predict(np.array([temp]).reshape((1,look_back,1))).flatten()
        recursive_predictions.append(next_prediction)
        temp1 = np.vstack([temp[1:], next_prediction.reshape((1,1))])
        last_window = deepcopy(temp1)
    recursive_custom_test_scaler = scaler_opener('recursive_custom_test') 
    recursive_predictions = recursive_custom_test_scaler.inverse_transform(recursive_predictions)
    data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)          # whole data from yf in data df to plot

    #print('Recursive Prediction and Actual Prices for ' + crypto_coin + ' from ' + str(dt4) + ' to ' + str(dt2) + ' as following:')
    
    # plotting recursive_predictions and actual values
    overlap = ((today - dt4).days + 1)
    periods = len(data) + delta - overlap
    if (len(data) + delta - overlap) < len(data): periods = len(data)

    custom_test_dates = pd.date_range(start=start, periods= periods, freq='D')
    
    if (dt2>today and dt4<=today_plus_one) or (dt2<=today and dt4<= today_plus_one):
        rec_start_index = len(data) - overlap
        rec_end_index = rec_start_index + len(recursive_predictions)

    plt.figure(figsize=(16,8))
    plt.plot(custom_test_dates[:len(data)], np.array(data.loc[:,'Close']), marker = '.', color='orange',label='Actual Prices')
    plt.plot(custom_test_dates[rec_start_index:rec_end_index], recursive_predictions[:,], marker = '.', color='blue',label='Recursive Predictions Prices')
    plt.title(f'{crypto_coin} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'Price in {currency}')
    plt.legend(loc='upper left')
    plt.show()
    
    return True

# recursive_predict2 : predicts recursive predictions from dt4 to dt2, AND regular/non recursive predictions from dt4 to dt2
def recursive_predict2(dt4,dt2):

    dt2 = dt.datetime.strptime(dt2, '%Y-%m-%d').date()
    today = dt.datetime.now().date()
    today_plus_one = today + dt.timedelta(days=1)
    dt4 = dt.datetime.strptime(dt4, '%Y-%m-%d').date()
    delta = abs((dt2-dt4).days)

    if (dt2 <= dt4) or (dt4 > today_plus_one):
        print('The chosen recursive start date must be before recursive end date or recursive start date must be before or equal to today_plus_one')
        return

    #----------------------------------------Preparing prediction_custom_test---------------------------------------------------
    dt0 = dt4
    dt1 = dt2

    today = dt.datetime.now().date()
    #dt0 = dt.datetime.strptime(dt0, '%Y-%m-%d').date()
    #dt1 = dt.datetime.strptime(dt1, '%Y-%m-%d').date()
    dt1_plus_one = dt1 + dt.timedelta(days=1)                            # cause yf download exclude end date in downloading
    dt3 = dt0 - timedelta(days=look_back)
    
    # #if dt2 <= today :
    if (dt1 <= dt0) or (dt0 > today_plus_one) :
        print('The chosen recursive start date must be before recursive end date or recursive start date must be before or equal to today_plus_one')
        return
    if (dt1 > today_plus_one):
        dt1 = today_plus_one
    # To avoid downloading extra/incorrect data, so this if is for safery measure:
    if (dt0 == today_plus_one): 
        dt1_plus_one = today_plus_one

    custom_test_data = yf.download(f'{crypto_coin}-{currency}', start=dt3, end=dt1_plus_one, progress=False,)
    custom_test_data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_custom_test_dates.csv')

    custom_test_scaled_data = custom_scaler(custom_test_data['Close'],'custom_test')
    prediction_custom_test = np.array([])
    if (dt0 != today_plus_one) :
        custom_x_test, custom_y_test = dataset_generator_lstm(custom_test_scaled_data)
        lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model
        custom_test_scaler = scaler_opener('custom_test')
        lstm_model.evaluate(custom_x_test, custom_y_test)                   # use results of evaluation if needed on custom test set
    
        prediction_custom_test = lstm_model.predict(custom_x_test,batch_size=1)
        prediction_custom_test = custom_test_scaler.inverse_transform(prediction_custom_test)
        actual_custom_test = custom_test_scaler.inverse_transform(custom_test_scaled_data)

        # print('Prediction and Actual Prices for ' + crypto_coin + ' with selected date from ' + str(dt0) + ' to ' + str(dt1) + ' as following:')
    
        # Ploting with dates: Predict the prices using model on custom test data and plot the actual vs predicted prices graph
        dates = pd.date_range(dt0, periods=len(custom_test_data)-look_back, freq='D')
        # Following chunk working as expected witout any need for removal of the chunk of code
        # plt.figure(figsize=(16,8))
        # plt.plot(dates, prediction_custom_test.reshape(-1,), marker = '.', color='green', label='Prediction Prices')
        # plt.plot(dates, actual_custom_test[look_back:,].reshape(-1,), marker = '.', color='orange',label='Ground Truth Prices')
        # plt.title(f'{crypto_coin} Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel(f'Price in {currency}')
        # plt.legend(loc='upper right')
        # plt.show()

    #-------------------------------------------------------------------------------------------
    #Adding a missing prediction for last input of: prediction_custom_test if dt4==today or today_plus_one, 
    #which will be unique, so adding manually, lstm_model.predict(custom_x_test_unique)
    if (dt2 > today_plus_one) :

        unique_custom_test_data = custom_test_data[-look_back:]
        unique_custom_test_scaled_data = custom_scaler(unique_custom_test_data['Close'],'unique_custom_test')
        unique_custom_x_test = recursive_dataset_generator(unique_custom_test_scaled_data)
        lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model
        unique_custom_test_scaler = scaler_opener('unique_custom_test')

        unique_prediction = []

        next_prediction = lstm_model.predict(np.array([unique_custom_x_test]).reshape((1,look_back,1))).flatten()
        unique_prediction.append(next_prediction)
        unique_prediction = unique_custom_test_scaler.inverse_transform(unique_prediction)
        if np.any(prediction_custom_test) == True:        # checking if not empty
            prediction_custom_test = np.vstack([prediction_custom_test,(unique_prediction.reshape(1,1))])
        elif np.any(prediction_custom_test) == False:          # checking if empty
            prediction_custom_test = unique_prediction.reshape(1,1)
    #-------------------------------------------------------------------------------------------
    #--------------------------------------Finished Preparing prediction_custom_test-----------------------------------------------------

    #------------------------------------------Preparing Recursive Prediction-------------------------------------------------
    today = dt.datetime.now().date()
    delta = abs((dt2-dt4).days) + 1
    dt3 = dt4 - timedelta(days=look_back)
    
    recursive_custom_test_data = yf.download(f'{crypto_coin}-{currency}', start=dt3, end=dt4, progress=False,)
    recursive_custom_test_data.to_csv(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_cryptoData_recursive_custom_test_dates.csv')

    recursive_custom_test_scaled_data = custom_scaler(recursive_custom_test_data['Close'],'recursive_custom_test')
    recursive_custom_x_test = recursive_dataset_generator(recursive_custom_test_scaled_data)
    lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model
    recursive_custom_test_scaler = scaler_opener('recursive_custom_test')

    # recursive prediction code
    recursive_predictions = []
    lstm_model = load_model(f'./data_modelCreation/{crypto_coin}/{crypto_coin}_lstm_model.keras')                             # to reuse the lstm_model

    last_window = deepcopy(recursive_custom_x_test[-1])
    for i in range(delta):
        temp1 = np.array([])
        temp = deepcopy(last_window)
        next_prediction = lstm_model.predict(np.array([temp]).reshape((1,look_back,1))).flatten()
        recursive_predictions.append(next_prediction)
        temp1 = np.vstack([temp[1:], next_prediction.reshape((1,1))])
        last_window = deepcopy(temp1)
    recursive_custom_test_scaler = scaler_opener('recursive_custom_test') 
    recursive_predictions = recursive_custom_test_scaler.inverse_transform(recursive_predictions)
    
    #------------------------------------------Finished Preparing Recursive Prediction-------------------------------------------------
    
    data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)          # whole data from yf in data df to plot

    #------------------------------------------ Preparing to Plot All results-------------------------------------------------
    
    # Plotting Recursive_predictions, Regular/non recursive Predictions, Actual Prices
    #print('Recursive_predictions, Regular/non recursive Predictions, Actual Prices for ' + crypto_coin + ' from ' + str(dt4) + ' to ' + str(dt2) + ' as following:')
    

    overlap = ((today - dt4).days + 1)
    periods = len(data) + delta - overlap
    if (len(data) + delta - overlap) < len(data): periods = len(data)

    custom_test_dates = pd.date_range(start=start, periods= periods, freq='D')
    
    if (dt2<=today and dt4<= today_plus_one) or (dt2>today and dt4<=today_plus_one):
        pred_start_index = len(data) - overlap
        if dt4==today_plus_one:
            pred_start_index = len(data) - overlap
        pred_end_index = pred_start_index + len(prediction_custom_test)
    if (dt2>today and dt4<=today_plus_one) or (dt2<=today and dt4<= today_plus_one):
        rec_start_index = len(data) - overlap
        rec_end_index = rec_start_index + len(recursive_predictions)

    plt.figure(figsize=(16,8))
    plt.plot(custom_test_dates[:len(data)], np.array(data.loc[:,'Close']), marker = '.', color='orange',label='Actual Prices')
    plt.plot(custom_test_dates[rec_start_index:rec_end_index], recursive_predictions[:,], marker = '.', color='blue',label='Recursive Predictions Prices')
    plt.plot(custom_test_dates[pred_start_index:pred_end_index], prediction_custom_test[:,], marker = '.', color='green',label='Predictions Prices')
    plt.title(f'{crypto_coin} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'Price in {currency}')
    plt.legend(loc='upper left')
    plt.show()
    
    return True



# recursive_predict2 : predicts recursive predictions AND regular/non recursive predictions in daterange

today = dt.datetime.now()
default_date_before_3_month = (today - dt.timedelta(days=90)).strftime("%Y-%m-%d")              # default dates mainly for streamlit date selector
default_date_after_2_month = (today + dt.timedelta(days=60)).strftime("%Y-%m-%d") 
start_date = default_date_before_3_month
end_date = default_date_after_2_month
# start_date = '2024-06-05'
# end_date = '2024-07-06'
recursive_predict2(start_date,end_date)




# predict : predicts regular/non recursive predictions in daterange

today = dt.datetime.now()
default_date_before_4_month = (today - dt.timedelta(days=120)).strftime("%Y-%m-%d")             # default dates mainly for streamlit date selector
default_date_before_2_month = (today - dt.timedelta(days=60)).strftime("%Y-%m-%d") 
pred_start_date = default_date_before_4_month
pred_end_date = default_date_before_2_month
# pred_start_date = '2024-03-04'                                       
# pred_end_date = '2024-05-04'
predict(pred_start_date, pred_end_date)



# recursive_predict1 : predicts recursive predictions in daterange

today = dt.datetime.now()
default_date_after_4_month = (today + dt.timedelta(days=120)).strftime("%Y-%m-%d")             # default dates mainly for streamlit date selector
recursive_date_start = today_plus_one.strftime('%Y-%m-%d')
recursive_date_end = default_date_after_4_month
# recursive_date_start = '2024-07-05'
# recursive_date_end = '2024-11-02'
recursive_predict1(recursive_date_start,recursive_date_end)
