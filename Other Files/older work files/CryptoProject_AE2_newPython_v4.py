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




#Loading the data
global start                                            # start, end for whole dataframe for training/testing whole
global end
global crypto_coin
global currency

crypto_coin = 'BTC'
currency = 'CAD'

start = dt.datetime(2021,1,1)                           # start, end for whole dataframe for training/testing whole
today = dt.datetime.now()
today_plus_one = today + dt.timedelta(days=1)                        # Because, yf download excludes end date while downloading data
end = today_plus_one

data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)
data.to_csv('cryptoData_whole.csv')

#prepare data
print(data.shape)
print(data.head())

#global split_percent
split_percent = 0.8
split = int(split_percent*len(data))
train_data = data[:split]
test_data = data[split:]

# creates scalaed data and dumps scalar with scalar_name_model.pkl using MinMaxScalar
def custom_scaler(dataset, scaler_name):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(dataset.values.reshape(-1,1))
    #save the scaler
    dump(scaler, open(f'{scaler_name}_scaler.pkl','wb'))
    return data_scaled

train_scaled = custom_scaler(train_data['Close'],'train')
test_scaled = custom_scaler(test_data['Close'],'test')

# opens/loads saved scaler model in .pkl format
def scaler_opener(scaler_name):
    return load(open(f'{scaler_name}_scaler.pkl','rb'))
#train_scaler = scaler_opener('train')                                  # use function/open scaler when you need them
#test_scaler = scaler_opener('test')                                    

global look_back
#global future_day                                                      # global one not useful in the current context of code
look_back = 60
future_day = 1                                                          # future day prediction =61st day prediction from learning past 60 days/look_back days:60/time_stamps_of_lstm:60 

# #creating x_train and y_train from train_scaled using dataset_generator_lstm function

def dataset_generator_lstm(dataset, look_back=60):
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    dataX, dataY = np.array(dataX), np.array(dataY)
    return np.reshape(dataX,(dataX.shape[0],dataX.shape[1], 1)), dataY

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
lstm_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)                 #keep epochs at least 10
#save the lstm_model                                        # too big model for pickle to save, so use keras.models
#dump(lstm_model, open('lstm_model.pkl','wb'))
lstm_model.save('lstm_model.keras')                                # to save keras model and load the model

# Evaluate the model on test data
lstm_model.evaluate(x_test, y_test)

# Predict the model on test data and plot the output graph
prediction_test = lstm_model.predict(x_test, batch_size=1)

test_scaler = scaler_opener('test')
prediction_test = test_scaler.inverse_transform(prediction_test)
actual_test = test_scaler.inverse_transform(test_scaled)
test_t = np.arange(len(test_data))

# recursive prediction code (on test_data)
dates_data = pd.date_range(start=start, periods=len(data), freq='D')
dates_train = dates_data[:split,]
dates_test = dates_data[split:,]

recursive_predictions = []

last_window = deepcopy(x_train[-1])
for i in dates_test:
    temp1 = np.array([])
    temp = deepcopy(last_window)
    next_prediction = lstm_model.predict(np.array([temp]).reshape((1,look_back,1))).flatten()
    recursive_predictions.append(next_prediction)
    temp1 = np.vstack([temp[1:], next_prediction.reshape((1,1))])
    last_window = deepcopy(temp1)
train_scaler = scaler_opener('train') 
recursive_predictions = train_scaler.inverse_transform(recursive_predictions)


plt.figure(figsize=(16,8))
plt.plot(dates_test[:], actual_test, marker = '.', color='orange', label='actual_test Prices')
plt.plot(dates_test[look_back:], prediction_test, marker = '.', color='green',label='prediction_test Prices')
plt.plot(dates_test[:], recursive_predictions, marker = '.', color='blue',label='recursive_predictions Prices')               # keep uncommented only if recursive_predict above code uncommented on test_set
plt.title(f'{crypto_coin} Price Prediction')
plt.xlabel('Time (in days)')
plt.ylabel(f'Price in {currency}')
plt.legend(loc='upper left')
plt.show()

# predict function to predict crypto prices between selected dates and plot the graph for actual vs predicted prices, & print actual and predicted prices
def predict(dt0, dt1):

    dt0 = dt.datetime.strptime(dt0, '%Y-%m-%d').date()
    dt1 = dt.datetime.strptime(dt1, '%Y-%m-%d').date()
    dt1_plus_one = dt1 + dt.timedelta(days=1)                            # cause yf download exclude end date in downloading
    dt3 = dt0 - timedelta(days=look_back)
    
    custom_test_data = yf.download(f'{crypto_coin}-{currency}', start=dt3, end=dt1_plus_one, progress=False,)
    custom_test_data.to_csv("cryptoData_custom_test_dates.csv")

    custom_test_scaled_data = custom_scaler(custom_test_data['Close'],'custom_test')
    custom_x_test, custom_y_test = dataset_generator_lstm(custom_test_scaled_data)
    lstm_model = load_model('lstm_model.keras')                             # to reuse the lstm_model
    custom_test_scaler = scaler_opener('custom_test')
    lstm_model.evaluate(custom_x_test, custom_y_test)                   # use results of evaluation if needed on custom test set
 
    prediction_custom_test = lstm_model.predict(custom_x_test,batch_size=1)
    prediction_custom_test = custom_test_scaler.inverse_transform(prediction_custom_test)
    actual_custom_test = custom_test_scaler.inverse_transform(custom_test_scaled_data)

    print('Prediction and Actual Prices for ' + crypto_coin + ' with selected date from ' + str(dt0) + ' to ' + str(dt1) + ' as following:')
   
    # Ploting with dates: Predict the prices using model on custom test data and plot the actual vs predicted prices graph
    dates = pd.date_range(end=dt1, periods=len(custom_test_data)-look_back, freq='D')

    plt.figure(figsize=(16,8))
    plt.plot(dates, prediction_custom_test.reshape(-1,), marker = '.', color='orange', label='Prediction Prices')
    plt.plot(dates, actual_custom_test[look_back:,].reshape(-1,), marker = '.', color='green',label='Ground Truth Prices')
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


def recursive_predict(dt2):
    
    dt2 = dt.datetime.strptime(dt2, '%Y-%m-%d').date()
    today = dt.datetime.now().date()
    delta = abs((dt2 - today).days)  

    if dt2 <= today :
        print('The chosen recursive date is in past, Select any future date for recursive_prediction')
        return
    
    # recursive prediction code
    recursive_predictions = []
    lstm_model = load_model('lstm_model.keras')                             # to reuse the lstm_model

    last_window = deepcopy(x_test[-1])
    for i in range(delta):
        temp1 = np.array([])
        temp = deepcopy(last_window)
        next_prediction = lstm_model.predict(np.array([temp]).reshape((1,look_back,1))).flatten()
        recursive_predictions.append(next_prediction)
        temp1 = np.vstack([temp[1:], next_prediction.reshape((1,1))])
        last_window = deepcopy(temp1)
    test_scaler = scaler_opener('test') 
    recursive_predictions = test_scaler.inverse_transform(recursive_predictions)
    data = yf.download(f'{crypto_coin}-{currency}', start=start, end=end, progress=False,)          # whole data from yf in data df to plot

    # # # Ploting without dates: Predict the model on custom test data and plot the actual vs predicted prices graph
    # custom_test_t = np.arange(len(data)+delta)
    # plt.figure(figsize=(16,8))
    # plt.plot(custom_test_t[:len(data)], np.array(data.loc[:,'Close']), marker = '.', color='pink',label='Actual Prices')
    # plt.plot(custom_test_t[len(data):], recursive_predictions[:,], marker = '.', color='blue',label='recursive_predictions Prices')
    # plt.title(f'{crypto_coin} Price Prediction')
    # plt.xlabel('Time (in days)')
    # plt.ylabel(f'Price in {currency}')
    # plt.legend(loc='upper left')
    # plt.show()

    # # Ploting with dates: Predict the model on custom test data and plot the actual vs predicted prices graph
    custom_test_dates = pd.date_range(end=dt2, periods= len(data)+delta, freq='D')
    plt.figure(figsize=(16,8))
    plt.plot(custom_test_dates[:len(data)], np.array(data.loc[:,'Close']), marker = '.', color='pink',label='Actual Prices')
    plt.plot(custom_test_dates[len(data):], recursive_predictions[:,], marker = '.', color='blue',label='recursive_predictions Prices')
    plt.title(f'{crypto_coin} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'Price in {currency}')
    plt.legend(loc='upper left')
    plt.show()
    
    return True



# to recursive predict crypto value, from today till recursive_date which will be end_date
default_date_after_2_month = (today + dt.timedelta(days=60)).strftime("%Y-%m-%d")             # default dates mainly for streamlit date selector
recursive_date = default_date_after_2_month
#recursive_date = '2024-08-26'
recursive_predict(recursive_date)


# to predict crypto value betbeen custom_test daterange
start_date = '2024-05-01'                                       # start_date, end_date for selected dates for prediction
end_date = '2024-05-27'
predict(start_date, end_date)