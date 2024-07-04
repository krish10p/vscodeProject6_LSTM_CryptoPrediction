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
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from dateutil import parser
from pickle import dump
from pickle import load
from keras.models import load_model
import yfinance as yf
# # importing libraries
# import streamlit as st
# import datareader as datareader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas_datareader as web
# from datetime import datetime, timedelta
# import datetime as dt
# from sklearn.preprocessing import MinMaxScaler
# from keras.layers import Dense, Dropout, LSTM
# from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from dateutil import parser
# from pickle import dump
# from pickle import load
# from keras.models import load_model





#st.title('Crypto Currency Prediction')

#Loading the data
global coin_name
global start_date
global end_date

coin_list = ['BTC','ETH','HOT1','RSR','NULS','NIM','AION','QASH','VITE','APL','QRL','BCN','GBYTE','LBC','POA','PAC','ILC','BEPRO','GO','XMC']
#coin_name = st.sidebar.selectbox('Select Crypto Currency',coin_list)
#start_date = st.sidebar.date_input('Provide a starting date')
#end_date = st.sidebar.date_input('Provide an ending date')
currency_list = ['USD','GBP','EUR','INR', 'CAD', 'AUD','JPY','CNY']
#currency = st.sidebar.selectbox('Select a Currency',currency_list)
#start_date = start_date.strftime("%Y-%m-%d")
#end_date = end_date.strftime("%Y-%m-%d")
#start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()                #converting date to date type
#end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()                    #converting date to date type

coin_name = 'BTC'
start_date = '2024-05-01'
end_date = '2024-05-27'
currency = 'USD'

# print('Select any of the coins :')
# print(coin_list)
# print('Enter the coin name :')
# #coin_name = input()
# print('Entered coin is : ', coin_name)
# print('Enter the date format in following way :')
# print('Enter the date range for ', coin_name +' price prediction in yyyy-mm-dd Format :')
# print('Enter the start date :')
# #start_date = input()
# print('Enter the end name :')
# #end_date = input()
# print('Entered date range is :', start_date,'to', end_date)
# #currency = input()
# print('Entered currency is : ', currency)


crypto_currency = coin_name
against_currency = currency

start = dt.datetime(2022,1,1)
end = dt.datetime.now()
end = end + dt.timedelta(days=1)                        # cause yf download exclude end date in downloading

#data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo',start, end)
data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end, progress=False,)



data.to_csv('cryptoCoinYahoo.csv')
#prepare data
print(data.shape)
print(data.head())
print(data.info())
#data                                                                            #df

#checking if there is any null values in data frame columns as part of data-preprocessing
data.isnull()
data.isnull().sum()


#creating the scaler for transforming the data to normalized values as part of data-preprocessing
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
#save the scaler
dump(scaler, open('scaler_model1.pkl','wb'))

#global split_percent
split_percent = 0.8
split = int(split_percent*len(scaled_data))
train_scaled = scaled_data[:split]
test_scaled = scaled_data[split:]

global look_back
global future_day
look_back = 60
future_day = 1                                            #   future day prediction = 1st day or 61st prediction from learning 60 days

#creating x_train and y_train from train_scaled

x_train, y_train = [], []

train_scaled1 = scaled_data[:split+future_day]

for x in range(look_back,(len(train_scaled1)-future_day)):                                        # for loop in range (60, 1871)
    x_train.append(train_scaled1[x-look_back:x,0])
    y_train.append(train_scaled1[x:x+future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)                             #   x_train : (1871, 60), y_train :   (1871, 1)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))              #   x_train after reshaping : (1871, 60, 1) and converting into 3d array because LSTM receives input in 3D or 3 dimensional array

#creating x_test and y_test from test_scaled
test_scaled1 = scaled_data[split-look_back:,:]

x_test = []
y_test = []

for x in range(look_back,len(test_scaled1)-future_day):                         #   (60, 483-6))
    x_test.append(test_scaled1[x-look_back:x,0])
    y_test.append(test_scaled1[x:x+future_day, 0])

x_test, y_test = np.array(x_test), np.array(y_test)                             #   x_train : (483, 60), y_train :   (483,1)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))              #   x_test after reshaping : (483, 60, 1)

#Checking shapes
print(type(x_train), type(y_train))                               #   numpy array
print(x_train.shape)
print(y_train.shape)
print(type(x_test), type(y_test))                                 #   numpy array
print(x_test.shape)                                               #   (483, 60, 1)
print(y_test.shape)                                               #   (483,)


# Create the model :
# Create Neural Network
model1 = Sequential()
model1.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model1.add(Dropout(0.2))                                            #   We add Dropout layer to avoid overfitting of the model
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50))
model1.add(Dropout(0.2))
model1.add(Dense(units=future_day))
model1.summary()

model1.compile(optimizer='adam', loss='mean_squared_error')
model1.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)                 #keep epochs at least 10
#save the model1                                        # too big model for pickle to save, so use keras.models
#dump(model1, open('model1.pkl','wb'))
model1.save('model1.h5')                                # to save keras model and load the model

# Evaluate the model on test data
model1.evaluate(x_test, y_test)

# Predict the model on test data and plot the output graph
prediction_prices = model1.predict(x_test, batch_size=1)
prediction_prices = scaler.inverse_transform(prediction_prices)
actual_prices = scaler.inverse_transform(test_scaled)

plt.figure(figsize=(16,8))
plt.plot(actual_prices, color='orange', label='Actual Prices')
plt.plot(prediction_prices, color='green',label='Prediction Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
#plt.show()


# input_set method to receive the date range and returns X_test,Y_test data set for that specific range
def input_set(dt0,dt1):

    dt0 = dt.datetime.strptime(dt0, '%Y-%m-%d').date()
    dt1 = dt.datetime.strptime(dt1, '%Y-%m-%d').date()
    dt1 = dt1 + dt.timedelta(days=1)                            # cause yf download exclude end date in downloading
    delta = abs((dt0 - dt1).days)

    dt00 = dt0 - timedelta(days=look_back)
    #data11 = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo',dt00, dt1)
    data11 = yf.download(f'{crypto_currency}-{against_currency}', start=dt00, end=dt1, progress=False,)
    data11.to_csv("coin_SelectedDate.csv")
    scaler11 = MinMaxScaler(feature_range=(0, 1))
    scaled_data11 = scaler11.fit_transform(data11['Close'].values.reshape(-1, 1))

    #test_scaled11 = scaled_data[-delta:]
    dt0,dt1,delta

    test_scaled111 = scaled_data11[:,:]

    x_test11 = []
    y_test11 = []
    c = look_back
    for x11 in range(look_back,len(test_scaled111)-future_day+1):                             #   (60, 397))      (60,27+60-1+1)=(60,87)
        x_test11.append(test_scaled111[x11-look_back:x11, 0])
        y_test11.append(test_scaled111[x11:x11+future_day, 0])      

    x_test11, y_test11 = np.array(x_test11), np.array(y_test11)                             #   x_train : (483, 60), y_train :   (483,1)
    x_test11 = np.reshape(x_test11, (x_test11.shape[0], x_test11.shape[1], 1))              #   x_test after reshaping : (483, 60, 1)

    print(x_test11.shape,y_test11.shape,type(x_test11),type(y_test11))                      #   nd-array

    return x_test11,y_test11

# predict method to predict on specific test_set  and plot the chart for actual vs predicted prices + print actual and predicted prices
def predict(x_test1111,y_test1111):

    print('Prediction for ' + coin_name + ', for selected date range from ' + start_date + ' to ' + end_date + ' is as following :')
    #st.header('Prediction for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date + ' is as following:')

    model = load_model('model1.h5')                             # to reuse the model
    scaler_model1 = load(open('scaler_model1.pkl','rb'))                                                 # to reuse the scalar

    model.evaluate(x_test1111, y_test1111)
    prediction_test1111 = model.predict(x_test1111,batch_size=1)
    prediction_test1111 = scaler_model1.inverse_transform(prediction_test1111)
    actual_prices1111 = scaler_model1.inverse_transform(y_test1111)

    dates = pd.date_range(start_date, end_date, freq='D')

    df1111 = pd.DataFrame({'dates':dates,'Prediction':prediction_test1111.reshape(-1,),'Actual':actual_prices1111.reshape(-1,)})
    df1111.set_index('dates',inplace=True)
    
    plt.figure(figsize=(16,8))
    plt.plot(df1111.iloc[:,0], color='orange', label='Prediction Prices')
    plt.plot(df1111.iloc[:,1], color='green',label='Ground Truth Prices')
    plt.title(f'{crypto_currency} price prediction', fontsize=24)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel(f'Price in {against_currency}', fontsize=24)
    plt.legend(loc='upper right')
    plt.show()
    #st.pyplot(plt)

    #st.write('Prediction and Actual Prices for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date, df1111)
    print('Prediction and Actual Prices for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date)
    print(df1111)

    return True

## For Final output
# User provide input date range goes here in below method and predict output and plot the graph adn print values too
a = input_set(start_date,end_date)
predict(a[0],a[1])

