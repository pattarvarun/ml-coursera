#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:19:41 2019

@author: varun
"""
##################
#
#DATA PREPROCESSING

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import training dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps
X_train = []
y_train = []

for i in range(60,len(training_set)-1):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)


#Reshaping

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))



#BUILDING RNN

#import keras libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialse RNN
regressor = Sequential()

#adding first lstm layer and some dropout regularisation

regressor.add(LSTM(units=50,return_sequences=True,input_shape(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#second lstm layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#3rd lstm layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#4th lstm layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units=1))

#compile the network
regressor.compile(optimizer='adam',loss = 'mean_squared_error')


#Fit the network to the training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

#MAKING PREDICTIONS AND VISUALISING THE RESULTS

#Actual stock price(real price)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#Getting predictions
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_prices = regressor.predict(X_test)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices)


#visualising the results.

plt.plot(real_stock_price,color = 'red',label = "Real Google Stock Price")
plt.plot(predicted_stock_prices,color = 'blue',label = 'Predicted stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()


