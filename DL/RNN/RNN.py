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

