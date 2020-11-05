# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 00:09:03 2020

@author: danie
"""



#############################
##Part 1 - Data Preprocessing
#############################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


# Downloading the train set
yf.pdr_override()

dataset_train = pdr.get_data_yahoo("PETR4.SA", start="2015-01-01", end="2020-08-01")
training_set = dataset_train.iloc[:,3:4].values

# Parameters
days_ahead =  5
timesteps =  60

# Feature Scalling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
#testing_set_scaled = sc.fit_transform(testing_set)


#creating a train data structure with defined timesteps and 1 output 
x_train = []
y_train = []
for i in range(timesteps, len(dataset_train)):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping to Keras
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))




############################
## Part 2 - Building the RNN
############################

# Importing the libraries
import tensorflow as tf
import keras as kr
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding  the output layer
regressor.add(Dense(units = 5))


# Compilling the RNN to the Training set
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 30, batch_size = 32)



#############################################################
## Part 3 - Making the prediction and visualising the results
#############################################################

# Download the test set
dataset_test = pdr.get_data_yahoo("PETR4.SA", start="2020-08-02", end="2020-10-28")
testing_set = dataset_test.iloc[:,3:4].values

# Creating a test data structure with defined timesteps
total_set = np.concatenate((training_set, testing_set), axis=0)
testing_set_with_timesteps = total_set[len(total_set) - len(testing_set) - timesteps:].reshape(-1,1)

# Scalling
testing_set_with_timesteps_scaled = sc.fit_transform(testing_set_with_timesteps)

# start to prediction
X_test = []
y_test = []

# point in the y_test the prediction will starts
pointer = timesteps   

# Predict while to exist values to predict into y_test
while pointer + days_ahead <= len(testing_set_with_timesteps_scaled):
    
    # x_test used to predict the next days_ahead
    X_test = testing_set_with_timesteps_scaled[:pointer]

    # Predict prices for the next days_ahead
    for i in range(pointer, pointer+days_ahead):
        print(i)
        
        # X-test window specificly to this iteration
        X_test_temp = X_test[i-pointer:i, 0]
        X_test_temp = np.array(X_test_temp)
        
        # Reshaping to Keras
        X_test_temp = np.reshape(X_test_temp, (1, X_test_temp.shape[0], 1))
        
        # Predict price
        prediction = regressor.predict(X_test_temp)
        print(prediction)
        
        # Update y_test and also x_test for the next prediction
        y_test.extend(prediction[0])
        X_test = np.append(X_test, prediction, axis=0)


    # Transform y_test to print
    y_test_transformed = sc.inverse_transform(np.array(y_test).reshape(-1,1))
    x_index_seq = list(range(pointer, pointer+days_ahead))
    plt.plot(x_index_seq, y_test_transformed[len(y_test_transformed)-days_ahead:], color = 'blue' )
    
    # Update the pointer for the next sequence of predictions
    pointer += days_ahead

y_test = sc.inverse_transform(np.array(y_test).reshape(-1,1))
inputsUpdatedTransformed = sc.inverse_transform(np.array(X_test).reshape(-1,1))


dataset_total_print = np.array(total_set[len(total_set) - (pointer):])
#predicted_print = np.append(dataset_total_print[:len(dataset_total_print) - len(dataset_test)], y_test)


# Visualising
plt.plot(dataset_total_print, color = 'red', label = 'Real Petrobras Stock Price')
#plt.plot(predicted_print, color = 'blue', label = 'Predicted Petrobras Stock Price')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Petrobras Stock Price')
#plt.xlim(0, pointer)
plt.legend()
plt.show()
