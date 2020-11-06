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

dataset_train = pdr.get_data_yahoo("Vale", start="2010-01-01", end="2019-09-01")
training_set = dataset_train.iloc[:,3:4].values



# Parameters
days_ahead =  5
timesteps =  60

# Feature Scalling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
#testing_set_scaled = sc.fit_transform(testing_set)


#creating a train data structure with defined timesteps and 5 output 
x_train = []
y_train = []
for i in range(timesteps, len(dataset_train)-days_ahead):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    
    y_train.append(training_set_scaled[i:i+days_ahead,0])
    


y_train = np.array(y_train)
x_train = np.array(x_train) 

#Reshaping to Keras
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))






##### Valiation data #######

dataset_validate = pdr.get_data_yahoo("Vale", start="2019-09-01", end="2020-11-01")
validate_set = dataset_validate.iloc[:,3:4].values
validate_set_scaled = sc.fit_transform(validate_set)

x_validate =  validate_set_scaled[:timesteps,0].reshape(-1,1)
y_validate = validate_set_scaled[timesteps:timesteps+days_ahead,0].reshape(-1,1)


x_validate = np.reshape(x_validate, (x_validate.shape[1], x_validate.shape[0], 1))
y_validate = np.reshape(y_validate, (y_validate.shape[1], y_validate.shape[0], 1))


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
regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 128))
regressor.add(Dropout(0.2))

# Adding  the output layer
regressor.add(Dense(units = days_ahead))


# Compilling the RNN to the Training set
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 50, batch_size = 128, shuffle = True, validation_data=(x_validate, y_validate))



#############################################################
## Part 3 - Making the prediction and visualising the results
#############################################################

# Download the test set
dataset_test = pdr.get_data_yahoo("Vale", start="2019-09-01", end="2020-11-01")
testing_set = dataset_test.iloc[:,3:4].values

# Creating a test data structure with defined timesteps
total_set = np.concatenate((training_set, testing_set), axis=0)
testing_set_with_timesteps = total_set[len(total_set) - len(testing_set) - timesteps:].reshape(-1,1)

# Scalling
testing_set_with_timesteps_scaled = sc.fit_transform(testing_set_with_timesteps)

# start to prediction
X_test = []
y_test = []
prediction_seq = []

# point in the y_test the prediction will starts
pointer = timesteps   



for i in range(pointer, len(testing_set_with_timesteps_scaled), days_ahead):
    print(i)
        
    X_test = testing_set_with_timesteps_scaled[i-timesteps:i]
    X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
    
    prediction = regressor.predict(X_test)
    print(prediction)
    
    prediction_seq.append(prediction[0])
    
    # Transform y_test to print
    prediction_transformed = sc.inverse_transform(np.array(prediction).reshape(-1,1))
    x_index_seq = np.array(list(range(i, i+days_ahead))).reshape(-1,1)
    plt.plot(x_index_seq, prediction_transformed, color = 'blue')
    


#Y_test = testing_set_with_timesteps_scaled[testing_set_with_timesteps_scaled.shape[0] - days_ahead :]
Y_test = testing_set_with_timesteps_scaled
Y_test_transformed = sc.inverse_transform(np.array(Y_test).reshape(-1,1))


#prediction_transformed = sc.inverse_transform(np.array(prediction).reshape(-1,1))


# Visualising
plt.plot(Y_test_transformed, color = 'red', label = 'Real Petrobras Stock Price')
#plt.plot(prediction_transformed, color = 'blue', label = 'Predicted Petrobras Stock Price')
plt.title('Petrobras Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Petrobras Stock Price')
#plt.xlim(0, pointer)
plt.legend()
plt.show()




