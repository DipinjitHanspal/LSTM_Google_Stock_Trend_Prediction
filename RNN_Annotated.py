# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:35:13 2020

@author: Dipinjit Hanspal
"""

# Recurrent Neural Network
"""
Creating a stacked LSTM using a RNN
"""


## -- Part 1 - Data Preprocessing -- ##

# Import libraries

# Import training set

from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset_train = pd.read_csv('./dataset/Google_Stock_Price_Train.csv')
# Convert to np array for LSTM. To prevent the conversion to a vector,
# we give a range for columns (1:2).
training_set = dataset_train.iloc[:, 1:2].values

## -- Feature Scale -- ##
"""
Normalize values instead of standardizing. Whenever a sigmoid is used as 
the activation function, normalization is recommended over standardization 
"""
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

## -- Timesteps -- ##
# Create datastructure with 60 time steps and 1 output for the RNN
# 60 timesteps = 20 business days/month * 3 months of observed data for output
"""
    The number of timesteps is the number of previous outputs the RNN will 
    consider before predicting the output h at time t + 1. Important to optimize 
    the time steps to prevent overfitting. If you use too few time steps the
    trend captured may not be generalized enough for accurate predictions since
    the examined timespan is too short. 
"""
X_train, y_train = [], []

for i in range(60, 1258):
    # Previous 60 days to train
    X_train.append(training_set_scaled[i-60:i, 0])
    # Stock price at time t + 1
    y_train.append(training_set_scaled[i, 0])
# Convert back to np array from DataFrame
X_train, y_train = np.array(X_train), np.array(y_train)

## -- Reshaping -- ##
# Add more indicators in more dimensions that you can use to improve prediction
# reshape requires batch_size/# of observations, # timesteps, # dimensions (if you're using more indicators you add them here)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN
# Import Keras libraries

## -- Input -- ##
regressor = Sequential()

regressor.add(LSTM(50, return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
"""
    LSTM requires 3 arguments : 
        units : # of LSTM cells in the layer
        return_sequences : true for a stacked LSTM
        input_shape : Shape of input (timesteps, indicators). 
                        # of observations is accounted for automatically 
"""
regressor.add(Dropout(0.20))

## -- Second LSTM layer with Dropout reglarization -- ##
regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.20))

## -- Third LSTM layer with Dropout reglarization -- ##
regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.20))

## -- Fourth LSTM layer with Dropout reglarization -- ##
# No return sequences required (false by default)
regressor.add(LSTM(50))
regressor.add(Dropout(0.20))

## -- Output Layer -- ##
regressor.add(Dense(units=1))

## -- Compile RNN -- ##
regressor.compile(optimizer='adam', loss='mse')
"""
    Keras documentation recommends RMSprop but adam works better in our case.
    When choosing an optimizer for a Neural Network, adam is always a safe bet
    because it provides some relevant updates to the weights of the RNN. 
"""

## -- Fit to training set -- ##

regressor.fit(X_train, y_train, batch_size=32, epochs=100)

# Part 3 - Make predictions and visualize results

# Get real stock price
dataset_test = pd.read_csv('./dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Get predicted stock price
"""
    Problem : You need both the train and test set to accurately predict stocks 
        (since you need) 60 previous days, some of which will be in train. However,
        you can't just concatenate them because they are scaled independently ( and only
        relative to other values within the respective sets because they were normalized), 
        and scaling them together will change the values
    Solution : Only scale the 60 inputs that you are using instead of the whole concatenated set
"""

# Get opening values of the google stock
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# get previous 60 days of values (from before the first day in the test set)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Fit to single column
inputs = inputs.reshape(-1,1)
# Transform only because the sc object was already fit to the input shape
inputs = sc.transform(inputs)
## -- Transform to into 3D datastructure to feed into RNN -- ##
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
# Convert back to np array from DataFrame
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
## -- Inverse the scaling to get real values -- ##
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google stock Price Prediction')
plt.xlabel('time')
plt.ylabel('price ($USD)')
plt.legend()
plt.show()