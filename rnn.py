#Recurrent



# Part 1 Data Pre Processing

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the training set (only numpy arrays can be the input)

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values #gives numpy array of one columns rather than a simple vector



# Feature Scaling (choose normalization over standardisation because there is a sigmoid function as activation function)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0 , 1))
training_set_scaled = sc.fit_transform(training_set) #fit gets the min and max and applies the normalization formula for each stock price


# Create a data structure with 60 timesteps and 1 output

X_train = []   # input , for each observation (financial day) contains the 60 previous stock prices before that financial day
y_train	= []   # stock price for the next financial day

for i in range(60, len(training_set_scaled)):
	X_train.append(training_set_scaled[i-60:i, 0])
	y_train.append(training_set_scaled[i , 0])

X_train , y_train  = np.array(X_train), np.array(y_train) #1198 x 60 for X_train, first line is time = 60, each column are previous 59 values leading up to that day

# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #anytime you want to add a new dimension you need to use the reshape function
									#input shape keras expects is 3D: 
										#batch size (total number of stock prices in training data)
										#timesteps (60)
										#input_dim (optional third dimension)







# Part 2 Build RNN (layers, neurons, LSTM, output, dropout regularization to avoid overfitting)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Part 3 - Predict and Visualize