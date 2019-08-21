# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Recurrent



# -----------------------     Part 1 DATA PRE-PROCESSING     -----------------

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







# ---------------------------------    PART 2 BUILD RNN    -------------------
#(layers, neurons, LSTM, output, dropout regularization to avoid overfitting)
                                        
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



regressor = Sequential()


# Adding the first LSTM layer and some dropout regularization

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = .20))

# Adding the second LSTM layer and some dropout regularization

regressor.add(LSTM( units = 50, return_sequences = True ))
regressor.add(Dropout( rate = .20 ))

# Adding the third LSTM layer and some dropout regularization

regressor.add(LSTM( units = 50, return_sequences = True ))
regressor.add(Dropout( rate = .20 ))

# Adding the fourth LSTM layer and some dropout regularization

regressor.add(LSTM( units = 50 ))
regressor.add(Dropout( rate = .20 ))

# Adding the output layer

regressor.add(Dense( units = 1 ))

# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  #Adam optimizer vs RMSprop optimizer (see keras.io/optimizers), binary cross entropy for classification

# Fit the network to training data

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) #choose epochs value where you observe convergence of the loss


# -------------------   Part 3 - Predict and Visualize   ---------------------

# Getting the real stock prices

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock prices

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #axis = 1 is horizontal (cbind, 0 = rbind)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []

for i in range(60,80): #only 20 days of test data, and we get 60 days leading up to each of those 20 financial days
    x_test.append(inputs[i-60:i,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualizing the final result (actual vs predicted)


plt.plot(predicted_stock_price, color='blue', label='Predicted Price')
plt.plot(real_stock_price, color = 'red', label = 'Real Price')
plt.title('Real vs Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.ylim(700,900)
plt.show()




