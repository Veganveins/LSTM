# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:00:54 2019

@author: JJ5JXT
"""
# -----------------------     Part 1 DATA PRE-PROCESSING     -----------------

import pandas as pd
from keras.preprocessing import sequence
from itertools import chain
#from imdb_example.py import encode_baskets, encode_xtrain

basket_training_simple = pd.read_csv("C:\\Users\\jj5jxt\\Desktop\\june\\x_train_simple.csv")
basket_training_simple = pd.read_csv("C:\\Users\\jj5jxt\\Desktop\\june\\x_train_proposed.csv")
basket_training_simple = pd.read_csv("C:\\Users\\jj5jxt\\Desktop\\june\\x_train_orig.csv")

ilinks_simp = basket_training_simple.ILINK.unique() # get unique list of ILINKs

x_train_s = []   # some sequence of a customers baskets
y_train_s	= []   # the next basket given a sequence of a customer's baskets


for ilink in ilinks_simp:
    
    append_field = basket_training_simple.loc[basket_training_simple['ILINK'] == ilink].iloc[:,2:3].values  # store a list of baskets for each ilink
    samples = len(basket_training_simple.loc[basket_training_simple['ILINK'] == ilink]['Concat_CLASS_NAME']) # count how many total baskets exist in each ilink's unique sequence
    
    for i in range(1,samples):
    
        x_train_s.append(append_field[0:i])  # for each basket sequence of length n, save each sequence (1 -> 2, 1:2 -> 3, 1:3 -> 4 ... 1:n-1 -> n)
        y_train_s.append(append_field[i])    # store the appropriate y for each sub sequence


# ---------------- HANDLE y_train --------------------------------------------

    
# create a method to convert y_train elements to arrays rather than strings with np.char.split

y_train2_s = y_train_s.copy()
for i in range(0, len(y_train_s)):   
     y_train2_s[i] = np.char.split(y_train_s[i][0], sep = ',')   # would adding [0] here remove need for second loop?

# convert list of objects to a list of lists

for i in range(0, len(y_train_s)):
    y_train2_s[i] = y_train2_s[i].tolist()

# encode y_train with vector of 1's and 0's

y_train_encoded_s = encode_baskets(y_train2_s, balanced_departments)

    
# ---------------- HANDLE x_train --------------------------------------------
        
# split each sub array within x_train
x_train2_s = x_train_s
for i in range(0, len(x_train_s)):
    new_array = []
    
    for j in range(0, len(x_train_s[i])):
        split = np.char.split(x_train2_s[i][j].tolist(), sep = ',')[0]
        new_array.append(split)
    
    x_train2_s[i] = new_array

# encode x_train with vector of 1's and 0's
 
x_train_encoded_s = encode_x_train(x_train_s, balanced_departments)
    
# pad sequences needs list of integers, can be done last 

for i in range(0, len(x_train_encoded_s)):
    
    while len(x_train_encoded_s[i]) < 10:
        x_train_encoded_s[i].insert(0, np.zeros(8, dtype=int))


x_train_s , y_train_encoded_s  = np.array(x_train_s), np.array(y_train_encoded_s) #85988 x 20 x 75 for X_train, first line is time = 60, each column are previous 59 values leading up to that day

# Reshaping

#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  #anytime you want to add a new dimension you need to use the reshape function
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
import tensorflow as tf
import keras.backend as K
import keras_metrics as km



classifier4 = Sequential()


# Adding the first LSTM layer and some dropout regularization

classifier4.add(LSTM(units = 50, return_sequences = True, activation = 'relu', input_shape = (x_train_s.shape[1], 8)))
classifier4.add(Dropout(rate = .20))

# Adding the second LSTM layer and some dropout regularization

classifier4.add(LSTM( units = 50, return_sequences = True ))
classifier4.add(Dropout( rate = .20 ))

# Adding the third LSTM layer and some dropout regularization

classifier4.add(LSTM( units = 50, return_sequences = True ))
classifier4.add(Dropout( rate = .20 ))

# Adding the fourth LSTM layer and some dropout regularization

classifier4.add(LSTM( units = 50 ))
classifier4.add(Dropout( rate = .20 ))

# Adding the output layer

classifier4.add(Dense( units = 8, activation = 'sigmoid'))

# Compiling the RNN


# try with hamming loss, hn_multilabel_loss, binary accuracy, f1 score
classifier4.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=[hamming_loss])  #Adam optimizer vs RMSprop optimizer (see keras.io/optimizers), binary cross entropy for classification

# Fit the network to training data

epochs = 35
batch_size = 300

classifier4.fit(x_train_s, y_train_encoded_s, epochs = epochs, batch_size = batch_size, callbacks=[plot_losses], validation_split = 0.2) #choose epochs value where you observe convergence of the loss



#history = classifier.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[plot_losses], sample_weight = )

predictions = classifier4.predict(x_test)
score = classifier4.evaluate(x_train_s, y_train_encoded_s, verbose=1)
print('Train loss:', score[0])
print('Train hamming:', score[1])
score_test = classifier3.evaluate(x_test, y_test_encoded, verbose=1)
print('Test loss:', score_test[0])
print('Test hamming:', score_test[1])



from collections import Counter

pred_sizes = []
act_sizes = []
for i in range(0, len(predictions)):
    pred_sizes.append(sum(predictions[i]))
    act_sizes.append(sum(y_test_encoded[i]))

pred_sizes_d = dict(Counter(pred_sizes))
act_sizes_d = dict(Counter(act_sizes))

plt.bar(pred_sizes_d.keys(), pred_sizes_d.values())
plt.bar(act_sizes_d.keys(), act_sizes_d.values())






class_distribution = dict.fromkeys(balanced_departments, 0)

for i in range(0, len(y_train_encoded_s)):
    for j in range(0, len(balanced_departments)):
        if y_train_encoded_s[i][j] == 1:
        # get the department corresponding to that "i" value:
            winner = balanced_departments[j]
        # add to dictionary
            class_distribution[winner] += 1

plt.barh(list(class_distribution.keys()), list(class_distribution.values()))
            
class_distribution_test = dict.fromkeys(balanced_departments, 0)

for i in range(0, len(y_test_encoded)):
    for j in range(0, len(balanced_departments)):
        if y_test_encoded[i][j] == 1:
        # get the department corresponding to that "i" value:
            winner = balanced_departments[j]
        # add to dictionary
            class_distribution_test[winner] += 1
            
            
prediction_distribution = dict.fromkeys(balanced_departments, 0)

for i in range(0, len(y_test_encoded)):
    for j in range(0, len(balanced_departments)):
        if predictions2[i][j] == 1:
        # get the department corresponding to that "i" value:
            winner = balanced_departments[j]
        # add to dictionary
            prediction_distribution[winner] += 1
import matplotlib.pyplot as plt
from matplotlib.figure import autofmt_xdate()
import seaborn as sns


plt.barh(list(prediction_distribution.keys()), list(prediction_distribution.values()))
plt.barh(list(class_distribution_test.keys()), list(class_distribution_test.values()))
       

counts = list(class_distribution_test.values())
bars = list(class_distribution_test.keys())
y_pos = len(bars)
plt.bar(y_pos, counts)
plt.xticks(y_pos, bars, rotation=90)










from sklearn.utils.class_weight import compute_class_weight
y = [1,1,1,1,1,1,12,3,4,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6]

class_weights = compute_class_weight('balanced', np.unique(y), y)

def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score



def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))





























    
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()   
    
    
    
from keras.layers import Conv2D, MaxPooling2D 
from keras.optimizers import Adam   
    
clf = Sequential()

clf.add(Dropout(0.3))
clf.add(Dense(units=75, activation='relu', input_shape = (x_train_s.shape[0],x_train_s.shape[1], 75)))
clf.add(Dropout(0.6))
clf.add(Dense(units=75, activation='relu'))
clf.add(Dropout(0.6))
clf.add(Dense(units=75, activation='relu'))
clf.add(Dropout(0.6))
clf.add(Dense(75, activation='sigmoid'))

clf.compile(optimizer=Adam(), loss='binary_crossentropy')

clf.fit(xt, yt, batch_size=64, nb_epoch=300, validation_data=(xs, ys), class_weight=W, verbose=0)
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





def find_max_sequence(x_train):
    maxlen = 0
    for i in range(0, len(x_train)):
        if maxlen < len(x_train[i]):
            maxlen = len(basket)
    return maxlen




def encode_x_train(array):
    
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            
            encoded_basket = np.zeros(len(unique_classes), dtype=int)
            for item in array[i][j]:
                index = unique_classes.index(item)
                encoded_basket[index] = 1
            array[i][j] = encoded_basket
    
    return array


def format_baskets(np_array):
	for i in range(0, len(np_array)):
		np_array[i] = np.char.split(np_array[i][0], sep=',')
	
	layer = list(chain.from_iterable(np_array))
	return layer


def encode_baskets(array_of_baskets):

	array_of_encoded_baskets = []
	for basket in array_of_baskets:
		encoded_basket = np.zeros(len(unique_classes), dtype=int)

		for item in basket:
			index = unique_classes.index(item)
			encoded_basket[index] = 1

		array_of_encoded_baskets.append(encoded_basket)

	return array_of_encoded_baskets


if __name__ == '__main__':
	output_layer = format_baskets('output_layer.csv')
	input_layer = format_baskets('input_classes_full.csv')

	output_vector = encode_baskets(output_layer)
	input_vector = encode_baskets(input_layer)