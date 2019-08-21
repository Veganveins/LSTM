# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:00:54 2019

@author: JJ5JXT
"""
# -----------------------     Part 1 DATA PRE-PROCESSING     -----------------

import pandas as pd
import numpy as np
#from itertools import chain
from imdb_example.py import encode_baskets, encode_x_train, plot_losses
from statistics import mean

basket_training = pd.read_csv("C:\\Users\\jj5jxt\\Desktop\\june\\x_train.csv")

basket_test = pd.read_csv("C:\\Users\\jj5jxt\\Desktop\\june\\x_test_proposed.csv")

#ilinks = basket_training.ILINK.unique() # get unique list of ILINKs
ilinks_test = basket_test.ILINK.unique()

#x_train = []   # some sequence of a customers baskets
#y_train	= []   # the next basket given a sequence of a customer's baskets

x_test = []
y_test = []

for ilink in ilinks_test:
    
    #append_field = basket_training.loc[basket_training['ILINK'] == ilink].iloc[:,2:3].values  # store a list of baskets for each ilink
    #samples = len(basket_training.loc[basket_training['ILINK'] == ilink]['Concat_CLASS_NAME']) # count how many total baskets exist in each ilink's unique sequence
    
    append_field = basket_test.loc[basket_test['ILINK'] == ilink].iloc[:,2:3].values  # store a list of baskets for each ilink
    samples = len(basket_test.loc[basket_test['ILINK'] == ilink]['Concat_CLASS_NAME']) # count how many total baskets exist in each ilink's unique sequence
    
    
    for i in range(1,samples):
    
        #x_train.append(append_field[0:i])  # for each basket sequence of length n, save each sequence (1 -> 2, 1:2 -> 3, 1:3 -> 4 ... 1:n-1 -> n)
        #y_train.append(append_field[i])    # store the appropriate y for each sub sequence

        x_test.append(append_field[0:i])  # for each basket sequence of length n, save each sequence (1 -> 2, 1:2 -> 3, 1:3 -> 4 ... 1:n-1 -> n)
        y_test.append(append_field[i])    # store the appropriate y for each sub sequence

# ---------------- HANDLE y_train --------------------------------------------

    
# create a method to convert y_train elements to arrays rather than strings with np.char.split

y_train2 = y_train
for i in range(0, len(y_train)):   
     y_train2[i] = np.char.split(y_train[i][0], sep = ',')   # would adding [0] here remove need for second loop?

# convert list of objects to a list of lists

for i in range(0, len(y_train)):
    y_train2[i] = y_train2[i].tolist()

# encode y_train with vector of 1's and 0's

y_train_encoded = encode_baskets(y_train2, rebalanced_departments)

    
# ---------------- HANDLE x_train --------------------------------------------
        
# split each sub array within x_train
x_train2 = x_train
for i in range(0, len(x_train)):
    new_array = []
    
    for j in range(0, len(x_train[i])):
        split = np.char.split(x_train2[i][j].tolist(), sep = ',')[0]
        new_array.append(split)
    
    x_train2[i] = new_array

# encode x_train with vector of 1's and 0's
 
x_train_encoded = encode_x_train(x_train, rebalanced_departments)
    
# pad sequences needs list of integers, can be done last 

for i in range(0, len(x_train_encoded)):
    
    while len(x_train_encoded[i]) < 20:
        x_train_encoded[i].insert(0, np.zeros(75, dtype=int))


x_train , y_train_encoded  = np.array(x_train), np.array(y_train_encoded) #1198 x 60 for X_train, first line is time = 60, each column are previous 59 values leading up to that day


#-------------------- TEST Y ----------------------------------------------
# create a method to convert y_train elements to arrays rather than strings with np.char.split

y_test2 = y_test
for i in range(0, len(y_test)):   
     y_test2[i] = np.char.split(y_test[i][0], sep = ',')   # would adding [0] here remove need for second loop?

# convert list of objects to a list of lists

for i in range(0, len(y_test)):
    y_test2[i] = y_test2[i].tolist()

# encode y_train with vector of 1's and 0's

y_test_encoded = encode_baskets(y_test2, balanced_departments)

    
# ---------------- TEST X --------------------------------------------
        
# split each sub array within x_train
x_test2 = x_test
for i in range(0, len(x_test)):
    new_array = []
    
    for j in range(0, len(x_test[i])):
        split = np.char.split(x_test2[i][j].tolist(), sep = ',')[0]
        new_array.append(split)
    
    x_test2[i] = new_array

# encode x_train with vector of 1's and 0's
 
x_test_encoded = encode_x_train(x_test, balanced_departments)
    
# pad sequences needs list of integers, can be done last 

for i in range(0, len(x_test_encoded)):
    
    while len(x_test_encoded[i]) < 10:
        x_test_encoded[i].insert(0, np.zeros(8, dtype=int))


x_test , y_test_encoded  = np.array(x_test), np.array(y_test_encoded) #1198 x 60 for X_train, first line is time = 60, each column are previous 59 values leading up to that day



# ---------------------------------    PART 2 BUILD RNN    -------------------
#(layers, neurons, LSTM, output, dropout regularization to avoid overfitting)
                                        
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
import tensorflow as tf
from scipy.stats import pearsonr
from collections import Counter


classifier = Sequential()


# Adding the first LSTM layer and some dropout regularization

classifier.add(LSTM(units = 50, return_sequences = True, activation = 'elu', input_shape = (x_train.shape[1], 75)))
#classifier.add(Dropout(rate = .20))

# Adding the second LSTM layer and some dropout regularization

classifier.add(LSTM( units = 50, return_sequences = True ))
#classifier.add(Dropout( rate = .20 ))

# Adding the third LSTM layer and some dropout regularization

classifier.add(LSTM( units = 50, return_sequences = True ))
#classifier.add(Dropout( rate = .20 ))

# Adding the fourth LSTM layer and some dropout regularization

classifier.add(LSTM( units = 50 ))
#classifier.add(Dropout( rate = .20 ))

# Adding the output layer

classifier.add(Dense( units = 75, activation = 'sigmoid'))

# Compiling the RNN

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=[hamming_loss])  #Adam optimizer vs RMSprop optimizer (see keras.io/optimizers), binary cross entropy for classification

#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the network to training data

epochs = 5
batch_size = 200



classifier.fit(x_train, y_train_encoded, epochs = epochs, batch_size = batch_size, callbacks=[plot_losses], validation_split = 0.2) #choose epochs value where you observe convergence of the loss



history = classifier.fit(x_train, y_train_encoded, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[plot_losses])
score = classifier.evaluate(x_test[0:10], y_test_encoded[0:10], batch_size=batch_size, verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

predictions = classifier4.predict(x_test)
#predictions[1]
#y_test[1]
#y_test_encoded[1]

predictions2 = predictions
for i in range(0, len(predictions2)):
    
    # scale all the prediction values
    max_confidence = max(predictions2[i])
    
    # reset the prediction values to either 0 or 1
    for j in range(0, len(predictions2[i])):
        if predictions2[i][j] >= .9*max_confidence:
            predictions2[i][j] = 1
        else:
            predictions2[i][j] = 0
            
            
# calculate hamming losses
losses = []
perfect = []
for i in range(0, len(predictions2)):
    loss = hamming_loss_list(y_test_encoded[i], predictions2[i])
    losses.append(loss)
    if loss == .25:
        perfect.append(i)
    

# count perfect (16%)
total = sum(Counter(losses).values()) 
a = Counter(losses)
b = {k: v/total for k,v in a.items()}
for key in sorted(b.keys()):
    print("%s: %s" % (key, b[key]))
    
plt.bar(b.keys(), b.values())

# .9 -> .1599
# .8 -> .1422
# .89 -> .1576
# .95 -> .1705
sums = []
for index in perfect:
    sums.append(sum(y_test_encoded[index]))
    
    

    

































# data leak accuracy

for i in range(0,9):    
    print('Max Index True',np.argmax(y_test_encoded[i]))
    print('Max Index Predicted:', np.argmax(predictions[i]))
    
    
predictions_reset = predictions

for i in range(0, len(predictions)):
    #prediction_indices = predictions[i].argsort()[-(sum(y_test_encoded[i])):][::-1]
    prediction_indices = predictions[i].argsort()[-3:][::-1]
    for j in range(0,16):
        if j in prediction_indices:
            predictions_reset[i][j] = 1
        else:
            predictions_reset[i][j] = 0
    
predictions_reset = list(np.int_(predictions_reset))    
accuracy = []

   
for i in range(0, len(predictions)):
    overlap = len(np.intersect1d(np.where(predictions_reset[i] ==1)[0],np.where(y_test_encoded[i] == 1)[0]))
    accuracy.append(overlap/sum(y_test_encoded[i]))
    
mean(accuracy)   
  

scaled_loss = []
pure_loss = []
for i in range(0, len(predictions)):
    save = np.interp(predictions[i], (predictions[i].min(), predictions[0].max()), (0,1))
    pure_loss.append(hamming_loss_list(y_test_encoded[i], predictions[i]))
    scaled_loss.append(hamming_loss_list(y_test_encoded[i], save))

  
def hamming_loss(y_true, y_pred):
    return K.mean(y_true*(1-y_pred) + (1-y_true)*y_pred)

def hn_multilabel_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

#pearson accuracy

accuracy = []   
for i in range(0, len(predictions)):
    save = np.interp(predictions[i], (predictions[i].min(), predictions[i].max()), (-1, 1))
    accuracy.append(pearsonr(save, y_test_encoded[i])[0])

accuracy = np.array(accuracy)
accuracy = (accuracy+1)/2
    
#The precision, which, for class C, is the ratio of examples labelled with class C that are predicted to have class C.
#The recall, which, for class C, is the ratio of examples predicted to be of class C that are in fact labelled with class C   

def pearson_accuracy(y_pred, y_true):
    
    preds = tf.norm(y_pred)
    
    accuracy = []
    
    for i in range(0, len(preds)):
        save = np.interp(preds[i], preds[i].min(), preds[i].max(), (-1,1))
        accuracy.append(pearsonr(save, real[i])[0])
        
    accuracy = np.array(accuracy)
    accuracy = (accuracy+1)/2
    
    tensor = tf.constant(accuracy) 
    
    return K.mean(tensor)
        

def f1_score(y_true, y_pred):
	"""
	Compute the micro f(b) score with b=1.
	"""
	y_true = tf.cast(y_true, "float32")
	y_pred = tf.cast(y_pred, "float32")

    #scale the tensor from 0 to 1
	y_pred = tf.math.divide(
		tf.subtract(
			y_pred,
			tf.reduce_min(y_pred)),
		tf.subtract(
				tf.reduce_max(y_pred),
				tf.reduce_min(y_pred))
	)


	y_pred = tf.cast(tf.round(y_pred), "float32") # implicit 0.5 threshold via tf.round
	y_correct = y_true * y_pred


	sum_true = tf.reduce_sum(y_true, axis=1)
	sum_pred = tf.reduce_sum(y_pred, axis=1)
	sum_correct = tf.reduce_sum(y_correct, axis=1)


	precision = sum_correct / sum_pred
	recall = sum_correct / sum_true
	f_score = 2 * precision * recall / (precision + recall)
	f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)


	return tf.reduce_mean(f_score)

#pearson accuracy

scaled_preds = []   
for i in range(0, len(predictions)):
    save = np.interp(predictions[i], (predictions[i].min(), predictions[i].max()), (0, 1))
    scaled_preds.append(save)

    
    
    
    
    
    
    
    
    
    
    
    
    