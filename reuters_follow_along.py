import keras
from keras.datasets import reuters

(xtrain, ytrain), (xtest, ytest) = reuters.load_data(num_words=None, test_split=0.2)
word_index = reuters.get_word_index(path="reuters_word_index.json")

print('# of Training Samples: {}'.format(len(xtrain)))
print('# of Test Samples: {}'.format(len(xtest)))

num_classes = max(ytrain) + 1
print('# of Classes: {}'.format(num_classes))
# of Training Samples: 8982
# of Test Samples: 2246
# of Classes: 46
index_to_word = {}
for key, value in word_index.items():
    index_to_word[value] = key
print(' '.join([index_to_word[x] for x in xtrain[0]]))
print(ytrain[0])

from keras.preprocessing.text import Tokenizer

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
xtrain = tokenizer.sequences_to_matrix(xtrain, mode='binary')
xtest = tokenizer.sequences_to_matrix(xtest, mode='binary')

ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)
print(xtrain[0])
print(len(xtrain[0]))

print(ytrain[0])
print(len(ytrain[0]))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.metrics_names)

batch_size = 32
epochs = 3

history = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
score = model.evaluate(xtest, ytest, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
