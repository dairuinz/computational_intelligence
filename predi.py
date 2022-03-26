import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def pred(X, y):
    model = Sequential()
    model.add(Dense(20, input_shape=(8519,), activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    model.fit(X, y, epochs=50)