import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def scale(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)     #normalization      #slight improvement

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)     #standardization      #poor results, not needed

    # scaler = StandardScaler(with_std=False)
    # X = scaler.fit_transform(X)     #centering      #poor results, not needed

    return X

def pred(X, y):
    model = Sequential()
    model.add(Dense(20, input_shape=(8519,), activation='sigmoid'))

    model.compile(
        optimizer='adam',
        # loss='categorical_crossentropy',
        loss='mean_squared_error',
        metrics=['accuracy']
        )

    model.fit(X, y, epochs=50)
