import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow import keras
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def scale(X):
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)     #normalization      #slight improvement
    X = tf.keras.utils.normalize(X, axis=1)     #normalization

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)     #standardization      #poor results, not needed

    # scaler = StandardScaler(with_std=False)
    # X = scaler.fit_transform(X)     #centering      #poor results, not needed

    return X

def pred(X, y):
    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    rrseList = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # y_test = y_test.transpose()
    # X_test = X_train.transpose()
    # print(y_test)

    hl_n = 0
    answer = input('Select number of hidden layer neurons:\n\t 1. equal to output neurons\n\t '
                   '2. equal to (output neurons + input neurons) / 2\n\t 3. equal to output neurons + input neurons\n')
    if answer == '1':
        hl_n = 20
    elif answer == '2':
        hl_n = int((20 + 8519) / 2)
    elif answer == '3':
        hl_n = 8519

    for i, (train, test) in enumerate(kfold.split(X)):
        model = Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(Dense(hl_n, activation='relu'))
        # model.add(Dense(hl_n, activation='relu'))
        model.add(Dense(20, input_shape=(8519,), activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=opt,
            # loss='binary_crossentropy',
            loss='mean_squared_error',
            # metrics=[rmse]
            metrics=['accuracy']
        )

        es = EarlyStopping(             #early stopping
            monitor='val_loss',
            patience=30,
            verbose=1,
            mode='min'
        )

        mc = ModelCheckpoint(
            'best_m.h5',
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_best_only=True
        )

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[es, mc], verbose=2)

        print('val_loss, val_acc: ', model.evaluate(X_test, y_test), sep='')

        scores = model.evaluate(X_test, y_test, verbose=0)
        rmseList.append(scores[0])
        print("Fold :", i+1, " loss", scores[0])

