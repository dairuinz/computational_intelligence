import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D
from keras import regularizers
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow import keras
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)       #splits data into train, test

    hl_n = 0
    answer = input('Select number of hidden layer neurons:\n\t 1. equal to output neurons\n\t '
                   '2. equal to (output neurons + input neurons) / 2\n\t 3. equal to output neurons + input neurons\n')
    if answer == '1':
        hl_n = 20                   #hidden layer neurons number
    elif answer == '2':
        hl_n = int((20 + 8519) / 2)
    elif answer == '3':
        hl_n = 8519

    # hl_n2 = 0             #SECOND hidden layer neurons number
    # answer2 = input('Select number of the SECOND hidden layer neurons:\n\t 1. equal to first hidden layer\n\t '
    #                 '2. equal to (output neurons + input neurons) / 4\n\t 3. equal to output neurons\n')
    # if answer2 == '1':
    #     hl_n2 = int((20 + 8519) / 2)
    # elif answer2 == '2':
    #     hl_n2 = int((20 + 8519) / 4)
    # elif answer2 == '3':
    #     hl_n2 = 20

    # opt = keras.optimizers.Adam(learning_rate=0.01)                   #adam
    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.2)       #sgd

    for i, (train, test) in enumerate(kfold.split(X)):
        model = Sequential()
        # model.add(Dense(8519, input_shape=(8519,), activation='softmax', kernel_regularizer=regularizers.l2(0.1)))      #kernel l2 regulirization
        model.add(Dense(8519, input_shape=(8519,), activation='relu'))          #input
        model.add(Dense(hl_n, activation='relu'))                       #hidden
        # model.add(Dense(hl_n2, activation='relu'))                         #2nd hidden
        model.add(Dense(20, activation='sigmoid'))                      #output

        model.compile(
            optimizer=opt,
            # loss='binary_crossentropy',
            loss='mean_squared_error',
            # metrics=[rmse]
            metrics=['binary_accuracy']
        )

        es = EarlyStopping(             #early stopping
            monitor='val_loss',
            patience=5,
            verbose=0,
            mode='min'
        )

        mc = ModelCheckpoint(       #model checkpoint to save best model each epoch
            'best_m.h5',
            monitor='binary_accuracy',
            mode='max',
            verbose=1,
            save_best_only=True
        )

        history = model.fit(X_train, y_train,           #model fit
                  validation_data=(X_test, y_test),
                  epochs=50,
                  callbacks=[es, mc],
                  verbose=0)

        print('val_loss, val_acc: ', model.evaluate(X_test, y_test), sep='')        #model evaluation

        scores = model.evaluate(X_test, y_test, verbose=0)
        rmseList.append(scores[0])
        print("Fold :", i+1, " loss", scores[0])

        plt.plot(history.history['binary_accuracy'])            #plots accuracy, val_accuracy/epoch graph
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

