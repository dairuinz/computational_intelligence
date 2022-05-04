from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

vocab = pd.read_csv('vocabs.txt', sep=',', header=None)
vocab.columns = ['word', 'id']

# words.head()

sentences = pd.read_csv('train-data.dat', sep=',', header=None)
sentences.head()

dict = {}                                       #creates dictionary from vocabs file
with open('vocabs.txt') as file:
    for line in file:
        (key, value) = line.split(', ')
        dict[key] = int(value)

com = []
for line in df.sentence:
    sen = ''
    for word in line.split():
      for id in vocab.id:
        if int(word) == int(id):
          sen = sen + ' ' + str(vocab.word[id]) 
    # print(sen)
    com.append(sen)

print(com)

encoded_docs = [one_hot(d, 50) for d in com]
X = pad_sequences(encoded_docs, maxlen=10, padding='post')

from sklearn.model_selection import KFold, train_test_split
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
    hl_n = int((20 + 20) / 2)
elif answer == '3':
    hl_n = 20

opt = keras.optimizers.Adam(learning_rate=0.01)                   #adam
# opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.2)       #sgd

for i, (train, test) in enumerate(kfold.split(X)):
    model = Sequential()
    # model.add(Dense(20, input_shape=(20,), activation='softmax', kernel_regularizer=regularizers.l2(0.1)))
    # model.add(tf.keras.layers.Embedding(20, 64, input_length=30))       #input Embedding layer
    model.add(Dense(20, input_shape=(20,), activation='relu'))          #input
    model.add(Dense(hl_n, activation='relu'))                       #hidden
    # model.add(Dense(10, activation='relu'))                         #hidden
    model.add(Dense(20, activation='sigmoid'))                          #output

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        # loss='mean_squared_error',
        # metrics=[rmse]
        metrics=['accuracy']
    )
    
    es = EarlyStopping(             #early stopping
        monitor='val_loss',
        patience=5,
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
    
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              callbacks=[es, mc],
              verbose=0)
    
    print('val_loss, val_acc: ', model.evaluate(X_test, y_test), sep='')
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    rmseList.append(scores[0])
    print("Fold :", i+1, " loss", scores[0])
