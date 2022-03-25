import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf

def pred(X, y):
    model = Sequential()

    model.add(Dense(10, activation="relu", input_dim=8))
    model.add(Dense(1, activation="linear", input_dim=10))

    # Compile model
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    tf.keras.optimizers.SGD(lr=0.08, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])

    # Fit model
    model.fit(X, y, epochs=500, batch_size=500, verbose=0)

    # Evaluate model
    scores = model.evaluate(X, y, verbose=0)
    print(scores)
    # rmseList.append(scores[0])
    # print("Fold :", i, " RMSE:", scores[0])