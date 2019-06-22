# -*- coding: utf-8 -*-

from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import optimizers


def data():
    import pandas as pd

    data = pd.read_csv(r'../input/data.csv', header=0, sep=',')
    data = data.drop(['id', 'Unnamed: 32'], 1)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data['diagnosis'])
    y = le.transform(data['diagnosis'])

    data = data.drop('diagnosis', 1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=777)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=777)

    from sklearn.preprocessing import StandardScaler

    for i in X_train.columns:
        scaler = StandardScaler()
        scaler.fit(X_train[i].values.reshape(-1, 1))
        X_train[i] = scaler.transform(X_train[i].values.reshape(-1, 1))
        X_val[i] = scaler.transform(X_val[i].values.reshape(-1, 1))
        X_test[i] = scaler.transform(X_test[i].values.reshape(-1, 1))

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(X_train, y_train, X_val, y_val):
    from keras import models
    from keras import layers
    import numpy as np

    model = models.Sequential()
    model.add(layers.Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7)])}}, input_shape=(len(data.columns),)))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5)])}}))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(layers.Dense(1, activation='sigmoid'))

    from keras import callbacks
        
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)


    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs={{choice([25, 50, 75, 100])}},
              batch_size={{choice([16, 32, 64])}},
              validation_data=(X_val, y_val),
              callbacks=[reduce_lr])

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# if __name__ == '__main__':

#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=15,
#                                           trials=Trials())
#     X_train, X_val, X_test, y_train, y_val, y_test = data()
#     print("Evalutation of best performing model:")
#     print(best_model.evaluate(X_test, y_test))
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)

#     best_model.save('breast_cancer_model.h5')