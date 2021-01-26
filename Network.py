import warnings

warnings.filterwarnings('ignore')

import os
import numpy
import re
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_sample_data(template_path):
    filename = template_path.split(os.sep)[-1]
    data = re.findall("\d*\d", filename)
    signs = re.findall('[+ -]', filename)
    vertical = int(data[2]) if signs[0] == '+' else -int(data[2])
    horizontal = int(data[3]) if signs[1] == '+' else -int(data[3])

    return vertical, horizontal


def _get_samples_paths(codebook_path):
    dir_list = os.listdir(codebook_path)
    template_paths = []

    for directory in dir_list:
        directory = os.path.join(codebook_path, directory)
        template_paths.append(directory)

    return template_paths


def prepare_data(codebook_path='Codebook'):
    # columns = [*range(4, 1770, 1)]
    # dataset = pd.DataFrame(columns=columns, index=[])
    dataset = pd.DataFrame()
    final = []

    list_of_names = _get_samples_paths(codebook_path)
    for sample in list_of_names:
        vertical, horizontal = get_sample_data(sample)
        hogs = numpy.load(sample)
        list_of_hogs = []
        for x in hogs:
            list_of_hogs.append(x[0])

        list_of_hogs.append(horizontal)
        list_of_hogs.append(vertical)
        final.append(list_of_hogs)

    dataset = dataset.append(final, ignore_index=True)
    print(dataset)
    return dataset


def get_example(example_path):
    vertical, horizontal = get_sample_data(example_path)
    hogs = numpy.load(example_path)
    list_of_hogs = []
    for x in hogs:
        list_of_hogs.append(x[0])

    list_of_hogs.append(horizontal)
    list_of_hogs.append(vertical)
    example = list_of_hogs[:-2]
    return example


def create_model():
    model = Sequential()
    model.add(Dense(144, input_dim=144, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def loss_history_model(model, dataset):
    early_stopping_monitor = EarlyStopping(patience=10)

    X = dataset[:, :-2]
    Y = dataset[:, -2]
    scalar = MinMaxScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    history = model.fit(X, Y, validation_split=0.3, epochs=200, batch_size=1, verbose=1,
                        callbacks=[early_stopping_monitor])

    scores = model.evaluate(X, Y, verbose=1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return None


class Network:
    def __init__(self):
        '''
        constructor function for the class
        '''
