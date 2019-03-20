import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler


def rescale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def print_shapes(X_train, X_test, y_train, y_test):
    print('X_train.shape:\t', X_train.shape)
    print('y_train.shape:\t', y_train.shape)
    print('X_test.shape:\t', X_test.shape)
    print('y_test.shape:\t', y_test.shape)


def load_usps():
    print('Loading usps...')

    with h5py.File('data/usps.h5', 'r') as hf:
        train = hf.get('train')
        test = hf.get('test')

        X_train = train.get('data')[:]
        y_train = train.get('target')[:]
        X_test = test.get('data')[:]
        y_test = test.get('target')[:]

    X_train, X_test = rescale(X_train, X_test)
    print_shapes(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def load_letter():
    print('Loading letter...')
    dataset = pd.read_csv('data/letter-recognition.data', header=None)

    data = dataset.loc[:, 1:]
    data = np.array(data)

    target = dataset[0]
    target = target.apply(lambda x: ord(x) - ord('A'))
    target = np.array(target)

    X_train, X_test, y_train, y_test = train_test_split(data, target)

    X_train, X_test = rescale(X_train, X_test)
    print_shapes(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def load_satim():
    print('Loading satim...')
    train = pd.read_csv('data/sat.trn', sep=' ', header=None)
    test = pd.read_csv('data/sat.tst', sep=' ', header=None)

    train = np.array(train)
    X_train = train[:, :36]
    y_train = train[:, 36]

    test = np.array(test)
    X_test = test[:, :36]
    y_test = test[:, 36]

    X_train, X_test = rescale(X_train, X_test)
    print_shapes(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def load_dna():
    print('Loading dna...')
    X_train, y_train = load_svmlight_file('data/dna.scale.tr')
    X_train = X_train.A
    X_test, y_test = load_svmlight_file('data/dna.scale.t')
    X_test = X_test.A

    X_train, X_test = rescale(X_train, X_test)
    print_shapes(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test
