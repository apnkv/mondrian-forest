import time
import numpy as np
from sklearn.metrics import accuracy_score


def clf_fit_predict(clf, X_batch, y_batch, X_test, y_test):
    t = time.time()
    clf.fit(X_batch, y_batch)
    ft = time.time() - t

    tr_acc = accuracy_score(y_batch, clf.predict(X_batch))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return ft, tr_acc, test_acc


def classical_rf_refit(clf, dataset, n_batches):
    X_train, X_test, y_train, y_test = dataset

    fit_time = []
    train_accuracy = []
    test_accuracy = []

    n_samples = X_train.shape[0]
    batch_size = n_samples // n_batches

    for i in range(n_batches):
        end = min((i + 1) * batch_size, n_samples)
        X_batch = X_train[0:end]
        y_batch = y_train[0:end]

        ft, tr_acc, test_acc = clf_fit_predict(clf, X_batch, y_batch, X_test, y_test)
        fit_time.append(ft)
        train_accuracy.append(tr_acc)
        test_accuracy.append(test_acc)

    return fit_time, train_accuracy, test_accuracy


def classical_rf_window(clf, dataset, n_batches, h=1):
    X_train, X_test, y_train, y_test = dataset

    fit_time = []
    train_accuracy = []
    test_accuracy = []

    n_samples = X_train.shape[0]
    batch_size = n_samples // n_batches

    for i in range(n_batches):
        end = min((i + 1) * batch_size, n_samples)
        start = max(0, (i + 1 - h) * batch_size)

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        ft, tr_acc, test_acc = clf_fit_predict(clf, X_batch, y_batch, X_test, y_test)
        fit_time.append(ft)
        train_accuracy.append(tr_acc)
        test_accuracy.append(test_acc)

    return fit_time, train_accuracy, test_accuracy


def classical_rf_incremental(clf, dataset, n_batches, new_frac=0.1):
    X_train, X_test, y_train, y_test = dataset

    fit_time = []
    train_accuracy = []
    test_accuracy = []

    n_samples = X_train.shape[0]
    batch_size = n_samples // n_batches

    n_new = int(new_frac * clf.n_estimators)

    for i in range(n_batches):
        start = max(0, i * batch_size)
        end = min((i + 1) * batch_size, n_samples)

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        t = time.time()
        clf.fit(X_batch, y_batch)
        ft = time.time() - t

        tr_acc = accuracy_score(y_batch, clf.predict(X_batch))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        fit_time.append(ft)
        train_accuracy.append(tr_acc)
        test_accuracy.append(test_acc)

        clf.estimators_ = clf.estimators_[n_new:]

    return fit_time, train_accuracy, test_accuracy


def mondrian_rf_skgarden(clf, dataset, n_batches):
    X_train, X_test, y_train, y_test = dataset
    classes = np.unique(y_train)

    fit_time = []
    train_accuracy = []
    test_accuracy = []

    n_samples = X_train.shape[0]
    batch_size = n_samples // n_batches

    for i in range(n_batches):
        start = max(0, i * batch_size)
        end = min((i + 1) * batch_size, n_samples)

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        t = time.time()
        clf.partial_fit(X_batch, y_batch, classes)
        ft = time.time() - t

        tr_acc = accuracy_score(y_batch, clf.predict(X_batch))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        fit_time.append(ft)
        train_accuracy.append(tr_acc)
        test_accuracy.append(test_acc)

    return fit_time, train_accuracy, test_accuracy
