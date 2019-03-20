import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from skgarden import MondrianForestClassifier
from data_loaders import load_usps, load_letter, load_satim, load_dna
from methods import classical_rf_refit, classical_rf_window, classical_rf_incremental, mondrian_rf_skgarden


def run_method_on_dataset(method, dataset, n_iter, n_batches, n_estimators, max_depth):
    mean_fit_time = []
    mean_train_acc = []
    mean_test_acc = []

    for i in range(n_iter):

        if method == 'classical_rf_refit':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_refit(clf, dataset, n_batches)

        elif method == 'classical_rf_window_1':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=1)

        elif method == 'classical_rf_window_3':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=3)

        elif method == 'classical_rf_window_5':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=5)

        elif method == 'classical_rf_increment_frac_0.1':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, warm_start=True)
            fit_time, train_acc, test_acc = classical_rf_incremental(clf, dataset, n_batches, new_frac=0.1)

        elif method == 'classical_rf_increment_frac_0.5':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, warm_start=True)
            fit_time, train_acc, test_acc = classical_rf_incremental(clf, dataset, n_batches, new_frac=0.5)

        elif method == 'extratrees_rf_refit':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_refit(clf, dataset, n_batches)

        elif method == 'extratrees_rf_window_1':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=1)

        elif method == 'extratrees_rf_window_3':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=3)

        elif method == 'extratrees_rf_window_5':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = classical_rf_window(clf, dataset, n_batches, h=5)

        elif method == 'extratrees_rf_increment_frac_0.1':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, warm_start=True)
            fit_time, train_acc, test_acc = classical_rf_incremental(clf, dataset, n_batches, new_frac=0.1)

        elif method == 'extratrees_rf_increment_frac_0.5':
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, warm_start=True)
            fit_time, train_acc, test_acc = classical_rf_incremental(clf, dataset, n_batches, new_frac=0.5)

        elif method == 'mondrian_rf_skgarden':
            clf = MondrianForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            fit_time, train_acc, test_acc = mondrian_rf_skgarden(clf, dataset, n_batches)

        mean_fit_time.append(fit_time)
        mean_train_acc.append(train_acc)
        mean_test_acc.append(test_acc)

    mean_fit_time = np.mean(mean_fit_time, axis=0)
    mean_train_acc = np.mean(mean_train_acc, axis=0)
    mean_test_acc = np.mean(mean_test_acc, axis=0)

    return mean_fit_time, mean_train_acc, mean_test_acc


def run_all_methods_on_dataset(dataset, ax, n_iter, n_batches, n_estimators, max_depth):
    methods = [
        'classical_rf_refit',
        'classical_rf_window_1',
        'classical_rf_window_3',
        'classical_rf_window_5',
        'classical_rf_increment_frac_0.1',
        'classical_rf_increment_frac_0.5',
        'extratrees_rf_refit',
        'extratrees_rf_window_1',
        'extratrees_rf_window_3',
        'extratrees_rf_window_5',
        'extratrees_rf_increment_frac_0.1',
        'extratrees_rf_increment_frac_0.5',
        'mondrian_rf_skgarden',
    ]

    for method in methods:

        mean_fit_time, mean_train_acc, mean_test_acc = run_method_on_dataset(method, dataset, n_iter, n_batches,
                                                                             n_estimators, max_depth)

        ax[0].plot(np.arange(n_batches), mean_test_acc, label=method, marker='o')
        ax[0].set(xlabel='batch num', ylabel='test accuracy')
        ax[0].legend()

        ax[1].plot(mean_fit_time, mean_test_acc, label=method, marker='o')
        ax[1].set(xlabel='fit time', ylabel='test accuracy')
        ax[1].set_xscale('log')
        ax[1].legend()


def run_all_methods_all_datasets(n_iter, n_batches, n_estimators, max_depth):
    datasets = {
        'usps': load_usps(),
        'letter': load_letter(),
        'satim': load_satim(),
        'dna': load_dna()
    }

    fig, ax = plt.subplots(2, 4, figsize=(25, 12))

    for i, name in enumerate(datasets):
        dataset = datasets[name]

        ax[0, i].set_title(name)
        sub_ax = (ax[0, i], ax[1, i])

        run_all_methods_on_dataset(dataset, sub_ax, n_iter, n_batches, n_estimators, max_depth)
