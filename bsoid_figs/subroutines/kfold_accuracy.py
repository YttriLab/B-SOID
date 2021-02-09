import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from operator import itemgetter
from utilities.load_data import appdata
from utilities.save_data import results
import sys, getopt
from ast import literal_eval

def generate_kfold(path, name, k):
    appdata_ = appdata(path, name)
    f_10fps_sub, train_embeddings = appdata_.load_embeddings()
    min_cluster_range, assignments, soft_clusters, soft_assignments = appdata_.load_clusters()
    y = assignments[assignments >= 0]
    X = f_10fps_sub[assignments >= 0, :]
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    accuracy_data = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        accuracy_vec = []
        predictions = classifier.predict(X_test)
        for i in range(len(np.unique(y_test))):
            accuracy_vec.append(len(np.argwhere((predictions - y_test == 0) & (y_test == i)))
                                / len(np.argwhere(y_test == i)))
        accuracy_data.append(np.array(accuracy_vec))
    accuracy_data = np.array(accuracy_data)
    return accuracy_data


def reorganize_accuracy(accuracy_data, order):
    accuracy_ordered = []
    for i in range(len(accuracy_data)):
        accuracy_ordered.append(itemgetter(order)(accuracy_data[i]))
    return accuracy_ordered


def main(argv):
    path = None
    name = None
    k = None
    order = None
    vname = None
    options, args = getopt.getopt(
        argv[1:],
        'p:f:k:o:v:',
        ['path=', 'file=', 'kfold=', 'order=', 'variable='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-k', '--kfold'):
            k = option_value
        elif option_key in ('-o', '--order'):
            order = option_value
        elif option_key in ('-v', '--variable'):
            vname = option_value

    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('K-FOLD   :', k)
    print('ORDER    :', order)
    print('VARIABLE   :', vname)
    print('*' * 50)
    print('Computing...')
    accuracy_data = generate_kfold(path, name, int(k))
    accuracy_ordered = reorganize_accuracy(accuracy_data, literal_eval(order))
    results_ = results(path, name)
    results_.save_sav([accuracy_data, accuracy_ordered], vname)


if __name__ == '__main__':
    main(sys.argv)

