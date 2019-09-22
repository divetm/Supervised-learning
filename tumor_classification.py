import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load our dataset
train_data = pd.read_csv("tumor_classification_data.csv", delimiter=";")

# extract the images and labels from the dictionary object
y = train_data.pop('malignant').values
ids = train_data.pop('id').values
X = train_data

# transform y into a column
y = y.T

# shuffle to avoid underlying distributions
X, y = shuffle(X, y, random_state=26)

# split set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

# learning model definition (here is where we'll be tuning hyper-parameters)
decision_tree_classifier = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None,
                                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                  max_features=None, random_state=None, max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                                  presort=False)

neural_network_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=0.0001,
                                          batch_size="auto", learning_rate="constant", learning_rate_init=1,
                                          power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                          verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                          early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                          epsilon=1e-08, n_iter_no_change=10)

boosting_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="gini", splitter="best",
                                         max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                         max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                         class_weight=None, presort=False), n_estimators=50, learning_rate=1.0,
                                         algorithm="SAMME.R", random_state=None)

svm_classifier = SVC(C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, shrinking=True,
                     probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                     decision_function_shape="ovr", random_state=None)

k_neighbors_classifier = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto", leaf_size=30, p=2,
                                              metric="minkowski", metric_params=None, n_jobs=None)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def get_learning_curves(classifier):
    title = "{}_Learning_Curves_for_Tumor_classification".format(classifier.__class__.__name__)
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cross_validation = ShuffleSplit(n_splits=100, test_size=0.2, random_state=26)
    plot_learning_curve(classifier, title, X_train, y_train, ylim=(0.0, 1.1), cv=cross_validation, n_jobs=4, train_sizes=np.linspace(0.03, 1.0, 33))
    plt.savefig("{}.png".format(title))


def get_validation_curve(classifier, param_name, param_range):
    cross_validation = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_scores, test_scores = validation_curve(classifier, X_train, y_train, param_name, param_range, groups=None,
                     cv=cross_validation, scoring=None, n_jobs=None, pre_dispatch="all", verbose=0,
                     error_score="raise -deprecating")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    title = "{}_{}_Validation_Curve_for_Tumor_classification".format(classifier.__class__.__name__, param_name)
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("{}.png".format(title))


# getting learning and validation curves
get_learning_curves(decision_tree_classifier)
get_validation_curve(decision_tree_classifier, "max_depth", np.linspace(1, 32, 32, endpoint=True, dtype=np.int64))
get_validation_curve(decision_tree_classifier, "min_samples_split", np.linspace(0.1, 1.0, 10, endpoint=True))
get_learning_curves(neural_network_classifier)
get_validation_curve(neural_network_classifier, "activation", np.asarray(["identity", "logistic", "tanh", "relu"]))
get_validation_curve(neural_network_classifier, "alpha", np.linspace(100, 1000000, 10, endpoint=True))
get_learning_curves(boosting_classifier)
get_validation_curve(boosting_classifier, "base_estimator__max_depth", np.linspace(1, 32, 32, endpoint=True, dtype=np.int64))
get_validation_curve(boosting_classifier, "base_estimator__min_samples_split", np.linspace(0.1, 1.0, 10, endpoint=True))
get_learning_curves(svm_classifier)
get_validation_curve(svm_classifier, "C", np.linspace(0.001, 10000, 10, endpoint=True))
get_validation_curve(svm_classifier, "degree", np.linspace(1, 1000000, 10, endpoint=True, dtype=np.int64))
get_learning_curves(k_neighbors_classifier)
get_validation_curve(k_neighbors_classifier, "n_neighbors", np.linspace(1, 363, 10, endpoint=True, dtype=np.int64))
get_validation_curve(k_neighbors_classifier, "algorithm", np.asarray(["auto", "ball_tree", "kd_tree", "brute"]))
get_validation_curve(k_neighbors_classifier, "p", np.asarray([1, 2], dtype=np.int64))

# preprocessing of the X_train dataset to lower computational complexity
X_train = preprocessing.scale(X_train)


# tuning hyperparameters by gridsearch + training + accuracy score
def grid_search_and_train_model(classifier, tuned_parameters):
    score = "accuracy"
    clf = GridSearchCV(classifier, tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


decision_tree_accuracy = grid_search_and_train_model(decision_tree_classifier, {"max_depth": [20], "min_samples_split": [0.7]})
neural_network_accuracy = grid_search_and_train_model(neural_network_classifier, {"activation": ["relu"], "hidden_layer_sizes": [(100,)], "alpha": [100000]})
boosting_accuracy = grid_search_and_train_model(boosting_classifier, {"n_estimators": [1000], "learning_rate": [1], "base_estimator__min_samples_split": [0.9], "base_estimator__max_depth": [2]})
svm_accuracy = grid_search_and_train_model(svm_classifier, {'degree': [1], 'C': [1000]})
k_neighbors_accuracy = grid_search_and_train_model(k_neighbors_classifier, {"n_neighbors": [3], "p": [1]})

classifiers = [1, 2, 3, 4, 5]
accuracy_scores = [decision_tree_accuracy, neural_network_accuracy, boosting_accuracy, svm_accuracy, k_neighbors_accuracy]

labels = ["Decision Tree", "Neural Network", "Boosting", "SVM", "KNN"]

plt.bar(classifiers, accuracy_scores, align='center')
plt.xticks(classifiers, labels)
plt.show()

# get learning curves for iterative algorithms (using the best parameters found with the GridSearch) over iterations
# Neural Network
nn_classifier = MLPClassifier(activation="relu", alpha=1000000, hidden_layer_sizes=(100,), learning_rate_init=1)
nn_classifier.fit(X_train, y_train)
plt.figure()
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.title("MLPClassifier Learning rate =" + str(1))
plt.plot(nn_classifier.loss_curve_)
plt.show()

# Boosted Decision Tree
boosted_tree_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="gini", splitter="best",
                                         max_depth=2, min_samples_split=0.9), n_estimators=1000, learning_rate=1)
boosted_tree_classifier.fit(X_train, y_train)
plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Boosted Tree Learning rate = 1, min_samples_split = 0.9")
plt.plot(list(boosted_tree_classifier.staged_score(X_train,y_train)))
plt.show()

