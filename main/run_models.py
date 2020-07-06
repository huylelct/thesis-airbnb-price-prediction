import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from sklearn import linear_model
from sklearn import feature_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import sklearn as sklearn
import multiprocessing

from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn import tree

import statsmodels.api as sm
from tensorflow.keras import optimizers
from sklearn.linear_model import SGDRegressor
# import statsmodels.api as sm


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1

import copy

# NN parameters
NUM_ITERATIONS = 300
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LASSO_COEF = 50.0
DECAY_RATE = 0.0001

plt.style.use("ggplot")
n_cores = multiprocessing.cpu_count()  # Getting the number of cores for multiprocessing


def get_gaussian_process_regressor():
    gp = GaussianProcessRegressor()
    return [gp], ['Gaussian Process']


def get_mlp_regressor(num_hidden_units=51):
    mlp = MLPRegressor(hidden_layer_sizes=num_hidden_units)
    return [mlp], ['Multi-Layer Perceptron']


def get_ensemble_models():
    grad = GradientBoostingRegressor(n_estimators=17, random_state=42, loss='lad', learning_rate=0.12, max_depth=10)
    classifier_list = [grad]
    classifier_name_list = ['Gradient Boost']
    return classifier_list, classifier_name_list


def print_evaluation_metrics(trained_model, trained_model_name, X_test, y_test, percent=True):
    print('--------- For Model: ', trained_model_name, ' --------- (Test Data)\n')
    predicted_values = trained_model.predict(X_test)
    # print(trained_model)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    print("PERCENT: ", percent)
    if percent:
        print("Mean absolute percent error: ",
              np.mean(np.abs((y_test - predicted_values) / y_test)) * 100)
    print("Median absolute error: ",
          metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))
    dfscores = pd.DataFrame(y_test)
    dfcolumns = pd.DataFrame(predicted_values)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    featureScores.to_csv("../data/" + trained_model_name + ".csv")
    saiso = []
    for i in range(len(y_test)):
        error = abs(y_test[i] - predicted_values[i])
        if error > 15:
            print(error, y_test[i], predicted_values[i])
        saiso.append(error)

    # print(featureScores)

    # fig, ax = plt.subplots()
    # data_x = saiso
    # bins = 80
    # if bins > 0:
    #     ax.hist(data_x, bins)
    #
    # ax.set_xlabel("Saiso")
    # ax.set_title(trained_model_name)
    #
    # plt.savefig("../data_visualization/" + trained_model_name + ".png")


def print_evaluation_metrics2(trained_model, trained_model_name, X_test, y_test):
    print('--------- For Model: ', trained_model_name, ' --------- (Train Data)\n')
    predicted_values = trained_model.predict(X_test)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    # print("Mean absolute percente error: ",
    #       np.mean(np.abs((y_test - predicted_values) / y_test)) * 100)
    print("Median absolute error: ",
          metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))
    # dfscores = pd.DataFrame(y_test)
    # dfcolumns = pd.DataFrame(predicted_values)
    #
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Specs', 'Score']
    # print(featureScores)


def svm(X_train, y_train, X_val, y_val):
    model = SVR(gamma=0.05, verbose=True)  # was empty #0.1 #the - best gamma 0.05, c=0.5
    model.fit(X_train, y_train)
    print(model)
    print_evaluation_metrics(model, "svm", X_val, y_val.values.ravel())
    print_evaluation_metrics2(model, "svm", X_train, y_train.values.ravel())


def LinearModel(X_train, y_train, X_val, y_val):
    regr = linear_model.LinearRegression(n_jobs=int(0.8 * n_cores)).fit(X_train, y_train)
    print_evaluation_metrics(regr, "linear model", X_val, y_val.values.ravel())
    print_evaluation_metrics2(regr, "linear model", X_train, y_train.values.ravel())

    return


def LinearModelRidge(X_train, y_train, X_val, y_val):
    regr = Ridge(alpha=7)  # 7
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_val.values)

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_val, y_pred))
    print('Variance score: %.2f' % r2_score(y_val, y_pred))
    print("R2:", sklearn.metrics.r2_score(y_val, y_pred))

    print_evaluation_metrics(regr, "Linear Model Ridge", X_val, y_val, False)
    print_evaluation_metrics2(regr, "Linear Model Ridge", X_train, y_train)
    return


def LinearModelLasso(X_train, y_train, X_val, y_val):
    regr = Lasso(alpha=0.5)  # 0.5
    regr.fit(X_train, y_train)

    print_evaluation_metrics(regr, "Linear Model Lasso", X_val, y_val, False)
    print_evaluation_metrics2(regr, "Linear Model Lasso", X_train, y_train)
    return


def simple_neural_network(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Dense(units=20, activation='relu', input_dim=len(X_train.values[0])))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=DECAY_RATE, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit(X_train, y_train, epochs=NUM_ITERATIONS, batch_size=BATCH_SIZE)
    print("finished fitting")
    print_evaluation_metrics(model, "NN", X_val, y_val)
    print_evaluation_metrics2(model, "NN", X_train, y_train)
    return


def TreebasedModel(X_train, y_train, X_val, y_val):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)

    classifier_list = [
        DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0, max_features=None,
                              random_state=500, max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, presort='auto'),
        GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                  learning_rate=0.12, loss='lad', max_depth=None,
                                  max_features=None, max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=2, min_samples_split=2,
                                  min_weight_fraction_leaf=0.004, n_estimators=20,
                                  n_iter_no_change=None, presort='auto',
                                  random_state=50, subsample=1.0, tol=0.0001,
                                  validation_fraction=0.1, verbose=0, warm_start=False),
        RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.004, max_features='auto',
                              max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                              bootstrap=True,
                              oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                              )
    ]

    classifier_name_list = [
        'Decision-Tree',
        'Gradient-Boost',
        'Random-Forest'
    ]
    # print(classifier_list)

    for classifier, classifier_name in zip(classifier_list, classifier_name_list):
        classifier.fit(X_train, y_train)
        print_evaluation_metrics(classifier, classifier_name, X_val, y_val)
        print_evaluation_metrics2(classifier, classifier_name, X_train, y_train)
        # fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        # cn = ['setosa', 'versicolor', 'virginica']
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=900)
        # tree.plot_tree(classifier)
        # fig.savefig(classifier_name + '.png')
    return


def kmeans(X_train, y_train, X_val, y_val):
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=0, n_jobs=int(0.8 * n_cores)).fit(X_train)
    c_train = kmeans.predict(X_train)
    c_pred = kmeans.predict(X_val)
    centroids = kmeans.cluster_centers_
    y_val_stats = None
    predicted_values = None
    y_train_stats = None
    labels_stats = None
    for i in range(n_clusters):
        print('--------analyzing cluster %d--------' % i)
        train_mask = c_train == i
        std_train = np.std(y_train[train_mask])
        mean_train = np.mean(y_train[train_mask])
        print("# examples & price mean & std for training set within cluster %d is:(%d, %.2f, %.2f)" % (
            i, train_mask.sum(), np.float(mean_train), np.float(std_train)))
        pred_mask = c_pred == i
        std_pred = np.std(y_val[pred_mask])
        mean_pred = np.mean(y_val[pred_mask])
        print("# examples & price mean & std for validation set within cluster %d is:(%d, %.2f, %.2f)" % (
            i, pred_mask.sum(), np.float(mean_pred), np.float(std_pred)))
        if pred_mask.sum() == 0:
            print('Zero membered test set! Skipping the test and training validation.')
            continue
        # LinearModelRidge(X_train[train_mask], y_train[train_mask], X_val[pred_mask], y_val[pred_mask])
        regr = Ridge(alpha=7)  # 7
        regr.fit(X_train[train_mask], y_train[train_mask])
        labels_pred = regr.predict(X_train[train_mask].values)
        y_pred = regr.predict(X_val[pred_mask].values)
        if (y_val_stats is None):
            y_val_stats = copy.deepcopy(y_val[pred_mask])
            y_train_stats = copy.deepcopy(y_train[train_mask])
            predicted_values = copy.deepcopy(y_pred)
            labels_stats = copy.deepcopy(labels_pred)

        else:
            y_val_stats = y_val_stats.append(y_val[pred_mask])
            y_train_stats = y_train_stats.append(y_train[train_mask])
            predicted_values = np.append(predicted_values, y_pred)
            labels_stats = np.append(labels_stats, labels_pred)
        print('--------Finished analyzing cluster %d--------' % i)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_val_stats, predicted_values))
    # print("Mean absolute percente error: ",
    #       np.mean(np.abs((y_val_stats - predicted_values) / y_val_stats)) * 100)
    print("Median absolute error: ",
          metrics.median_absolute_error(y_val_stats, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_val_stats, predicted_values))
    print("R2: ", metrics.r2_score(y_val_stats, predicted_values))
    print('------------TRAIN--------------------')
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_train_stats, labels_stats))
    # print("Mean absolute percente error: ",
    #       np.mean(np.abs((y_val_stats - predicted_values) / y_val_stats)) * 100)
    print("Median absolute error: ",
          metrics.median_absolute_error(y_train_stats, labels_stats))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_train_stats, labels_stats))
    print("R2: ", metrics.r2_score(y_train_stats, labels_stats))

    return c_pred, centroids


def linear_model_SGD(X_train, y_train, X_val, y_val):
    model = SGDRegressor()
    model.fit(X_train, y_train)
    print_evaluation_metrics(model, "sgd", X_val, y_val.values.ravel())
    print_evaluation_metrics2(model, "sgd", X_train, y_train.values.ravel())


if __name__ == "__main__":

    X_train = pd.read_csv('../data/train_X.csv')
    y_train = pd.read_csv('../data/train_y.csv')

    # X_val = pd.read_csv('../data/val_X.csv')
    # y_val = pd.read_csv('../data/val_y.csv')

    X_test = pd.read_csv('../data/test_X.csv')
    y_test = pd.read_csv('../data/test_y.csv')

    # coeffs = np.load('../Data/selected_coefs_pvals.npy')
    coeffs = np.load('../data/selected_feature.npy')
    col_set = set()

    # cherry_picked_list = [
    #     'host_identity_verified',
    #     'latitude',
    #     'longitude',
    #     'accommodates',
    #     'bathrooms',
    #     'bedrooms',
    #     'beds',
    #     'guests_included',
    #     'security_deposit',
    #     'cleaning_fee',
    #     'extra_people',
    #     'number_of_reviews',
    #     'review_scores_rating',
    #     'review_scores_accuracy',
    #     'review_scores_cleanliness',
    #     'review_scores_location',
    #     'review_scores_value',
    #     'reviews_per_month',
    #     'comments'
    #
    # ]
    for i in range(len(coeffs)):
        if (coeffs[i]):
            col_set.add(X_train.columns[i])
    col_set.remove('Unnamed: 0.1')
    # print(list(col_set))
    X_train = X_train[list(col_set)]
    for i in range(len(list(col_set))):
        print(i, X_train.columns[i])
    # X_val = X_val[list(col_set)]
    X_test = X_test[list(col_set)]
    # print(X_train.columns)
    # print(X_train.columns[45])
    # print(X_train.columns[46])
    # print(X_train.columns[29])

    # X_concat = pd.concat([X_train, X_val], ignore_index=True)
    # y_concat = pd.concat([y_train, y_val], ignore_index=True)

    # RUN WITHOUT FEATURE SELECTION FOR THE BASELINE
    # """
    # print("--------------------Linear Regression--------------------")
    # LinearModel(X_train, y_train, X_test, y_test)
    # """

    # print("--------------------Tree-based Model--------------------")
    # TreebasedModel(X_train, y_train, X_test, y_test)
    print("--------------------KMeans Clustering--------------------")
    c_pred, centroids = kmeans(X_train, y_train, X_test, y_test)

    # print("--------------------------------------------------")
    # LinearModelRidge(X_train, y_train, X_test, y_test)
    # print("--------------------------------------------------")
    #
    # print("--------------------------------------------------")
    # LinearModelLasso(X_train, y_train, X_test, y_test)
    # print("--------------------------------------------------")
    # simple_neural_network(X_train, y_train, X_test, y_test)

    # svm(X_train, y_train, X_test, y_test)
