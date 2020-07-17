import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, RidgeCV, ElasticNetCV, lars_path

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from scipy import stats


def show_graph():
    some_features = pickle.load(open("some_features.pickle", "rb"))

    X, y = some_features.drop(columns=['domestic_total_gross']), some_features['domestic_total_gross']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state = 420)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=69)

    std = StandardScaler()
    std.fit(X_train.values)
    X_tr = std.transform(X_train.values)
    X_te = std.transform(X_test.values)

    lasso_model = Lasso(alpha = 1000)
    lasso_model.fit(X_train, y_train)
    list(zip(X_train.columns, lasso_model.coef_))

    alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.figure(figsize=(10, 10))
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.legend(X_train.columns)
    plt.show()
