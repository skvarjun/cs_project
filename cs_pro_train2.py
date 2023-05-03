# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold
import bz2
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def RFtrain_and_pred(df):
    X, y = df.drop(['a'], axis=1), df['a']
    model = GridSearchCV(
        RandomForestRegressor(n_estimators=50, n_jobs=-1, criterion='squared_error', max_depth=100, min_samples_split=2,
                              min_samples_leaf=1),
        param_grid=dict(max_features=range(1, 6)), scoring='neg_mean_squared_error', cv=KFold(5, shuffle=True))
    rmse_plot = []
    for n in np.arange(0.1, 1, 0.1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, random_state=42)
        model.fit(X_train, y_train)
        pred_Adj_Close = model.predict(X_test)
        mse = mean_squared_error(y_test, pred_Adj_Close)
        rmse = np.sqrt(mse)
        rmse_plot.append(rmse)

    plt.plot(np.arange(0.1, 1, 0.1), rmse_plot, '-o', c='g')
    plt.xlabel('Split size in fraction')
    plt.ylabel('RMSE')
    plt.title('Performance on the training and test sets')
    plt.show()

def DTtrain_and_pred(df):
    X, y = df.drop(['a'], axis=1), df['a']
    model = GridSearchCV(DecisionTreeRegressor(criterion='poisson', splitter='best', max_depth=7, min_samples_split=10, min_samples_leaf=5), param_grid=dict(max_features=range(1, 6)), scoring='neg_mean_squared_error', cv=KFold(5, shuffle=True))
    rmse_plot = []
    for n in np.arange(0.1, 1, 0.1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, random_state=42)
        model.fit(X_train, y_train)
        pred_Adj_Close = model.predict(X_test)
        mse = mean_squared_error(y_test, pred_Adj_Close)
        rmse = np.sqrt(mse)
        rmse_plot.append(rmse)

    plt.plot(np.arange(0.1, 1, 0.1), rmse_plot, '-o', c='g')
    plt.xlabel('Split size in fraction')
    plt.ylabel('RMSE')
    plt.title('Performance on the training and test sets')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('main_features_dataset.csv', header=0)
    df = df.drop(['formula', 'composition_obj'], axis=1)
    df = df.dropna(how='any', subset=['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
    df = df.drop(df[df.c > 55].index)
    df = df.drop(df[df.b > 28].index)
    df = df.drop(df[df.a > 28].index)
    df = df.drop(df[df.alpha < 60].index)
    df = df.drop(df[df.beta < 70].index)
    DTtrain_and_pred(df)