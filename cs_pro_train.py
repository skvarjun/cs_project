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

def RanFor(df):
    X, y = df.drop(['a'], axis=1), df['a']
    model = GridSearchCV(RandomForestRegressor(n_estimators=50, n_jobs=-1, criterion='squared_error', max_depth=100, min_samples_split=2, min_samples_leaf=1),
                         param_grid=dict(max_features=range(1, 6)), scoring='neg_mean_squared_error', cv=KFold(5, shuffle=True))
    model.fit(X, y)

    with bz2.BZ2File('best_classifier.pbz2', 'wb') as f:
        pkl.dump(model, f)


def DTtune(df):
    pipeline = Pipeline([('regressor', DecisionTreeRegressor())])
    hyperparameters = {
        'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'regressor__splitter': ['best', 'random'],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [5, 10, 15]
    }

    X, y = df.drop(['a'], axis=1), df['a']
    model = GridSearchCV(pipeline, hyperparameters, cv=KFold(5, shuffle=True), scoring='neg_mean_squared_error')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    print("For train size of {}, the best hyperparameters are {}".format(0.2, model.best_params_))
    model = model.best_estimator_
    with bz2.BZ2File('best_DTR.pbz2', 'wb') as f:
        pkl.dump(model, f)
    print("Model has been built and saved as best_DTR.pbz2")

def RFtune(df):
    pipeline = Pipeline([('regressor', RandomForestRegressor())])
    hyperparameters = {
        'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'regressor__max_depth': [3, 5, 7],
        'regressor__n_estimators': [25, 50, 75],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [5, 10, 15],
        'regressor__max_features': range(1, 6)
    }

    X, y = df.drop(['a'], axis=1), df['a']
    model = GridSearchCV(pipeline, hyperparameters, cv=KFold(5, shuffle=True), scoring='neg_mean_squared_error')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print("For train size of {}, the best hyperparameters are {}".format(0.2, model.best_params_))
    model = model.best_estimator_
    with bz2.BZ2File('best_DTR.pbz2', 'wb') as f:
        pkl.dump(model, f)
    print("Model has been built and saved as best_DTR.pbz2")

if __name__ == '__main__':
    df = pd.read_csv('icsd_data_formula.csv', usecols=[3, 4, 5, 6, 7, 8])
    df = df.dropna(how='any', subset=['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
    df = df.drop(df[df.c > 55].index)
    df = df.drop(df[df.b > 28].index)
    df = df.drop(df[df.a > 28].index)
    df = df.drop(df[df.alpha < 60].index)
    df = df.drop(df[df.beta < 70].index)

    print("----->     Random Forest Classifier  <-----")
    print("Model has been built and saved as best_classifier.pbz2")
    #RanFor(df)
    #train_and_pred(df)
    #DTtune(df)
    RFtune(df)