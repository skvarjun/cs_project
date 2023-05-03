# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
from matminer.data_retrieval import retrieve_MDF
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import bz2
import pandas as pd
import pickle as pkl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold


def magpieFeaturizer():
    data = pd.read_csv('icsd_data_formula.csv', usecols=['formula', 'a'], header=0)
    data = data[~data['formula'].str.contains("D2O")]
    data = data[~data['formula'].str.contains("D")]
    data = data[~data['formula'].str.contains("\\?")]
    data = data[~data['formula'].str.contains("None")]
    data = data[~data['formula'].str.contains("_")]

    # df["formula"] = df['formula'].str.replace('Li','H')
    # df = df[~df['formula'].str.contains("O3")]

    data.columns = ["formula", "a"]

    # Training data processing , ignore_errors=True
    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'formula')

    # Remove compound which does not have reported value of mean_cov_radius
    for k in ['a']:
        data[k] = pd.to_numeric(data[k], errors='coerce')
    original_count = len(data)
    data = data[~ data['a'].isnull()]
    print('Removed %d out of %d entries where a is Nan' % (original_count - len(data), original_count))

    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    # Get feature names
    feature_labels = feature_calculators.feature_labels()

    # Compute features
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj')
    print("Featurization completed")
    print("--------Feature names--------")
    print(feature_labels)
    print('Generated %d features' % len(feature_labels))
    print('Training set size:', 'x'.join([str(x) for x in data[feature_labels].shape]))
    data.to_csv('main_features_dataset.csv')

    # Remove entries with NaN or infinite features
    original_count = len(data)
    data = data[~ data[feature_labels].isnull().any(axis=1)]
    print('Removed %d/%d entries with NaN or infinite features' % (original_count - len(data), original_count))

    # Random forest
    Low_com_effort = False
    model = GridSearchCV(RandomForestRegressor(n_estimators=20 if Low_com_effort else 150, n_jobs=-1),
                         param_grid=dict(max_features=range(8, 15)),
                         scoring='neg_mean_squared_error', cv=ShuffleSplit(n_splits=1, test_size=0.1))

    model.fit(data[feature_labels], data['a'])
    print("Model's best score".format(model.best_score_))

    # Plot the score as a function of alpha
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(model.cv_results_['param_max_features'].data, np.sqrt(-1 * model.cv_results_['mean_test_score']),
                marker='o', color='r', label='Mean test score')
    ax1.scatter([model.best_params_['max_features']], np.sqrt([-1 * model.best_score_]), marker='o', color='g', s=40,
                label='Best score')
    ax1.set_xlabel('Max. Features')
    ax1.set_ylabel('RMSE (Angstrom)')
    ax1.legend(loc="upper right")

    model = model.best_estimator_

    # Cross-validation Test
    cv_prediction = cross_val_predict(model, data[feature_labels], data['a'], cv=KFold(10, shuffle=True))
    # Compute aggregate statistics
    for scorer in ['r2_score', 'mean_absolute_error', 'mean_squared_error']:
        score = getattr(metrics, scorer)(data['a'], cv_prediction)
        print(scorer, score)

    with bz2.BZ2File('model_predict_a.pbz2', 'wb') as f:
        pkl.dump(model, f)

    # Plot the individual  predictions
    ax2.hist2d(pd.to_numeric(data['a']), cv_prediction, norm=LogNorm(), bins=64, cmap='Blues', alpha=0.9)
    ax2.set_xlim(ax2.get_ylim())
    ax2.set_ylim(ax2.get_xlim())

    mae = metrics.mean_absolute_error(data['a'], cv_prediction)
    r2 = metrics.r2_score(data['a'], cv_prediction)
    ax2.text(0.5, 0.1, 'MAE: {:.2f} Angstrom\n$R^2$:  {:.2f}'.format(mae, r2), transform=ax2.transAxes,
             bbox={'facecolor': 'w', 'edgecolor': 'k'})
    ax2.plot(ax2.get_xlim(), ax2.get_xlim(), 'k--')
    ax2.set_xlabel('Reported mean lattice constant a (Angstrom)')
    ax2.set_ylabel('ML predicted lattice constant a (Angstrom)')
    plt.tight_layout()
    plt.grid()
    fig.savefig('ml_outlook.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    magpieFeaturizer()
