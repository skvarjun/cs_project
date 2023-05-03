# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.composition import Composition
import numpy as np
import pandas as pd
import bz2
import _pickle as cPickle
import argparse


feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])


def generate(fake_df, ignore_errors=False):
    fake_df = np.array([fake_df])
    fake_df = pd.DataFrame(fake_df)
    fake_df.columns = ['full_formula']
    fake_df = StrToComposition().featurize_dataframe(fake_df, "full_formula", ignore_errors=ignore_errors)
    fake_df = fake_df.dropna()
    fake_df = feature_calculators.featurize_dataframe(fake_df, col_id='composition', ignore_errors=ignore_errors);
    fake_df["NComp"] = fake_df["composition"].apply(len)
    return fake_df


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def main():
    print('----------Predicting----------')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--formula', type=str, help="The input crystal formula.")
    args = parser.parse_args()
    form = args.formula
    print("Formula given: {}".format(form))

    ext_magpie = generate(form)
    print(ext_magpie)
    result = ext_magpie.drop(['NComp', 'composition', 'full_formula'], axis=1)




    a = decompress_pickle('model_predict_a.pbz2')
    a = a.predict(result)

    print('prediction=', a[0])
    print('-----------Complete-----------')


if __name__ == "__main__":
    main()
