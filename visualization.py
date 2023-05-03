# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from figrecipes import PlotlyFig
import numpy as np
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt

def ParaCo():
    df = pd.read_csv('icsd_data_formula.csv', nrows=10000, header=0)
    df = df.drop(df[df.c > 55].index)
    df = df.drop(df[df.b > 28].index)
    df = df.drop(df[df.a > 28].index)
    df = df.drop(df[df.alpha < 60].index)
    df = df.drop(df[df.beta < 70].index)
    #df = df.drop(df[df.gamma < 60].index)
    # df = df[((df["diffusionMode"] == "self") | (df["diffusionMode"] == "impurity") | (df["diffusionMode"] == "inter"))]
    # df["diffusionMode"] = df["diffusionMode"].replace(["self"], 0)
    # df["diffusionMode"] = df["diffusionMode"].replace(["impurity"], 1)
    # df["diffusionMode"] = df["diffusionMode"].replace(["inter"], 2)

    pf = PlotlyFig(df, title="Parallel coordinates | Lattice constant prediction modeling", colorscale='Jet')
    pf.parallel_coordinates(cols=['a', 'b', 'c', 'alpha', 'beta', 'gamma'], colors=np.arange(0, 5000, 1))

    pf = PlotlyFig(df)
    pf.scatter_matrix(cols=['a', 'b', 'c', 'alpha', 'beta', 'gamma'], labels='formula', colorscale='Viridis')


def counter(df):
    count = {}
    for word in df['space_group']:
        if word in count:
            count[word] += 1
        else:
            count[word] = 1
    #sorted(count)
    v, k = [], []
    for key, val in count.items():
        k.append(str(key))
        v.append(val)
    # k = list(count.keys())
    # v = list(count.values())
    print(len(k), len(v))
    plt.xlabel("Mode of diffusion")
    plt.ylabel("Number of data points")
    plt.title('Mode of diffusion and corresponding data points')
    plt.xticks(rotation='vertical')
    plt.bar(k, v, color='b', width=0.8)

    plt.savefig("fig_prob_3_a.png", dpi=300)
    plt.show()
    return count

if __name__ == "__main__":
    df = pd.read_csv('icsd_data_formula.csv', nrows=5000, header=0)
    df = df.drop(df[df.c > 55].index)
    df = df.drop(df[df.b > 28].index)
    df = df.drop(df[df.a > 28].index)
    df = df.drop(df[df.alpha < 60].index)
    df = df.drop(df[df.beta < 70].index)
    #df = df.drop(df[df.gamma < 60].index)
    #ParaCo()
    counter(df)