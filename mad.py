# Author: Arjun S Kulathuvayal. Intellectual property. Copyright strictly restricted
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca(df):
    df_normalized = (df - df.mean()) / df.std()
    pca = PCA(n_components=df.shape[1])
    pca.fit(df_normalized)

    loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
                            index=df.columns)
    print(loadings)

    plt.plot(pca.explained_variance_ratio_)
    plt.ylabel('Variance')
    plt.xlabel('Components')
    plt.show()


def mad(df):
    # df = pd.read_csv('vehicles.csv', usecols=[i for i in range(18)])
    mad = df.mad()
    print("Mean absolute deviation")
    print("------------------------")
    print(mad)
    print("Best features are those that have the smallest MAD values.")
    print("------------------------------------------------------------")
    x = dict(mad)
    # print(x)
    best_features = []
    for key, value in x.items():
        if value < 7:
            best_features.append(key)
    print(best_features)
    #
    print("Correlation matrix of dataframe")
    print("-------------------------------")
    corrM = df.corr()
    print(corrM)
    cor_matrix = df.corr().abs()
    upper_tri = corrM.where(np.triu(np.ones(corrM.shape), k=1).astype(np.bool))
    print("Upper triangular matrix")
    print("------------------------")
    print(upper_tri)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print("Column to be droped")
    print("------------------------")
    print(to_drop)

def pcaPer(df):

    df_normalized = (df - df.mean()) / df.std()
    pca = PCA(n_components=df.shape[1])
    pca.fit(df_normalized)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
                            index=df.columns)
    fig, ax = plt.subplots()
    x = np.arange(1, 6, 1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(x, y, '-o')
    plt.xlabel('Number of components')
    plt.ylabel('cumulative variance')
    plt.title('Number of components needed to describe variance')
    plt.axhline(y=0.90, color='r', linestyle='--')
    plt.text(0.5, 0.85, '90% cut-off threshold ~ 5 components', color='red', fontsize='12')
    plt.text(8, 0.95, '99% cut-off threshold ~ 10 components', color='green', fontsize='12')
    plt.axhline(y=0.99, color='g', linestyle='--')
    ax.grid(axis='x')
    plt.show()
    print("Number of componets required becomes exponentially higher for higher cumulative variance")

def pcadis(df):


        def myplot(score,coeff,labels=None):
            y = df['a'].copy()
            X = df.drop(columns=['a']).copy().values
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            pca = PCA()
            x_new = pca.fit_transform(X)

            xs = score[:,0]
            ys = score[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            plt.scatter(xs * scalex,ys * scaley, c = y)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()
        #I'am using only the 2 PCs here (also scalled the data points - above )
        myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('icsd_data_formula.csv', usecols=[3, 4, 5, 6, 7, 8])
    df = df.dropna(how='any', subset=['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
    df = df.drop(df[df.c > 55].index)
    df = df.drop(df[df.b > 28].index)
    df = df.drop(df[df.a > 28].index)
    df = df.drop(df[df.alpha < 60].index)
    df = df.drop(df[df.beta < 70].index)
    #mad(df)
    #pca(df)
    #pcaPer(df)
    pcsdis(df)