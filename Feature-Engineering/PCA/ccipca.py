"""
https://stackoverflow.com/questions/42020619/incremental-pca
https://github.com/kevinhughes27/pyIPCA
"""

import scipy.sparse as sp
import numpy as np
from scipy import linalg as la
import scipy.sparse as sps
from sklearn import datasets

class CCIPCA:
    def __init__(self, n_components, n_features, amnesic=2.0, copy=True):
        self.n_components = n_components
        self.n_features = n_features
        self.copy = copy
        self.amnesic = amnesic
        self.iteration = 0
        self.mean_ = None
        self.components_ = None
        self.mean_ = np.zeros([self.n_features], np.float64)
        self.components_ = np.ones((self.n_components,self.n_features)) / \
                           (self.n_features*self.n_components)

    def partial_fit(self, u):
        n = float(self.iteration)
        V = self.components_

        # amnesic learning params
        if n <= int(self.amnesic):
            w1 = float(n+2-1)/float(n+2)
            w2 = float(1)/float(n+2)
        else:
            w1 = float(n+2-self.amnesic)/float(n+2)
            w2 = float(1+self.amnesic)/float(n+2)

        # update mean
        self.mean_ = w1*self.mean_ + w2*u

        # mean center u
        u = u - self.mean_

        # update components
        for j in range(0,self.n_components):

            if j > n: pass
            elif j == n: V[j,:] = u
            else:
                # update the components
                V[j,:] = w1*V[j,:] + w2*np.dot(u,V[j,:])*u / la.norm(V[j,:])
                normedV = V[j,:] / la.norm(V[j,:])
                normedV = normedV.reshape((self.n_features, 1))
                u = u - np.dot(np.dot(u,normedV),normedV.T)

        self.iteration += 1
        self.components_ = V / la.norm(V)

        return

    def post_process(self):
        self.explained_variance_ratio_ = np.sqrt(np.sum(self.components_**2, axis=1))
        idx = np.argsort(-self.explained_variance_ratio_)
        self.explained_variance_ratio_ = self.explained_variance_ratio_[idx]
        self.components_ = self.components_[idx, :]
        self.explained_variance_ratio_ = (self.explained_variance_ratio_ / self.explained_variance_ratio_.sum())
        for r in range(0,self.components_.shape[0]):
            d = np.sqrt(np.dot(self.components_[r, :], self.components_[r, :]))
            self.components_[r, :] /= d


if __name__ == '__main__':

    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    print('X.shape =', X.shape)
    print('y.shape =', y.shape)

    """
    df = pd.read_csv('iris.csv')
    df = np.array(df)[:, :4].astype(float)
    """
    df = X

    pca = CCIPCA(n_components=2, n_features=4)
    S = 10
    print(df[0, :])
    for i in range(150):
        pca.partial_fit(df[i, :])
    pca.post_process()

    # ?????