import numpy as np


class PCA:
    def __init__(self, n_components=1):
        self.mean = None
        self.eig_vectors = None
        self.n_components = n_components

    def build(self, x):
        m = np.mean(x, axis=0)
        xm = x - m
        cov_mat = np.cov(xm.T)
        eig_values, eig_vectors = np.linalg.eig(cov_mat)

        idx = np.argsort(eig_values)[::-1]
        eig_vectors = eig_vectors[:, idx]
        v = eig_vectors[:, :self.n_components]
        projection = xm.dot(v)

        self.eig_vectors = eig_vectors
        self.mean = m
        return projection

    def project(self, x):
        xm = x - self.mean
        v = self.eig_vectors[:, :self.n_components]
        return xm.dot(v)

    def iproject(self, z):
        v = self.eig_vectors[:, :self.n_components]
        x = z * v.T + self.mean
        return x
