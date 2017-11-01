from __future__ import division, print_function
import numpy as np
from numpy.linalg import svd, norm
from scipy.sparse.linalg import svds
from ..base import algorithms
from .. import data


class RPCA(algorithms.Processor):
    """
    Robust PCA, Inexact ALM method
    Based on http://perception.csl.illinois.edu/matrix-rank/sample_code.html
    """

    def choosvd(self, n, d):
        y = False
        if n   <= 100: y = d / n <= 0.02
        elif n <= 200: y = d/n <= 0.06
        elif n <= 300: y = d/n <=0.26
        elif n <= 400: y = d/n <=0.28
        elif n <= 500: y = d/n <=0.34
        else: y = d/n <=0.38
        return y



    def __init__(self, iterations = 100, threshold = None, l = 1, mu = 1.25, rho = 1.5):
        self.iterations = iterations
        self.threshold = threshold
        self.l = l
        self.mu = mu
        self.rho = rho

    def process(self, X):
        X = X.T
        Y = X.copy()
        (m, n) = Y.shape
        n = Y.shape[1]
        self.l = self.l / np.sqrt(m)
        u,s,v = svds(Y,1,which="LM")
        norm_two = s[0]
        norm_inf = norm(Y.ravel(), np.inf) / self.l
        dual_norm = np.max([norm_two, norm_inf])
        Y = Y / dual_norm
        A = np.zeros(Y.shape)
        E = np.zeros(Y.shape)
        d_norm = norm(X, 'fro')
        mu = self.mu / norm_two
        mu_bar = mu * 1e7
        sv = 10

        for i in range(self.iterations):
            temp_T = X - A + (1 / mu) * Y
            E = np.maximum(temp_T - self.l / mu, 0)
            E += np.minimum(temp_T + self.l / mu, 0)
            sparse_svd = self.choosvd( n, sv)
            if sparse_svd:
                U, S, V = svds(X - E + (1 / mu) * Y, sv, which= "LM")
            else:
                U, S, V = svd(X - E + (1 / mu) * Y, full_matrices = False)
            svp = len(np.where(S > (1 / mu))[0])
            if svp < sv:
                sv = int(np.min([svp+1, n]))
            else:
                sv = int(np.min([svp + np.round(0.05 * n), n]))
            if sparse_svd:
                A = np.dot(np.dot(U[:,-svp:], np.diag(S[-svp:] -1/mu)), V[-svp:,:])
            else:
                A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
            Z = X - A - E
            Y = Y + mu * Z
            mu = np.min([mu * self.rho, mu_bar])
            err = norm(Z, 'fro') / d_norm
            print(i, err)
            if self.threshold is not None and err < self.threshold:
                break
        a, e = X.T.copy(), X.T.copy()
        a[:], e[:] = A.T, E.T
        return a, e
