import numpy as np
from numpy import linalg
from ..base.algorithms import Processor

class RPCA(Processor):
    """
    Robust PCA, Inexact ALM method
    Based on http://perception.csl.illinois.edu/matrix-rank/sample_code.html
    """
    
    def __init__(self, iterations = 100, threshold = None, l = 1, mu = 1.25, rho = 1.5):
        self.iterations = iterations
        self.threshold = threshold
        self.l = l
        self.mu = mu
        self.rho = rho
        
    def process(self, X):
        Y = X
        n = Y.shape[0]
        m = Y.shape[1]
        self.l = self.l / np.sqrt(m)
        norm_two = linalg.norm(Y.ravel(), 2)
        norm_inf = linalg.norm(Y.ravel(), np.inf) / self.l
        dual_norm = np.max([norm_two, norm_inf])
        Y = Y / dual_norm
        A = np.zeros(Y.shape)
        E = np.zeros(Y.shape)
        d_norm = linalg.norm(X, 'fro')
        mu = self.mu / norm_two
        mu_bar = mu * 1e7
        sv = 10.0
        

        for i in range(self.iterations):
            temp_T = X - A + (1 / mu) * Y
            E = np.maximum(temp_T - self.l / mu, 0) 
            E += np.minimum(temp_T + self.l / mu, 0)
            U, S, V = linalg.svd(X - E + (1 / mu) * Y, full_matrices = False)
            svp = (S > 1 / mu).shape[0]
            if svp < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(0.05 * n), n])
            A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
            Z = X - A - E
            Y = Y + mu * Z
            mu = np.min([mu * self.rho, mu_bar])
            err = linalg.norm(Z, 'fro') / d_norm
            print i, err
            if self.threshold is not None and err < self.threshold:
                break 
        return A, E