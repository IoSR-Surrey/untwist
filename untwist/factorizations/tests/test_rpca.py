import numpy as np
from ...factorizations import RPCA


def test_rpca():
    A = np.ones((10, 40))
    A[:, 2*np.arange(20)] = 0
    B = np.zeros((10, 40))
    B[2, :] = 10
    C = A + B
    rpca = RPCA(300)
    [L, S] = rpca.process(C)
    assert np.sum(np.abs(L - A)) < 100
    assert np.sum(np.abs(S - B)) < 100
    assert np.sum(np.abs((L + S) - C)) < 0.01
