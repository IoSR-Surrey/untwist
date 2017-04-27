from __future__ import print_function
import numpy as np
from ...factorizations import NMF


def test_nmf():
    X = np.random.random((6, 2))
    nmf = NMF(2, return_divergence=True)
    [W, H, err] = nmf.process(X)
    print(err[-1])
    assert(err[-1] < 1e-3)
