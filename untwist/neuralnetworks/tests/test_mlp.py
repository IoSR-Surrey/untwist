from ...neuralnetworks import MLP, SGD
from ...data import Dataset
import numpy as np
from numpy.random import multivariate_normal as normal


def test_mlp():
    mean1 = [0, 0]
    mean2 = [10, 10]
    cov = [[2, 0], [0, 2]] 
    dist = np.random.multivariate_normal
    x1 = normal(mean1, cov, 2000)
    x2 = normal(mean2, cov, 2000)
    y1 = np.ones((2000,1))
    y2 = np.zeros((2000,1))
    ds = Dataset(2, float, 1, int)
    ds.add(x1[:1000,:],y1[:1000])
    ds.add(x2[:1000,:],y2[:1000])
    mlp = MLP(2, 1, [2])
    sgd = SGD(mlp, iterations = 1000)
    sgd.train(ds)
    y = sgd.predict(x1[1000:,:])
    assert(np.sum(y > 0.5) == y.shape[0])
    y = sgd.predict(x2[1000:,:])
    assert(np.sum(y < 0.5) == y.shape[0])
