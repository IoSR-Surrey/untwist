from __future__ import print_function
from ...neuralnetworks import MLP, SGD
from ...data import Dataset
import numpy as np
from numpy.random import multivariate_normal as normal


def test_mlp():
    np.random.seed(0)
    mean1 = [0, 0]
    mean2 = [20, 20]
    cov = [[2, 0], [0, 2]]    
    x1 = normal(mean1, cov, 2000)
    x2 = normal(mean2, cov, 2000)
    y1 = np.ones((1000,1))
    y2 = np.zeros((1000,1))
    ds = Dataset(2, float, 1, int)
    ds.add(x1[:1000,:],y1)
    ds.add(x2[:1000,:],y2)
    mlp = MLP(2, 1, [2])
    sgd = SGD(mlp, rate_decay_th=0,iterations = 1000)
    sgd.train(ds)
    y = sgd.predict(x1[1000:,:])
    print(np.sum(y > 0.5) , y.shape[0])
    assert(np.sum(y > 0.5) == y.shape[0])
    y = sgd.predict(x2[1000:,:])
    print(np.sum(y < 0.5), y.shape[0])
    assert(np.sum(y < 0.5) == y.shape[0])
