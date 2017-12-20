from __future__ import division, print_function
import numpy as np


class RunningStats:
    '''
    Based on Welford's Method for variance,
    but not mean (due to vectorisation).
    '''

    def __init__(self, width, dtype):
        self.width = width
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.n = 1
        names = ['mean', 'var', 'm2', 'min', 'max']
        self.stats = {}
        for stat in names:
            self.stats[stat] = np.zeros(self.width, dtype=self.dtype)

    def update(self, x):

        if x.ndim == 1:
            x.shape = (1, -1)

        # Mean and variance
        mn = np.arange(self.n, self.n + x.shape[0])[:, np.newaxis]
        current_means = ((self.stats['mean'] * (mn[0]-1) / mn) +
                         np.cumsum(x, 0) / mn)
        current_means = np.vstack((self.stats['mean'], current_means))

        delta = x - current_means[1:]
        delta2 = x - current_means[:-1]
        m2_temp = delta * delta2

        self.stats['m2'] += np.sum(m2_temp, 0)
        self.stats['mean'] = current_means[-1]

        if mn[-1] == 1:
            self.stats['var'] = self.stats['m2']
        else:
            self.stats['var'] = self.stats['m2'] / (mn[-1] - 1)

        # Min and max
        if self.n == 1:
            self.stats['min'] = np.min(x, 0)
            self.stats['max'] = np.max(x, 0)

        self.stats['min'] = np.min(
            np.r_[x, self.stats['min'].reshape(1, -1)], 0)
        self.stats['max'] = np.max(
            np.r_[x, self.stats['max'].reshape(1, -1)], 0)

        self.n += x.shape[0]


def range_normalize(x, min=None, max=None, axis=0):
    if min is None or max is None:
        min, max = np.min(x, axis), np.max(x, axis)
    if min.ndim < x.ndim:
        min = np.expand_dims(min, axis=axis)
        max = np.expand_dims(max, axis=axis)
    return (x - min) / (max - min)


def standardise(x, mu=None, std=None, ddof=1, axis=0):
    if mu is None or std is None:
        mu, std = np.mean(x, axis=axis), np.std(x, ddof=ddof, axis=axis)

    if mu.ndim < x.ndim:
        mu = np.expand_dims(mu, axis=axis)
        std = np.expand_dims(std, axis=axis)
    return (x - mu) / std
