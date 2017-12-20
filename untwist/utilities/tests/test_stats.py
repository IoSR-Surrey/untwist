from __future__ import division, print_function
import numpy as np
from .. import stats


def test_runningstats():

    def test():
        n = np.random.randint(100, 10000)
        width = n // 4
        ob = stats.RunningStats(width, np.float64)
        x = np.random.normal(size=(n, width))

        size = n // 4
        x = x[:size * 4]
        end = 0
        for i in range(4):
            start = i * size
            end += size
            ob.update(x[start:end])

        mean_check = np.allclose(np.mean(x, 0), ob.stats['mean'])
        var_check = np.allclose(np.var(x, 0, ddof=1), ob.stats['var'])
        min_check = np.allclose(np.min(x, 0), ob.stats['min'])
        max_check = np.allclose(np.max(x, 0), ob.stats['max'])

        assert all([mean_check, var_check, min_check, max_check])

    [test() for i in range(10)]


def test_standardise():
    from scipy.stats import zscore

    x = np.random.normal(size=(10, 10))
    ddof = 1

    for axis in np.arange(x.ndim):
        assert(np.array_equal(
            zscore(x, ddof=ddof, axis=axis),
            stats.standardise(x, ddof=ddof, axis=axis))
        )

    # Arguments
    for axis in np.arange(x.ndim):
        m = np.mean(x, axis=axis)
        sd = np.std(x, ddof=ddof, axis=axis)
        assert(np.array_equal(
            zscore(x, ddof=ddof, axis=axis),
            stats.standardise(x, m, sd, ddof=ddof, axis=axis))
        )


def test_range_normalse():

    x = np.random.normal(size=(10, 10))

    for axis in np.arange(x.ndim):
        y = stats.range_normalize(x, axis=axis)
        assert(np.all(np.min(y, axis=axis) == 0))
        assert(np.all(np.max(y, axis=axis) == 1))

    # Arguments
    for axis in np.arange(x.ndim):
        min = np.min(x, axis=axis)
        max = np.max(x, axis=axis)
        y = stats.range_normalize(x, min, max, axis=axis)
        assert(np.all(np.min(y, axis=axis) == 0))
        assert(np.all(np.max(y, axis=axis) == 1))
