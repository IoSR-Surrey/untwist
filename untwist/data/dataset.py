"""
Dataset management utiliies.

A Dataset is assumed to consist of two main objects: a matrix X of training
examples and (optionally) a matrix Y of ground truth labels. In both cases,
rows are observations and columns are features.

For audio source separation in time-frequency, the columns of X will typically
be frequency bins. The matrix Y can be a hard or soft mask, or for other tasks
it may be just a class label.
"""
from __future__ import division, print_function
import abc
import os.path
import h5py
import numpy as np
from ..base import types
from ..utilities import stats


class DatasetBase:
    _metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def num_observations(self):
        return NotImplemented

    def num_batches(self, size):
        return self.num_observations // size

    @abc.abstractmethod
    def add(self, x, y=np.empty((0, 0))):
        return NotImplemented

    @abc.abstractmethod
    def get_batch(self, index, size):
        return NotImplemented

    @abc.abstractmethod
    def save(self, basename):
        return NotImplemented

    @abc.abstractmethod
    def load(self, basename):
        return NotImplemented

    # inspired by librosa, but here data points are rows
    def shingle(self, X, steps, delay=1):
        X = np.pad(X, [(int(steps - 1) * delay, 0), (0, 0)],
                   mode="constant", constant_values=[0])
        Xs = X
        for i in range(1, steps):
            Xs = np.hstack([Xs, np.roll(X, -i * delay, axis=0)])
        return Xs

    # average repeated positions, ignore absolute 0
    def unshingle(self, X, steps, delay=1):
        n_features = X.shape[1] // steps
        Xu = X[:, :n_features]
        for i in range(1, steps):
            Xtmp = X[:, i*n_features:(i+1)*n_features]
            Xu = np.dstack([Xu, np.roll(Xtmp, i * delay, axis=0)])
        if steps > 1:
            # Xu[Xu==0]=np.nan
            return np.mean(Xu, 2)[steps-1:, :]
        else:
            return Xu


class Dataset(DatasetBase):
    """
    In-memory Dataset
    """

    def __init__(self, x_width, x_type, y_width=0, y_type=types.int_):
        self.X = np.empty((0, x_width), x_type)
        self.Y = np.empty((0, y_width), y_type)

    @property
    def num_observations(self):
        return self.X.shape[0]

    def num_batches(self, size):
        return self.num_observations // size

    def add(self, x, y=np.empty((0, 0))):
        self.X = np.append(self.X, x, 0)
        self.Y = np.append(self.Y, y, 0)

    def shuffle(self):
        perm = np.random.permutation(self.X.shape[0])
        self.X = self.X[perm, :]
        self.Y = self.Y[perm, :]

    def standardize(self):
        self.X = self.standardize_points(self.X)

    def normalize(self):
        self.X = (self.X - np.amin(self.X, 0)) \
            / (np.amax(self.X, 0) - np.amin(self.X, 0))

    def get_batch(self, index, size):
        x = self.X[index * size:(index + 1) * size, :]
        y = self.Y[index * size:(index + 1) * size, :]
        return (x, y)

    def batcher(self,
                start=0,
                end=None,
                batch_size=100):
        '''
        A generator for batches.
        If batch_size > number of observations, batch_size is limited and 1
        batch is returned.
        '''
        if batch_size > self.num_observations:
            batch_size = self.num_observations

        if end is None or end <= start:
            end = self.num_observations // batch_size

        for i in range(start, end):
            yield self.get_batch(i, batch_size)

    def save(self, path):
        np.save(path + "/X.npy", self.X)
        np.save(path + "/Y.npy", self.Y)

    def load(self, path):
        self.X = np.load(path + "/X.npy")
        self.Y = np.load(path + "/Y.npy")

    def normalize_points(self, x):
        return np.divide(
            x - np.amin(self.X, 0),
            np.amax(self.X, 0) - np.amin(self.X, 0),
            np.empty_like(x))

    def standardize_points(self, x):
        return np.divide(x - np.mean(self.X, 0),
                         np.std(self.X, 0),  np.empty_like(x))


class HDF5Dataset(DatasetBase):

    def __init__(self,
                 path,
                 x_shape=None,
                 x_dtype=np.float,
                 y_shape=None,
                 y_dtype=np.float,
                 write_shuffle=False,
                 overwrite=True,
                 input_key='X',
                 output_key='Y'):

        self.path = "{}.hdf5".format(os.path.splitext(path)[0])

        if not os.path.isfile(self.path):
            overwrite = True

        self.idx = np.arange(x_shape[0])
        self.input_key = input_key
        self.output_key = output_key
        self.normaliser = None
        self.running_stats = None

        if overwrite:
            if x_shape[0] != y_shape[0]:
                raise ValueError("X and Y must have same number of rows")

            with h5py.File(self.path, "w") as f:

                f.create_dataset(input_key, x_shape, dtype=x_dtype)
                f.create_dataset(output_key, y_shape, dtype=y_dtype)

                write_indices = np.arange(x_shape[0])
                if write_shuffle:
                    np.random.shuffle(write_indices)
                f.create_dataset('write_indices', data=write_indices)

                f.attrs['row'] = 0
                f.attrs['input_key'] = input_key
                f.attrs['output_keys'] = np.array(
                    [np.string_(self.output_key)])

        with h5py.File(self.path, "r") as f:

            self._num_observations = f[input_key].shape[0]
            self.output_keys = [_.decode() for _ in f.attrs['output_keys']]

            if self.running_stats is None:
                width, dtype = f[input_key].shape[1], types.float_
                self.running_stats = stats.RunningStats(width, dtype)
            else:  # Load from disk
                group, tmp_stats = f['running_stats'], {}
                for key, ary in group.items():
                    tmp_stats[key] = ary[:]
                self.running_stats.stats = tmp_stats
                self.running_stats.n = f.attrs['n']

    @classmethod
    def load(cls, path):
        return cls(path, overwrite=False)

    @property
    def num_observations(self):
        return self._num_observations

    def _save_running_stats(self, group):

        for key, ary in self.running_stats.stats.items():
            ds = group.require_dataset(key,
                                       shape=ary.shape,
                                       dtype=ary.dtype)
            ds[:] = ary

        group.attrs['n'] = self.running_stats.n

    def get_data(self, name='X'):
        with h5py.File(self.path, "r") as f:
            return f[name][:]

    def create_data(self, name, shape, dtype=np.float):
        with h5py.File(self.path, "a") as f:
            f.create_dataset(name, shape, dtype=dtype)
            f.attrs['output_keys'] = np.append(
                f.attrs['output_keys'], np.string_(name))
            self.output_keys = [_.decode() for _ in f.attrs['output_keys']]

    def add(self, x, y, names=None):
        '''
        If y is a list, names must be a list with the corresponding h5df
        dataset names.
        '''

        with h5py.File(self.path, "a") as f:

            if not isinstance(names, list):
                names = self.output_keys
            r = f.attrs['row']
            indices = f['write_indices'][:]
            idx = indices[r:r + x.shape[0]]
            idx2 = idx.argsort()  # Since h5py only write in ascending order
            idx = idx[idx2]
            xIn = x[idx2, :]
            f[self.input_key][idx, :] = xIn

            self.running_stats.update(xIn)
            group = f.require_group('running_stats')

            self._save_running_stats(group)

            if isinstance(y, list):
                for name, ytmp in zip(names, y):
                    f[name][idx, :] = ytmp[idx2, :]
            else:
                f[self.output_key][idx, :] = y[idx2, :]

    def shuffle(self):
        np.random.shuffle(self.idx)

    @property
    def stats(self):
        return self.running_stats.stats

    def set_normaliser(self, norm_method=2):
        if norm_method == 1:
            def normaliser(x):
                return stats.rangeNormalise(
                    x,
                    self.stats['min'][:],
                    self.stats['max'][:],
                    axis=0)
        else:
            def normaliser(x):
                return stats.standardise(
                    x,
                    self.stats['mean'][:],
                    np.sqrt(self.stats['var'][:]),
                    ddof=1,
                    axis=0)
        self.normaliser = normaliser

    def batcher(self,
                start=0,
                end=None,
                batch_size=100):
        '''
        A generator for batches.
        If batch_size > number of observations, batch_size is limited and 1
        batch is returned.
        '''
        if not self.normaliser:
            self.set_normaliser(2)

        if batch_size > self.num_observations:
            batch_size = self.num_observations

        if end is None or end <= start:
            end = self.num_observations // batch_size

        with h5py.File(self.path, "r") as f:
            for i in range(start, end):
                indices = self.idx[i*batch_size: (i+1)*batch_size]
                x, y = f[self.input_key], f[self.output_key]
                x_batch = self.normaliser(x[indices, :])
                y_batch = y[indices, :]
                yield (x_batch, y_batch)
