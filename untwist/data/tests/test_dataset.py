from __future__ import print_function
from .. import Dataset, HDF5Dataset
import numpy as np
import os
from ...utilities import stats
from ...utilities import general

'''
TODO: Check _test_normalize
'''


def _test_shingle(ds):
    x = np.random.random((300, ds.X.shape[1]))
    y = ds.shingle(x, 5)
    z = ds.unshingle(y, 5)
    assert np.sum(x-z) < 1e-10


def _test_normalize(ds):
    ds.add(np.random.random((5000, ds.X.shape[1])))
    x = np.random.random((100, ds.X.shape[1]))
    x = ds.normalize_points(x)
    assert(np.max(x) - 1.0) < 1e-3
    x = np.random.random((100, ds.X.shape[1]))
    x = ds.standardize_points(x)
    print(np.sum(np.mean(x, 0) - np.mean(ds.X, 0)))
    assert np.sum(np.mean(x, 0) - np.mean(ds.X, 0)) < 1e-3


def _test_dataset_io():
    ds1 = Dataset(3, "float")
    ds1.add(np.random.random((3, 3)))

    with general.TemporaryDirectory() as tmp_dir:
        ds1.save(tmp_dir)
        ds2 = Dataset(3, "float")
        ds2.load(tmp_dir)
        assert(np.sum(ds1.X - ds2.X) == 0)


def test_dataset():
    _test_shingle(Dataset(3, "float"))
    _test_normalize(Dataset(3, "float"))
    _test_dataset_io()


def test_hdf5dataset():

    with general.TemporaryDirectory() as tmp_dir:
        n = np.random.randint(100, 500)
        shape = (n, n//2)
        path = os.path.join(tmp_dir, 'test.hdf5')

        ds = HDF5Dataset(path,
                         shape,
                         np.float,
                         shape,
                         np.float,
                         input_key='x',
                         output_key='y')

        # standardise when adding
        ds.set_normalizer(2)

        # input, 2 outputs
        x = np.random.normal(size=shape)
        y = np.random.normal(size=shape)
        y2 = np.random.normal(size=shape)

        # need to explictly create a h5df dataset if using more than 2 outputs
        ds.create_data('Y2', shape, np.float)

        # Add the lot in two blocks
        end = n//2
        ds.add(x[:end], [y[:end], y2[:end]])
        ds.add(x[end:], [y[end:], y2[end:]])

        # Standardise X for test
        x = stats.standardise(x)

        batch_size = ds.num_observations // 4

        # Run batcher on each input/output
        for output_key, y_temp in zip(ds.output_keys, [y, y2]):
            ds.output_key = output_key

            for i, batch in enumerate(
                ds.batcher(batch_size=batch_size)
            ):
                start = i * batch_size
                end = start + batch_size
                assert(np.allclose(batch[0], x[start:end]))
                assert(np.allclose(batch[1], y_temp[start:end]))
