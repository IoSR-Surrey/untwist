import os, shutil
from ...data import Dataset, MMDataset
import numpy as np


def _test_shingle(ds):    
    x = np.random.random((300, ds.X.shape[1]))
    y = ds.shingle(x,5)
    z = ds.unshingle(y,5)
    assert np.sum(x-z) < 1e-10

def _test_normalize(ds):
    ds.add(np.random.random((5000, ds.X.shape[1])))
    x = np.random.random((100, ds.X.shape[1]))
    x = ds.normalize_points(x)
    assert(np.max(x) - 1.0) < 1e-3
    x = np.random.random((100, ds.X.shape[1]))
    x = ds.standardize_points(x)    
    print np.sum(np.mean(x,0) - np.mean(ds.X, 0))    
    assert np.sum(np.mean(x,0) - np.mean(ds.X, 0)) < 1e-3

def _test_dataset_io():
    ds1 = Dataset(3,"float")
    ds1.add(np.random.random((3,3)))
    ds1.save("/tmp/")
    ds2 = Dataset(3,"float")
    ds2.load("/tmp/")
    assert(np.sum(ds1.X - ds2.X) == 0)
    os.remove("/tmp/X.npy")
    os.remove("/tmp/Y.npy")

def test_dataset():
    _test_shingle(Dataset(3,"float"))
    _test_normalize(Dataset(3,"float"))
    _test_dataset_io()
    
def test_mmdataset():
    ds_path = "/tmp/testDS/"
    os.mkdir("/tmp/testDS/")
    ds = MMDataset(ds_path, 10)
    _test_shingle(ds)
    _test_normalize(ds)
    shutil.rmtree(ds_path)