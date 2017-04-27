"""
Multiprocessing decorator
"""
from __future__ import absolute_import
import multiprocessing as mp
import numpy as np
from types import MethodType
try:
    import copy_reg
except:
    import copyreg as copy_reg

n_threads = 4

# allow pickling instance methods
# https://gist.github.com/bnyeggen/1086393


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)


class parallel_process(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def worker(self, a):
        return a[0].process(*a[1:])

    def __call__(self, process_func):

        def wrapper(*args):
            data_obj = args[1]
            if (len(data_obj.shape) <= self.input_dim or
                    data_obj.shape[-1] == 1):
                return process_func(*args)
            else:
                pool = mp.Pool(mp.cpu_count())  # TODO: make configurable
                arglist = [
                    (args[0],) +
                    (data_obj[..., i],) +
                    args[2:]
                    for i in range(data_obj.shape[-1])
                ]
                result = pool.map(self.worker, arglist)
                if self.output_dim > self.input_dim:  # expanding
                    return np.stack(result, -1)
                else:  # contracting
                    return np.concatenate(result, -1)
        return wrapper
