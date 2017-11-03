"""
Abstract base classes for different types of algorithms (mainly Processor for
now, we may have some other types of algorithms in the future)
"""
from __future__ import division, print_function
import abc


class Processor(object):
    """
    Processor objects are the core of the framework. It should be possible to
    implement any algorithm as a processor. The interface is similar to most
    audio signal processing frameworks. The object encapsulates a function that
    processes some information. Hence the only methods are __init__()
    (for initialization) and process().
    This base class defines the interface for processors. Parameters and return
    values for the process() method will vary accross algorithms
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self):
        pass


class Model(object):
    """
    Models are algorithms that can be trained and have some state.
    The state can be saved and used to make predictions
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, fname):
        pass

    @abc.abstractmethod
    def save(self, fname):
        pass

'''
Decorators
'''


def is_mono_exception(signal):
    from ..data import audio
    if isinstance(signal, audio.Signal):
        if not signal.is_mono():
            raise Exception("Unsupported channel layout")


def check_mono(func):
    '''
    Raises an exception if any Signal objective is mono.
    Example usage:

        @check_mono
        def process(self, data):

    '''

    def wrapper(*args, **kwargs):

        [is_mono_exception(_) for _ in args]

        if(kwargs):
            [is_mono_exception(_) for _ in kwargs.values()]

        return func(*args, **kwargs)

    return wrapper
