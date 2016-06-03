"""
Abstract base classes for different types of algorithms (mainly Processor for now, we may have some other types of algorithms in the future)
"""

import abc

class Processor:
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
    def process(self, data):
        pass        
    

class Model:
    """
    Models are algorithms that can be trained and have some state. 
    The state can be saved and used to make predictions
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, data, target):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass
