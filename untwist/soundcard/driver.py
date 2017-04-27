""""
Abstract base class that defines the methods that a soundcard 'driver'
(typically wrapping some library like pyaudio or pygame) should implement. The
most appropriate available implementation should be selected and bound at
module init.
"""

import abc


class AudioDriver:
    __metaclass__ = abc.ABCMeta

    """
    Expected to return a stream id (and not block)
    """
    @abc.abstractmethod
    def play(self, audio_data):
        raise NotImplementedError

    """
    Stop a stream
    """
    @abc.abstractmethod
    def stop(self, stream_id):
        raise NotImplementedError

    """
    Expected to return a stream id (and not block either)
    """
    @abc.abstractmethod
    def record(self):
        raise NotImplementedError


class RTAudioDriver:

    """
    Add a callback function to be called for real time operation
    The function will be expected to process an ndarray
    (TODO: define a way for the client to set the buffer size)
    """
    @abc.abstractmethod
    def add_callback(self, func):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError
