""""
PyAudio driver. Requires pyaudio >=0.26 (non-blocking operation).
Only tested with 0.28
"""
from __future__ import print_function
import pyaudio as pa
import numpy as np
from ..soundcard import driver
from ..base import types, defaults


class PyAudioDriver(driver.AudioDriver):

    class PlaybackStream(object):

        def __init__(self, signal, range, stop_func):
            self.signal = signal
            if range == ():
                range = (0, signal.shape[0]-1)
            self.range = range
            self.pos = range[0]
            self.stop_func = stop_func

        def callback(self, in_data, count, time_info, status):
            if self.pos + count > self.range[1]:
                output = self.signal[self.pos:self.range[1], :]
                status = pa.paComplete
                if self.stop_func:
                    self.stop_func()
            else:
                output = self.signal[self.pos:self.pos + count, :]
                status = pa.paContinue
                self.pos += count
            return (output.flatten().astype(np.float32).tostring(), status)

    class RecordStream(object):

        def __init__(self, signal, max_seconds, num_channels, sr, stop_func):
            self.signal = signal
            self.max_seconds = max_seconds
            self.num_channels = num_channels
            self.pos = 0
            self.total_frames = max_seconds * sr
            self.stop_func = stop_func

        def callback(self, in_data, count, time_info, status):
            import numpy as np
            input = np.fromstring(in_data, dtype=np.float32).astype(
                types.float_)
            input_frames = len(input) / self.num_channels
            input = np.reshape(input, (input_frames, self.num_channels))
            np.vstack((self.signal, input))
            if (self.pos + count) >= self.total_frames:
                status = pa.paComplete
                print("recording done")
                if self.stop_func:
                    self.stop_func()
            else:
                status = pa.paContinue
                self.pos += count
            return (None, status)

    def __init__(self):
        self.pyaudio = pa.PyAudio()
        self.streams = {}
        self.recordings = {}
        self.current_stream_id = -1

    def wrap_stop_func(self, stream_id, func):
        def wrapped():
            self.clean(stream_id)
            if func is not None:
                func()
        return wrapped

    def play(self, signal, range=(), sr=defaults.sample_rate, stop_func=None):
        self.current_stream_id += 1
        stream = PyAudioDriver.PlaybackStream(
            signal, range,
            self.wrap_stop_func(self.current_stream_id, stop_func)
        )

        pa_stream = self.pyaudio.open(
            format=pa.paFloat32,
            channels=signal.shape[1],
            rate=sr,
            output=True,
            stream_callback=stream.callback)

        self.streams[self.current_stream_id] = pa_stream
        return self.current_stream_id

    def record(self,
               max_seconds,
               num_channels,
               sr=defaults.sample_rate,
               stop_func=None):

        self.current_stream_id += 1
        self.recordings[self.current_stream_id] = np.ndarray((0, num_channels))

        stream = PyAudioDriver.RecordStream(
            self.recordings[self.current_stream_id],
            max_seconds, num_channels, sr,
            self.wrap_stop_func(self.current_stream_id, stop_func)
        )

        pa_stream = self.pyaudio.open(
            format=pa.paFloat32,
            channels=num_channels,
            rate=sr,
            input=True,
            stream_callback=stream.callback)

        self.streams[self.current_stream_id] = pa_stream
        return self.current_stream_id

    def clean(self, stream_id):
        if stream_id in self.streams:
            del self.streams[stream_id]
        if stream_id in self.recordings:
            del self.recordings[stream_id]

    def stop(self, stream_id):
        ret = None
        if stream_id in self.streams and self.streams[stream_id].is_active():
            self.streams[stream_id].stop_stream()
            self.streams[stream_id].close()
        if stream_id in self.recordings:
            recorded = self.recordings[stream_id]
            ret = recorded
        self.clean(stream_id)
        return ret
