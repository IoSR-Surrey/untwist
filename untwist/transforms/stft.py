"""
Forward and inverse Short-Time Fourier Transform
"""
from __future__ import division, print_function
import numpy as np
from ..base import algorithms
from ..base import parallel
from ..data import audio
from scipy import signal


class Framer(algorithms.Processor):
    '''
    Parameters:
    ----------
    window_size : size of the window in samples.
    hop_size : window increment in samples.
    pad_start : Pads the start of the signal such that the window is centred on
    the first sample (index 0).
    pad_end : Pads the end of the signal such that first sample of the window
    (not its centre) covers as many input samples as possible based on the hop
    size.
    return_copy: Returns a copy of the ndarray if true, view otherwise.

    Returns:
    -------
    If input is a Wave, returns an ndarray of shape: (num_frames, window_size)
    If input is a Spectrogram, returns an ndarray of shape:
    (num_frames, num_bands, window_size)
    '''

    def __init__(self,
                 window_size=1024,
                 hop_size=512,
                 pad_start=True,
                 pad_end=True,
                 return_copy=False):

        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.half_window = int(window_size // 2)
        self.pad_start = pad_start
        self.pad_end = pad_end
        self.return_copy = return_copy

    def calc_num_frames(self, x):
        if self.pad_end:

            if self.pad_start:

                num_frames = np.ceil(
                    (x.num_frames + self.half_window) /
                    self.hop_size)
            else:

                num_frames = np.ceil(x.num_frames / self.hop_size)

        elif self.pad_start:

            num_frames = np.floor(
                (x.num_frames + self.half_window - self.window_size) /
                self.hop_size) + 1
        else:

            num_frames = np.floor(
                (x.num_frames - self.window_size) /
                self.hop_size) + 1

        return int(num_frames)

    @algorithms.check_mono
    def process(self, x):

        # Calculate number of frames based on padding requirements
        num_frames = self.calc_num_frames(x)

        # Now padding
        pad_start = pad_end = 0

        if self.pad_start:
            pad_start = self.half_window

        # Add more on right to ensure no unexpected values when striding
        if self.pad_end:
            pad_end = self.window_size

        x = x.zero_pad(pad_start, pad_end)

        size = x.strides[1]  # no problem if Wave is mono (currently expected)

        if isinstance(x, audio.Spectrogram):

            shape = (num_frames, x.num_bands, self.window_size)
            strides = (size * self.hop_size, size * x.num_frames, size)

        elif isinstance(x, (audio.Signal, audio.Wave)):

            shape = (num_frames, self.window_size)
            strides = (size * self.hop_size, size)

        frames = np.lib.stride_tricks.as_strided(x,
                                                 shape=shape,
                                                 strides=strides)

        if self.return_copy:
            return frames.copy()
        else:
            return frames


class STFT(algorithms.Processor):
    """
    Short-Time Fourier Transform
    Input should be a mono Wave, output is a complex spectrogram
    """

    def __init__(self,
                 window='hann',
                 fft_size=1024,
                 hop_size=512):

        if isinstance(window, np.ndarray):
            self.window = window
        else:
            self.window = signal.get_window(window, fft_size)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.window_size = len(self.window)
        self.framer = Framer(self.window_size, self.hop_size, True, True, True)
        self.half_window = int(np.floor(len(self.window) / 2.0))
        self.overlap = self.window_size - self.hop_size

    # This appears to be faster than scipy's STFT
    @algorithms.check_mono
    @parallel.parallel_process(1, 2)
    def process(self, wave):

        frames = self.framer.process(wave)
        transform = np.fft.rfft(frames * self.window, self.fft_size)
        self.freqs = (np.arange(self.fft_size//2 + 1) * wave.sample_rate /
                      self.fft_size)

        return audio.Spectrogram(transform.T,
                                 wave.sample_rate,
                                 self.hop_size,
                                 self.freqs,
                                 'hertz')


class ISTFT(algorithms.Processor):
    """
    Inverse Short-Time Fourier Transform
    Input should be a complex spectrogram, output is mono Wave
    """

    def __init__(self,
                 window='hann',
                 fft_size=1024,
                 hop_size=512):

        if isinstance(window, np.ndarray):
            self.window = window
        else:
            self.window = signal.get_window(window, fft_size)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.window_size = len(self.window)
        self.overlap = self.window_size - self.hop_size

        # Correct for scipy spectrum scaling
        self.scale = 1.0 / np.sqrt(self.window.sum()**2)

        if not signal.check_COLA(self.window, self.window_size, self.overlap):
            raise Exception('COLA constraint not satisfied')

    @parallel.parallel_process(2, 1)
    def process(self, spectrogram):
        # best to use scipy here for overlap add
        t, result = signal.istft(spectrogram,
                                 1,
                                 self.window,
                                 self.window_size,
                                 self.overlap,
                                 self.fft_size)

        return audio.Wave(result * self.scale, spectrogram.sample_rate)
