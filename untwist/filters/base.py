from .. import data
from ..base import defaults, algorithms
from scipy import signal
import numpy as np


class Filter(algorithms.Processor):

    def __init__(self,
                 ff_coefs=None,
                 fb_coefs=None,
                 sample_rate=defaults.sample_rate):

        self.ff_coefs = ff_coefs
        self.fb_coefs = fb_coefs
        self.sample_rate = sample_rate

    def process(self, x):

        if isinstance(x, data.Spectrogram):
            axis = 1
        else:
            axis = 0

        out = signal.lfilter(self.ff_coefs,
                             self.fb_coefs,
                             x,
                             axis=axis).view(type(x))

        out.sample_rate = self.sample_rate
        return out

    def response(self, freqs=None, num_points=1024):

        if freqs is not None:
            freqs = np.array(freqs)
            w = 2 * np.pi * freqs / self.sample_rate
        else:
            w = num_points

        w2, h = signal.freqz(self.ff_coefs, self.fb_coefs, w)

        return data.Spectrum(h,
                             self.sample_rate,
                             freqs=(self.sample_rate * w2 / (2 * np.pi)))

    def plot_magnitude(self):

        return self.response().plot_magnitude()


class SOS(Filter):

    def __init__(self,
                 sos=None,
                 sample_rate=defaults.sample_rate):

        if sos is not None:
            sos.shape = (-1, 6)

        self.sos = sos
        self.sample_rate = sample_rate

    def append(self, sos):

        if self.sos is None:
            self.sos = sos
        else:
            self.sos = np.vstack((self.sos, sos))

    def process(self, x):

        if isinstance(x, data.Spectrogram):
            axis = 1
        else:
            axis = 0

        out = signal.sosfilt(self.sos, x, axis=axis).view(type(x))
        out.sample_rate = self.sample_rate

        return out

    def response(self, freqs=None, num_points=1024):

        if freqs is not None:
            freqs = np.array(freqs)
            w = 2 * np.pi * freqs / self.sample_rate
        else:
            w = num_points

        w2, h = signal.sosfreqz(self.sos, w)

        return data.Spectrum(h,
                             self.sample_rate,
                             freqs=(self.sample_rate * w2 / (2 * np.pi)))
