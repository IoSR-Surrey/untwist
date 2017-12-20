from ..base import defaults
from . import base
import numpy as np


class Biquad(base.SOS):

    def __init__(self, sample_rate):
        super(Biquad, self).__init__(np.zeros((1, 3)), self.sample_rate)


class HighPass(Biquad):

    def __init__(self, f0, q=defaults.q, sample_rate=defaults.sample_rate):
        super(HighPass, self).__init__(sample_rate)

        w0 = 2.0 * np.pi * f0 / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2.0 * q)

        self.sos[0] = (1.0 + cos_w0) / 2.0
        self.sos[1] = -(1.0 + cos_w0)
        self.sos[2] = (1.0 + cos_w0) / 2.0

        self.sos[3] = 1.0 + alpha
        self.sos[4] = -2.0 * cos_w0
        self.sos[5] = 1.0 - alpha


class LowPass(Biquad):

    def __init__(self, f0, q=defaults.q, sample_rate=defaults.sample_rate):
        super(LowPass, self).__init__(sample_rate)

        self.sample_rate = sample_rate
        self.sos, self.sos = np.zeros((2, 3))

        w0 = 2.0 * np.pi * f0 / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2.0 * q)

        self.sos[0] = (1.0 - cos_w0) / 2.0
        self.sos[1] = 1.0 - cos_w0
        self.sos[2] = (1.0 - cos_w0) / 2.0

        self.sos[3] = 1.0 + alpha
        self.sos[4] = -2.0 * cos_w0
        self.sos[5] = 1.0 - alpha


class LowShelf(Biquad):

    def __init__(self,
                 f0,
                 slope=1,
                 gain_dB=3,
                 sample_rate=defaults.sample_rate):
        super(LowShelf, self).__init__(sample_rate)

        self.sample_rate = sample_rate
        self.sos, self.sos = np.zeros((2, 3))

        a = 10.0 ** (gain_dB / 40.0)
        w0 = 2.0 * np.pi * f0 / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = (sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) *
                                        (1.0 / np.minimum(slope, 1) - 1) + 2))

        self.sos[0] = (a * ((a + 1) - (a - 1) *
                            cos_w0 + 2.0 * np.sqrt(a) * alpha
                            )
                       )

        self.sos[1] = (2.0 * a * ((a - 1) - (a + 1) * cos_w0))

        self.sos[2] = (a * ((a + 1) - (a - 1) *
                            cos_w0 - 2.0 * np.sqrt(a) * alpha
                            )
                       )

        self.sos[3] = ((a + 1) + (a - 1) * cos_w0 + 2.0 * np.sqrt(a) * alpha)

        self.sos[4] = (-2.0 * ((a - 1) + (a + 1) * cos_w0))

        self.sos[5] = ((a + 1) + (a - 1) * cos_w0 - 2.0 * np.sqrt(a) * alpha)


class HighShelf(Biquad):

    def __init__(self,
                 f0,
                 slope=1,
                 gain_dB=3,
                 sample_rate=defaults.sample_rate):
        super(HighShelf, self).__init__(sample_rate)

        a = 10.0 ** (gain_dB / 40.0)
        w0 = 2.0 * np.pi * f0 / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = (sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) *
                                        (1.0 / np.minimum(slope, 1) - 1) + 2))

        self.sos[0] = (a * ((a + 1) + (a - 1) *
                            cos_w0 + 2.0 * np.sqrt(a) * alpha
                            )
                       )

        self.sos[1] = (-2.0 * a * ((a - 1) + (a + 1) * cos_w0))

        self.sos[2] = (a * ((a + 1) + (a - 1) *
                            cos_w0 - 2.0 * np.sqrt(a) * alpha
                            )
                       )

        self.sos[3] = ((a + 1) - (a - 1) * cos_w0 + 2.0 * np.sqrt(a) * alpha)

        self.sos[4] = (2.0 * ((a - 1) - (a + 1) * cos_w0))

        self.sos[5] = ((a + 1) - (a - 1) * cos_w0 - 2.0 * np.sqrt(a) * alpha)


class Peaking(Biquad):

    def __init__(self,
                 f0,
                 q=defaults.q,
                 gain_dB=3,
                 sample_rate=defaults.sample_rate):
        super(Peaking, self).__init__(sample_rate)

        self.sample_rate = sample_rate
        self.sos, self.sos = np.zeros((2, 3))

        a = 10.0 ** (gain_dB / 40.0)
        w0 = 2.0 * np.pi * f0 / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2.0 * self.q)

        self.sos[0] = 1 + alpha * a
        self.sos[1] = -2 * cos_w0
        self.sos[2] = 1 - alpha * a

        self.sos[3] = 1 + alpha / a
        self.sos[4] = -2 * cos_w0
        self.sos[5] = 1 - alpha / a
