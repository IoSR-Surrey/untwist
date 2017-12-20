from . import base
from ..utilities import conversion
import numpy as np


class RLBFilter(base.SOS):

    def __init__(self, sample_rate):
        super(RLBFilter, self).__init__(None, sample_rate)

        ff = np.array([1.0, -2.0, 1.0])
        fb = np.array([1.0, -1.99004745483398, 0.99007225036621])

        fs_old = 48000
        ff, fb = conversion.biquad_coefficients(
            ff, fb, fs_old, self.sample_rate)

        self.append(np.r_[ff, fb])


class PreFilter(base.SOS):

    def __init__(self, sample_rate):
        super(PreFilter, self).__init__(None, sample_rate)


        ff = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
        fb = np.array([1.0, -1.69065929318241, 0.73248077421585])

        fs_old = 48000
        ff, fb = conversion.biquad_coefficients(
            ff, fb, fs_old, self.sample_rate)

        self.append(np.r_[ff, fb])


class KFilter(base.SOS):

    def __init__(self, sample_rate):
        super(KFilter, self).__init__(None, sample_rate)

        self.append(PreFilter(sample_rate).sos)
        self.append(RLBFilter(sample_rate).sos)
