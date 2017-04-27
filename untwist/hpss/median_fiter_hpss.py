"""
Harmonic-Percussive separation using median filters, from:

D. FitzGerald, 'Harmonic/percussive separation using median filtering',
Proceedings of the 13th International Conference on Digital Audio Effects
(DAFx-10), 2010.
"""
from __future__ import division, print_function
import numpy as np
from scipy import signal
from ..base.algorithms import Processor
from ..data import RatioMask


class MedianFilterHPSS(Processor):
    """
    Harmonic-Percussive separation, using a horizontal median filter of size
    harmonic_length and a vertical median filter of size percussive_length.
    """

    def __init__(self, harmonic_length, percussive_length,
                 mask_class=RatioMask, mask_exp=2):
        self.h_kernel = (1, harmonic_length)
        self.p_kernel = (percussive_length, 1)
        self.mask_class = mask_class
        self.mask_exp = mask_exp

    def process(self, S):
        mag = S.magnitude()
        H = np.empty_like(mag)
        H[:] = signal.medfilt(mag, self.h_kernel)
        P = np.empty_like(mag)
        P[:] = signal.medfilt(mag, self.p_kernel)
        h_mask = self.mask_class(H, P, self.mask_exp)
        p_mask = self.mask_class(P, H, self.mask_exp)
        return (S * h_mask, S * p_mask)
