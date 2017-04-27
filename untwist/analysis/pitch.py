"""
Pitch detection algorithms
"""
from __future__ import division, print_function
import numpy as np
from scipy import interpolate
from untwist.base.algorithms import Processor

'''
TODO
- Check weights multiply as currenty unused
'''


class ZCR(Processor):
    """
    Zero-crossing rate (can be used as rough pitch estimator or as feature).
    """

    def __init__(self):
        pass

    def process(self, wave):
        n_crossings = np.count_nonzero(np.diff(np.sign(wave[:, 0])))
        return 0.5 * n_crossings / wave.duration


class HPS(Processor):
    """
    Harmonic product spectrum pitch estimator.  Based on implementation in:
        Lerch, A. (2012). An introduction to audio content analysis:
        Applications in signal processing and music informatics. John Wiley &
        Sons.
    """

    def __init__(self, n_harms=8,  min_pitch=50, max_pitch=5000):
        self.n_harms = n_harms
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def process(self, X):
        min_bin = np.round((float(self.min_pitch)/X.sample_rate) * X.shape[0])
        max_bin = np.round((float(self.max_pitch)/X.sample_rate) * X.shape[0])

        HPS = X.magnitude().copy()
        for h in range(2, self.n_harms):
            H = X.magnitude()[0:HPS.shape[0]:h, :]
            HP = np.zeros_like(HPS)
            HP[0:H.shape[0], :] = H
            HPS *= HP
        pitch_bins = min_bin+np.argmax(HPS[min_bin:max_bin, :], 0)
        return pitch_bins * 0.5 * X.sample_rate / X.shape[0]


class YINFFT(Processor):
    """
    YINFFT F0 estimation algorithm
    Brossier, P. M. (2006). Automatic annotation of musical audio for
    interactive applications (Doctoral dissertation, Queen Mary University of
    London).
    Based on Essentia C++ implementation http://essentia.upf.edu
    """
    freq_mask = [
        0.,
        20.,
        25.,
        31.5,
        40.,
        50.,
        63.,
        80.,
        100.,
        125.,
        160.,
        200.,
        250.,
        315.,
        400.,
        500.,
        630.,
        800.,
        1000.,
        1250.,
        1600.,
        2000.,
        2500.,
        3150.,
        4000.,
        5000.,
        6300.,
        8000.,
        9000.,
        10000.,
        12500.,
        15000.,
        20000.,
        25100]

    weight_mask = [
        -75.8, -70.1, -60.8, -52.1, -44.2, -37.5, -31.3, -25.6, -20.9, -16.5,
        -12.6, -9.6, -7.0, -4.7, -3.0, -1.8, -0.8, -0.2, -0.0, 0.5, 1.6, 3.2,
        5.4, 7.8, 8.1, 5.3, -2.4, -11.1, -12.8, -12.2, -7.4, -17.8, -17.8,
        -17.8]

    def compute_weights(self):
        self.weights = np.zeros((self.n_bins, 1))
        j = 1
        bin_hz = float(self.sample_rate) / (2 * (self.n_bins - 1))
        for i in range(self.n_bins):
            freq = i * bin_hz
            while(self.freq_mask[j] < freq):
                j += 1
            a0 = self.weight_mask[j-1]
            f0 = self.freq_mask[j-1]
            a1 = self.weight_mask[j]
            f1 = self.freq_mask[j]
            if f0 == 0:
                self.weights[i] = (a1-a0)/f1*freq + a0
            else:
                self.weights[i] = (a1-a0)/(f1-f0)*freq +\
                    (a0 - (a1 - a0)/(f1/f0 - 1.))
            self.weights[i] = np.power(10, self.weights[i] / 10.0)

    def __init__(self,
                 n_bins,
                 sample_rate,
                 min_pitch=100,
                 max_pitch=5000,
                 interp=True):
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.compute_weights()
        self.tau_max = int(
            min(np.ceil(float(sample_rate) / min_pitch), n_bins))
        self.tau_min = int(
            min(np.floor(float(sample_rate) / max_pitch), n_bins))
        self.interp = interp

    def process(self, spectrogram):
        # mult = np.multiply(self.weights, np.ones((1, spectrogram.shape[1])))
        square_mag = np.power(spectrogram.magnitude(), 2)
        square_mag = np.vstack(
            (square_mag[1:, :], np.flipud(square_mag[1:, :])))
        square_mag_sum = np.sum(square_mag, 0)
        energy_threshold = np.percentile(square_mag_sum, 25)
        res = np.fft.rfft(square_mag.T, 2 * (self.n_bins - 1)).T
        resN = np.abs(res)
        resP = np.angle(res)
        yin = np.zeros((spectrogram.shape))
        yin[0, :] = 1
        tmp = np.zeros((spectrogram.shape[1],))
        for tau in range(1, self.n_bins):
            yin[tau, :] = square_mag_sum - resN[tau, :] * np.cos(resP[tau, :])
            tmp += yin[tau, :]
            yin[tau, :] *= tau/tmp

        tau = self.tau_min + np.argmin(yin[self.tau_min:self.tau_max, :], 0)
        y_min = np.min(yin[self.tau_min:self.tau_max, :], 0)
        tau = tau[:, np.newaxis]
        if self.interp:
            ranges = np.hstack((tau - 5, tau - 4, tau - 3, tau-2, tau-1, tau,
                                tau+1, tau + 2, tau + 3, tau + 4, tau + 5))
            new_tau = np.empty_like(tau)
            for frame in range(spectrogram.shape[1]):
                r = ranges[frame]
                r[0] = max(r[0], 0)
                r[1] = min(r[1], self.n_bins-1)
                val = yin[r.astype(int), frame]
                tck = interpolate.splrep(r, val, s=0, k=2)
                xnew = np.arange(r[0], r[-1], (r[-1] - r[0]) / 10)
                ynew = interpolate.splev(xnew, tck, der=0)
                new_tau[frame] = xnew[np.argmin(ynew)]
                y_min[frame] = np.min(ynew)
            tau = new_tau
        tau = tau[:, 0]
        pitch_confidence = 1 - y_min
        pitch = np.zeros((spectrogram.shape[1],))
        pitch[tau != 0] = np.nan_to_num(
            self.sample_rate * 1.0 / (tau[tau != 0]))
        pitch[tau == 0] = 0
        pitch_confidence[tau == 0] = 0

        pitch[square_mag_sum < energy_threshold] = 0
        pitch_confidence[square_mag_sum < energy_threshold] = 0
        return (pitch, pitch_confidence)
