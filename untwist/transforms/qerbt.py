"""
Quadratic ERB transform based on:

Emmanuel Vincent, "Musical source separation using time-frequency source
priors" , IEEE Trans. on Audio, Speech and Language Processing, 14(1):91-98,
2006.
"""
from __future__ import division, print_function
import numpy as np
from scipy import signal
from ..base import algorithms, parallel, defaults
from ..data import audio
from numpy import linalg


def fftfilt(b, x, *n):
    N_x = len(x)
    N_b = len(b)
    N = 2**np.arange(np.ceil(np.log2(N_b)), np.floor(np.log2(N_x)))
    cost = np.ceil(N_x / (N - N_b + 1)) * N * (np.log2(N) + 1)
    N_fft = int(N[np.argmin(cost)])
    N_fft = int(N_fft)
    # Compute the block length:
    L = int(N_fft - N_b + 1)
    # Compute the transform of the filter:
    H = np.fft.fft(b, N_fft)
    y = np.zeros(N_x, x.dtype)
    i = 0
    while i <= N_x:
        il = np.min([i+L, N_x])
        k = np.min([i+N_fft, N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il], N_fft)*H, N_fft)  # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]                          # and add
        i += L
    return y


def hz2erb(f):
    return 9.26 * np.log(0.00437 * f + 1)


def erb2hz(f):
    return (np.exp(f / 9.26) - 1) / 0.00437


class QERBT(algorithms.Processor):
    """
    Quadratic ERB transform processor, with independent window length
    and number of bins. Returns a Spectrogram.
    """

    def __init__(self, n_bins=350, w_len=2048, sr=defaults.sample_rate):
        self.n_bins = n_bins
        self.w_len = w_len
        self.sr = sr
        self.window = np.sin(
            np.arange(0.5, self.w_len + 1 - 0.5) / self.w_len * np.pi)
        self.window = self.window[:, np.newaxis]
        self.make_filterbank()

    def make_filterbank(self):
        erb_max = hz2erb(self.sr/2.0)
        self.erb_freqs = np.arange(
            0, self.n_bins) * erb_max / (self.n_bins - 1)
        self.hz_freqs = erb2hz(self.erb_freqs)
        self.widths = np.round(
            0.5 * (self.n_bins - 1) / erb_max * 9.26 * 0.00437 *
            self.sr * np.exp(-self.erb_freqs / 9.26) - 0.5)
        self.filters = []
        for b in range(self.n_bins):
            w = self.widths[b]
            f = self.hz_freqs[b]
            exponential = np.exp(
                np.complex(0, 1) * 2 * np.pi * f / self.sr *
                np.arange(-w, w + 1))
            self.filters.append(np.hanning(2 * w + 1) * exponential)

    def make_signal_window(self, n_frames):
        half_win = int(self.w_len / 2.0)
        signal_window = np.zeros(((n_frames + 1) * half_win, 1))
        for t in range(n_frames):
            s = t * half_win
            e = t * half_win + self.w_len
            signal_window[s:e, :] = signal_window[
                s:e, :] + np.square(self.window)
        signal_window = np.sqrt(signal_window)
        return signal_window

    @algorithms.check_mono
    @parallel.parallel_process(1, 2)
    def process(self, wave):
        if wave.sample_rate != self.sr:
            raise Exception("Wrong sample rate")
        n = int(np.ceil(2 * wave.num_frames / self.w_len))
        m = (n + 1) * self.w_len / 2
        swindow = self.make_signal_window(n)
        win_ratios = [
            self.window /
            swindow[t * self.w_len // 2: t * self.w_len // 2 + self.w_len]
            for t in range(n)]
        wave = wave.zero_pad(0, int(m - wave.num_frames))
        wave = audio.Wave(signal.hilbert(wave), wave.sample_rate)
        result = np.zeros((self.n_bins, n))

        for b in range(self.n_bins):
            w = self.widths[b]
            wc = 1 / np.square(w + 1)
            filter = self.filters[b]
            band = fftfilt(filter, wave.zero_pad(0, int(2 * w))[:, 0])
            band = band[int(w): int(w + m), np.newaxis]
            for t in range(n):
                frame = (band[t * self.w_len // 2:
                              t * self.w_len // 2 + self.w_len, :] *
                         win_ratios[t])
                result[b, t] = (wc * np.real(
                    np.conj(np.dot(frame.conj().T, frame)))
                )

        return audio.Spectrogram(result,
                                 self.sr,
                                 self.w_len / 2,
                                 self.erb_freqs,
                                 'cam')


class QERBFilter(QERBT):
    """
    Filterbank based on the QERBT transform.
    """

    def __init__(self, n_bins=350, w_len=2048, sr=defaults.sample_rate):
        super(QERBFilter, self).__init__()
        self.make_bin_weights()

    def make_bin_weights(self):
        erb_max = hz2erb(self.sr/2.0)
        ngrid = 1000
        self.erb_grid = np.arange(ngrid) * erb_max / (ngrid - 1)
        hz_grid = (np.exp(self.erb_grid / 9.26) - 1) / 0.00437
        resp = np.zeros((ngrid, self.n_bins))
        for b in range(self.n_bins):
            w = self.widths[b]
            r = (2.0 * w + 1.0) / self.sr * (hz_grid - self.hz_freqs[b])
            resp[:, b] = np.square(np.sinc(r) + 0.5 *
                                   np.sinc(r + 1.0) + 0.5 *
                                   np.sinc(r - 1.0))
        self.weights = np.dot(linalg.pinv(resp), np.ones((ngrid, 1)))

    @algorithms.check_mono
    def process(self, wave, W):
        if wave.sample_rate != self.sr:
            raise Exception("Wrong sample rate")
        n = int(np.ceil(2 * wave.num_frames / self.w_len))
        m = (n + 1) * self.w_len / 2
        if n != W.shape[1]:
            raise Exception("Wrong size for W")
        swindow = self.make_signal_window(n)
        win_ratios = [self.window / swindow[t * self.w_len // 2:
                                            t * self.w_len // 2 + self.w_len]
                      for t in range(n)]
        wave = wave.zero_pad(
            0, int((n + 1) * self.w_len / 2.0 - wave.num_frames))
        wave = audio.Wave(signal.hilbert(wave), wave.sample_rate)
        result = np.zeros(wave.shape)

        for b in range(self.n_bins):
            w = self.widths[b]
            # wc = 1/np.square(w + 1)
            filter = 1/w * self.filters[b]
            band = fftfilt(filter, wave.zero_pad(0, int(2 * w))[:, 0])
            band = band[int(w): int(w + (n + 1) * self.w_len / 2), np.newaxis]
            out_band = audio.Wave(
                np.zeros(
                    band.shape,
                    np.complex128),
                wave.sample_rate)
            for t in range(n):
                start = int(t * self.w_len / 2)
                end = int(t * self.w_len / 2 + self.w_len)
                frame = band[start:end, :] * win_ratios[t]**2
                out_band[start:end, :] = out_band[
                    start:end, :] + frame * W[b, t]
            out_band = np.real(
                               fftfilt(
                                   filter, out_band.zero_pad(0, int(2 * w))
                                   [:, 0]))
            result[:, 0] = result[
                :, 0] + self.weights[b] * out_band[int(w): int(w + m)]
        return result
