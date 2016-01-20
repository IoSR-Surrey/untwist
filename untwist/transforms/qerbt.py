"""
Quadratic ERB transform

Based on:
Emmanuel Vincent, "Musical source separation using time-frequency source priors," 
IEEE Trans. on Audio, Speech and Language Processing, 14(1):91-98, 2006
"""

from untwist.base import Processor
from untwist.data import audio
import numpy as np
from numpy.lib import stride_tricks
from scipy import signal

def fftfilt(b, x, *n):
    N_x = len(x)
    N_b = len(b)
    N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
    cost = np.ceil(N_x / (N - N_b + 1)) * N * (np.log2(N) + 1)
    N_fft = int(N[np.argmin(cost)])
    N_fft = int(N_fft)
    
    # Compute the block length:
    L = int(N_fft - N_b + 1)
    
    # Compute the transform of the filter:
    H = np.fft.fft(b,N_fft)

    y = np.zeros(N_x, x.dtype)
    i = 0
    while i <= N_x:
        il = np.min([i+L,N_x])
        k = np.min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y  
    
class QERBT(Processor):

    def __init__(self, n_bins = 350, w_len = 2048):
        self.n_bins = n_bins
        self.w_len = w_len
        
    def process(self, wave):
        wave.check_mono()
        sample_rate = wave.sample_rate
        erb_max= 9.26 * np.log(0.00437 * wave.sample_rate/2.0 + 1)
        erb_freqs = np.arange(0, self.n_bins) * erb_max / float(self.n_bins - 1)
        hz_freqs =(np.exp(erb_freqs / 9.26) - 1) / 0.00437
        widths = 0.5 * (self.n_bins - 1) / erb_max * 9.26 * 0.00437 \
            * wave.sample_rate * np.exp(-erb_freqs / 9.26) - 0.5
        n_frames = int(np.ceil(2 * wave.num_frames / float(self.w_len)))
        half_win = self.w_len / 2.0
        wave = wave.zero_pad(0, (n_frames +1) * half_win - wave.num_frames)
        wave = audio.Wave(signal.hilbert(wave), wave.sample_rate)
        window = np.sin(np.arange(0.5, self.w_len + 1 - 0.5) / self.w_len * np.pi)
        window = window[:, np.newaxis]
        swindow = np.zeros(((n_frames + 1) * half_win,1))
        for t in range(n_frames):
            start_idx = t * half_win
            end_idx = t * half_win + self.w_len
            swindow[start_idx:end_idx,:] = swindow[start_idx:end_idx,:] + np.square(window)
        swindow = np.sqrt(swindow)
        result = np.zeros((self.n_bins, n_frames))
        for b in range(self.n_bins - 1, -1, -1): 
            width = np.round(widths[b])
            filter = np.hanning((2 * width) + 1) * np.exp(np.complex(0,1) * 2 * \
                np.pi * hz_freqs[b] / sample_rate * np.arange(-width, width + 1))
            band = fftfilt(filter, wave.zero_pad(0, 2*width)[:,0])
            band = band[width:width + (n_frames + 1) * self.w_len / 2, np.newaxis]
    
            for t in range(n_frames):
                frame = band[t * self.w_len / 2:t * self.w_len / 2 + self.w_len,:]\
                    * window / swindow[t * self.w_len / 2 : t * self.w_len / 2 + self.w_len]
                result[b, t] = 1/np.square(width+1) * np.real(np.conj(np.dot(frame.conj().T,frame)))
        return result   