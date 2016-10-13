"""
Forward and inverse Short-Time Fourier Transform
"""

import numpy as np
from numpy.lib import stride_tricks
from ..base import Processor, parallel_process
from ..data import audio



class STFT(Processor):
    """
    Short-Time Fourier Transform
    Input should be a mono Wave, output is a complex spectrogram
    """

    def __init__(self, window = None, fft_size = 1024, hop_size = 512):
        if window is None:
            self.window = np.hanning(fft_size)
        else:
            self.window = window
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = len(self.window)
        self.half_window = int(np.floor(len(self.window) / 2.0))
     
    @parallel_process(1,2)
    def process(self, wave):
        wave.check_mono()
        wave = wave.zero_pad(self.half_window, self.half_window)
        num_frames = int(1 + np.ceil((wave.shape[0] - self.window_size) /
            float(self.hop_size)))                
        col_size = wave.strides[0]
        frames = stride_tricks.as_strided(wave, shape=(num_frames, self.window_size),
            strides=(col_size*self.hop_size, col_size)).copy()
        frames *= self.window
        transform = np.fft.rfft(frames, self.fft_size)
        return audio.Spectrogram(transform.T,
            wave.sample_rate, len(self.window), self.hop_size)
            
            
class ISTFT(Processor):
    """
    Inverse Short-Time Fourier Transform
    Input should be a complex spectrogram, output is mono Wave
    """   

    def __init__(self, window = None, fft_size = 1024, hop_size = 512, sample_rate = 44100):
        if window is None:
            self.window = np.hanning(fft_size)
        else:
            self.window = window
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window_size = len(self.window)
        self.half_window = int(np.floor(len(self.window) / 2.0))
        
    @parallel_process(2,1)
    def process(self, spectrogram):
        frames = np.fft.irfft(spectrogram.T)
        n_frames = frames.shape[0]
        result_length = ((n_frames - 1) * self.hop_size) + self.fft_size
        result = np.zeros((result_length, 1))
        norm = np.zeros((result_length, 1))
        wsquare = (self.window * self.window).reshape(-1, 1)
        for i in range(n_frames):
            indices = i * self.hop_size + np.r_[0:self.fft_size]
            result[indices] +=  frames[i,:].reshape(-1, 1) * self.window.reshape(-1, 1)
            norm[indices] += wsquare
        min_float = np.finfo(spectrogram.dtype).tiny
        result /= np.where(norm > min_float, norm, 1.0)
        result = result[self.half_window:result_length - self.half_window]
        return audio.Wave(result, self.sample_rate)