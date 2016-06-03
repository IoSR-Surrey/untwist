"""
Forward and inverse STFT

"""

import numpy as np
from numpy.lib import stride_tricks
from ..base import Processor
from ..data import audio


"""
Short-time Fourier Transform
Input should be a mono Wave, output is a complex spectrogram
"""

class STFT(Processor):

    def __init__(self, window=None, fft_size=1024, hop_size=512):
        if window is None:
            self.window = np.hanning(fft_size)
        else:
            self.window = window
        self.fft_size = fft_size
        self.hop_size = hop_size        
        
    def process(self, wave):
        wave.check_mono()
        window_size = len(self.window)
        half_window = np.floor(window_size / 2.0)
        wave = wave.zero_pad(half_window, half_window)        
        num_frames = 1 + np.ceil((wave.shape[0] - window_size) / 
            float(self.hop_size))                
        col_size = wave.strides[0]
        frames = stride_tricks.as_strided(wave, shape=(num_frames, window_size),
            strides=(col_size*self.hop_size, col_size)).copy()
        frames *= self.window
        transform = np.fft.rfft(frames, self.fft_size)
        return audio.Spectrogram(transform.T,
            wave.sample_rate, len(self.window), self.hop_size)
            
            
"""
Inverse Short-time Fourier Transform
Input should be a complex spectrogram, output is mono Wave
"""   
class ISTFT(Processor):

    def __init__(self, window=None, fft_size=1024, hop_size=512, sample_rate = 44100):
        if window is None:
            self.window = np.hanning(fft_size)
        else:
            self.window = window
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
    def process(self, spectrogram):
        frames = np.fft.irfft(spectrogram.T)
        result_length = ((frames.shape[0] - 1) * self.hop_size) + self.fft_size
        result = np.zeros((result_length, 1))
        for i in range(frames.shape[0]):
            indices = i * self.hop_size + np.r_[0:self.fft_size]
            result[indices] = result[indices] + frames[i,:].reshape(self.fft_size,1)
        result = result[self.hop_size:result_length - self.hop_size]
        return audio.Wave(result, self.sample_rate)