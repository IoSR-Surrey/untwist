"""
Audio representations, i.e. Wave, Spectrum, Spectrogram.
Should always inherit from ndarray, but utility functions may be added, e.g. loading audio files, playing or plotting

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from untwist.base import types
from untwist.soundcard import audio_driver
from untwist.base.exceptions import *


"""
Time domain signal. Layout is one column per channel
"""

class Signal(np.ndarray):
    
    __array_priority__ = 10
    def __new__(cls, data, sample_rate= 44100):
        instance = np.ndarray.__new__(cls, 
            data.shape, dtype = data.dtype, strides = data.strides, buffer = data)
        instance.sample_rate = sample_rate
        return instance

    def __array_finalize__(self, obj):
        if obj is None: return
        self.sample_rate = getattr(obj, 'sample_rate', None)

    def __array_prepare__(self, out_arr, context = None):        
        return np.ndarray.__array_prepare__(self, out_arr, context)
        
    def __array_wrap__(self, out_arr, context = None):        
        return np.ndarray.__array_wrap__(self, out_arr, context)
                
    @property  
    def num_channels(self):
        return 1 if len(self.shape)==1 else self.shape[1]

    @property
    def num_frames(self):
        return self.shape[0]        

    def check_mono(self):
        if self.num_channels > 1:
            raise ChannelLayoutException()

class Wave(Signal):
    
    def __init__(self, samples, sample_rate):
        self.stream = None
        super(Wave, self).__init__(samples, sample_rate)
    
    @classmethod
    def read(cls,filename):
        sample_rate, samples = wavfile.read(filename)
        if samples.dtype==np.dtype('int16'):            
            samples = samples.astype(types.float_) / np.iinfo(np.dtype('int16')).min
        if len(samples.shape)==1:
            samples = samples.reshape((samples.shape[0],1))
        instance = cls(samples, sample_rate)
        return instance
        
    def write(self, filename):
        wavfile.write(filename, self.sample_rate, self)
        
    @classmethod
    def mix(cls, waves):
        if len(waves)==1: return waves[0].normalize()
        lengths = [np.atleast_2d(w).shape[0] for w in waves]
        widths = [np.atleast_2d(w).shape[1] for w in waves]
        if len(set(lengths)) > 1 or len(set(widths)) > 1:
            raise ArgumentException("inputs should have the same shape")
        mixed = np.zeros(waves[0].shape)
        for w in waves:
            w.normalize()
            w = np.divide(w,len(waves)) 
            mixed = mixed + w
        instance = cls(mixed, waves[0].sample_rate)
        return instance

    def normalize(self):
        return Wave(np.divide(self, np.max(self,0)), self.sample_rate)                
            
    def zero_pad(self, start_frames, end_frames=0):
        start = np.zeros((start_frames, self.num_channels),types.float_)
        end = np.zeros((end_frames,self.num_channels),types.float_)
        # avoid shape to be (N,)        
        tmp = self.reshape(self.shape[0], self.num_channels)
        return Wave(np.concatenate((start,tmp,end)), self.sample_rate)

    def plot(self):
        time_values = np.arange(self.num_frames)/float(self.sample_rate)
        if self.num_channels == 1:
            f = plt.plot(time_values, self)
        else:
            f, axes = plt.subplots(self.num_channels, sharex=True)
            for ch in range(self.num_channels):        
                axes[ch].plot(time_values, self[:,ch])
        plt.xlabel('time (s)')
        return f

    def play(self, stop_func = None):
        if self.stream is None: 
            self.stream = audio_driver.play(
                self, sr = self.sample_rate, stop_func = stop_func
            )

    def stop(self):
        audio_driver.stop(self.stream)
        self.stream = None
        
    @classmethod 
    def record(cls, max_seconds = 10, num_channels = 2, sr = 44100,
        stop_func= None):
        return audio_driver.record(max_seconds, num_channels, sr, stop_func)


"""
Audio Spectrum. Initialize with a complex spectral frame and sample rate. 
"""

class Spectrum(Signal):
    
    def magnitude(self):
       return np.abs(self) 

    def phase(self):
        return np.angle(self)
        
    def plot(self):# magnitude and phase
        f, axes = plt.subplots(2, sharex=True)
        axes[0].plot(self.magnitude())
        axes[0].plot(self.phase())
        return f
       
        
"""
Audio Spectrogram (complex). 
Rows are frequency bins (0th is the lowest frequency), columns are time bins.
"""
        
class Spectrogram(Spectrum):
    
    def __new__(cls, data, sample_rate = 44100, window_size = 1024, hop_size = 512):
        instance = Signal.__new__(cls, data,sample_rate)             
        instance.window_size = window_size
        instance.hop_size = hop_size
        return instance

    @property  
    def num_channels(self):
        return 1

    @property
    def num_frames(self):
        return self.shape[1]   
            
    def plot(self,**kwargs):
        self.maginutude_plot(**kwargs )
        
    def maginutude_plot(self, colormap="CMRmap", log_frequency = False, 
        min_freq=20, max_freq=20000):
        log_mag = 20.*np.log10(self.magnitude()+np.spacing(1))
        time_values = np.arange(self.num_frames) * float(self.hop_size)/self.sample_rate
        bin_width = self.sample_rate / (self.shape[0] *2)
        freq_values = np.arange(self.shape[0]) * bin_width        
        f = plt.figure()        
        plt.pcolormesh(time_values, freq_values, log_mag, cmap = colormap)
        plt.axis([0, time_values[-1], min_freq, max_freq])        
        if log_frequency: plt.yscale('symlog',base=2)
        plt.colorbar()        
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")    
        return f