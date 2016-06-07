"""
Audio representations, i.e. Wave, Spectrum, Spectrogram.
Should always inherit from ndarray, but utility functions may be added, e.g. loading audio files, playing or plotting

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from ..base import types
from ..base.exceptions import *
from ..soundcard import audio_driver
from matplotlib.colors import LinearSegmentedColormap



""" utility functions"""

def ensure2D(ndarray):    
    if len(ndarray.shape)==1:
        ndarray = ndarray.reshape((ndarray.shape[0],1))    
    return ndarray

eps = np.spacing(1)



class Signal(np.ndarray):
    """
    Time domain signal. Layout is one column per channel.    
    
    Parameters
    ----------
    
    samples: ndarray
        Signal data.
    sample_rate: int
        Sample rate in samples / second.
    """    
    __array_priority__ = 10
    def __new__(cls, samples, sample_rate = 44100):
        data = ensure2D(samples)
        instance = np.ndarray.__new__(cls, 
            samples.shape, dtype = samples.dtype, strides = samples.strides, buffer = samples)
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
        """
        Number of channels
        """
        return 1 if len(self.shape)==1 else self.shape[1]

    @property
    def num_frames(self):
        """
        Number of frames (samples)
        """
        return self.shape[0]

    def check_mono(self):
        """
        Utility for ensuring the signal is mono (one channel)
        """
        if self.num_channels > 1:
            raise ChannelLayoutException()
            
    def as_ndarray(self):
        """
        Return the data as ndarray again
        """
        return np.array(self)

class Wave(Signal):
    """
    Audio waveform signal.
    
    Parameters
    ----------    
    samples: ndarray
        Signal data.
    sample_rate: int
        Sample rate in samples / second.
    """

    def __init__(self, samples, sample_rate):
        self.stream = None
        super(Wave, self).__init__(samples, sample_rate)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.stream = getattr(obj, 'stream', None)
        self.sample_rate = getattr(obj, 'sample_rate', None)
            
    @classmethod
    def read(cls, filename):
        """
        Read an audio file (only wav is supported).
        
        Parameters
        ----------
        filename: string
            Path to the wav file.        
        """
        sample_rate, samples = wavfile.read(filename)
        if samples.dtype==np.dtype('int16'):            
            samples = samples.astype(types.float_) / np.iinfo(np.dtype('int16')).min
        if len(samples.shape)==1:
            samples = samples.reshape((samples.shape[0],1))
        instance = cls(samples, sample_rate)
        return instance
        
    def write(self, filename):
        """
        Write the data to an audio file (only wav is supported).
        
        Parameters
        ----------
        filename: string
            Path to the wav file.        
        """

        wavfile.write(filename, self.sample_rate, self)
        
    def normalize(self):
        """
        Normalize by maximum amplitude.
        """
        return Wave(np.divide(self, np.max(np.abs(self), 0)), self.sample_rate)                
            
    def zero_pad(self, start_frames, end_frames = 0):
        """
        Pad with zeros at the start and/or end
        
        Parameters
        ----------
        start_frames: int
            Number of zeros at the start.
        end_frames: int
            Number of zeros at the end.

        """

        start = np.zeros((start_frames, self.num_channels),types.float_)
        end = np.zeros((end_frames,self.num_channels),types.float_)
        # avoid 1d shape
        tmp = self.reshape(self.shape[0], self.num_channels)
        return Wave(np.concatenate((start,tmp,end)), self.sample_rate)

    def plot(self):
        """
        Plot the waveofrm using matplotlib.
        """
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
        """
        Play the sound with the current audio driver.
        
        Parameters
        ----------
        stop_func: function
            Function to execute when the sound ends.
        """
        if self.stream is None: 
            self.stream = audio_driver.play(
                self, sr = self.sample_rate, stop_func = stop_func
            )

    def stop(self):
        """
        Stop playback if playing        .
        """
        audio_driver.stop(self.stream)
        self.stream = None
        
    @classmethod 
    def record(cls, max_seconds = 10, num_channels = 2, sr = 44100,
        stop_func= None):
        return audio_driver.record(max_seconds, num_channels, sr, stop_func)



class Spectrum(Signal):
    """
    Audio spectrum complex signal.
    """
        
    def __array_finalize__(self, obj):
        if obj is None: return        
        self.sample_rate = getattr(obj, 'sample_rate', None)
        self.sample_rate = getattr(obj, 'window_size', None)
        self.sample_rate = getattr(obj, 'hop_size', None)
    
    def magnitude(self):
        """
        Return the magnitude spectrum.
        """
        return np.abs(self) 

    def phase(self):
        """
        Return the phase spectrum.
        """
        return np.angle(self)
        
    def plot(self):
        """
        Plot magnitude and phase.
        """
        f, axes = plt.subplots(2, sharex=True)
        axes[0].plot(self.magnitude())
        axes[0].plot(self.phase())
        return f
       
        
        
class Spectrogram(Spectrum):
    """
    Complex audio spectrogram matrix.
    Rows are frequency bins (0th is the lowest frequency), columns are time bins.
    
    Parameters
    ----------
    samples: complex
        Spectrogram data.
    sample_rate: int
        Sample rate in samples / second of the original time domain signal.
    window_size: int
        Window size of the time-frequency transform used to obtain the spectrogram.
    hop_size: int
        Hop size of the time-frequency transform used to obtain the spectrogram.
    """
    
    def __new__(cls, samples, sample_rate = 44100, window_size = 1024, hop_size = 512):
        instance = Signal.__new__(cls, samples, sample_rate)             
        instance.window_size = window_size
        instance.hop_size = hop_size
        return instance
    
    def __array_finalize__(self, obj):
        if obj is None: return        
        self.sample_rate = getattr(obj, 'sample_rate', None)
        self.window_size = getattr(obj, 'window_size', None)
        self.hop_size = getattr(obj, 'hop_size', None)
   
    @property  
    def num_channels(self):
        return 1

    @property
    def num_frames(self):
        """
        Number of spectral frames.
        """
        return self.shape[1]
            
    def plot(self,**kwargs):        
        return self.magnitude_plot(**kwargs )
        
    def magnitude_plot(self, colormap = "CMRmap", min_freq = 0, max_freq = None, 
        axes = None, label_x = True, label_y = True, title = None, 
        colorbar = True, log_mag = True):
        """
        Plot the magnitude spectrogram
        
        Parameters
        ----------
        colormap: string
            Matplotlib colormap.
        min_freq: float
            minimum frequency in Hz (for labelling the axis).
        max_freq: float
            maximum frequency in Hz (for labelling the axis).
        axes: matplotlib axes object
            Axes object for plotting on existing figure.
        label_x: boolean
            Add labels to x axis.
        label_y: boolean
            Add labels to y axis.
        title: string
            Plot title (overlaid on image).
        colorbar: boolean
            Add a colorbar.
        log_mag: boolean
            Plot log magnitude.            
        """
        mag = self.magnitude()
        if log_mag: 
            mag = 20. * np.log10((mag / np.max(mag)) + np.spacing(1))
            min_val = -60
        else:
            min_val = 0
        if max_freq is None: max_freq = self.sample_rate / 2.0
        hop_secs = float(self.hop_size) / self.sample_rate
        time_values = np.arange(self.num_frames) * hop_secs
        bin_hz = self.sample_rate / (self.shape[0] * 2)
        freq_values = np.arange(self.shape[0]) * bin_hz
        if axes == None: axes = plt.gca()
        img = axes.imshow(mag, 
            cmap = colormap,  
            aspect="auto", 
            vmin = min_val,
            origin ="low",
            extent = [0, time_values[-1], min_freq, max_freq]
        )
        if colorbar:plt.colorbar(img, ax = axes)
        if label_x: axes.set_xlabel("time (s)")
        if label_y: axes.set_ylabel("freq (hz)")        
        plt.setp(axes.get_xticklabels(), visible = label_x)
        plt.setp(axes.get_yticklabels(), visible = label_y)
        if title is not None:
            axes.text(0.9, 0.9, title, horizontalalignment = 'right',
                bbox={'facecolor':'white', 'alpha':0.7, 'pad':5}, 
                transform=axes.transAxes)
        return axes

class TFMask(Spectrogram):
    """
    Base time-frequency mask for multiplying with spectrograms.
    """

    def plot(self, mask_color = (1, 0, 0, 0.5), min_freq = 0, max_freq = None, 
        axes = None, label_x = True, label_y = True, title = None):
        """
        Plot the time-frequency mask. 
        
        Parameters
        ----------
        mask_color: tuple
            Color specification (including alpha) for the mask
        min_freq: float
            minimum frequency in Hz (for labelling the axis).
        max_freq: float
            maximum frequency in Hz (for labelling the axis).
        axes: matplotlib axes object
            Axes object for plotting on existing figure. 
            If provided, the mask is assumed to be overlaif on a spectrogram.
        label_x: boolean
            Add labels to x axis.
        label_y: boolean
            Add labels to y axis.
        title: string
            Plot title (overlaid on image).
        """                                        
        if axes == None: 
            colormap =  LinearSegmentedColormap.from_list("map",["white","black"])
        else:
            alpha_color = [mask_color[0], mask_color[1], mask_color[2], 0]
            colormap = LinearSegmentedColormap.from_list("map", [alpha_color, mask_color])
        Spectrogram.magnitude_plot(
            self, colormap, min_freq, max_freq, axes, label_x, label_y, title, 
            False, False
            )
        
class BinaryMask(TFMask):
    """
    Binary Mask based on a comparison between target and background.
    If the threshold is 0, the mask is 1 when the target magnitude is larger 
    than the background, and 0 otherwise.
    """

    def __new__(cls, target, background, threshold = 0):
        tm = target.magnitude() + eps
        bm = background.magnitude() + eps
        mask = (20 * np.log10(tm / bm) > threshold).astype(types.float_)
        instance = TFMask.__new__(cls, mask)
        instance.sample_rate = target.sample_rate
        instance.window_size = target.window_size
        instance.hop_size = target.hop_size
        return instance
    
class RatioMask(TFMask):
    """
    Ratio Mask: soft mask based on ratio of target to background magnitude, 
    with optional exponent p.
    """
    
    def __new__(cls, target, background, p = 1):
        tm = target.magnitude() + eps
        bm = background.magnitude() + eps
        mask = (tm**p / (tm + bm)**p).astype(types.float_)
        instance = TFMask.__new__(cls, mask)
        instance.sample_rate = target.sample_rate
        instance.window_size = target.window_size
        instance.hop_size = target.hop_size
        return instance        