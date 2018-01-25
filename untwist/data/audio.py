"""

Audio representations, i.e. Wave, Spectrum, Spectrogram.  Should always inherit
from ndarray, but utility functions may be added, e.g. loading audio files,
playing or plotting

TODO:
    - Signal.__reduce__, new_state undefined ?
"""
from __future__ import division, print_function
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..utilities import conversion
from ..base import types as _types
from ..base import defaults
from ..soundcard import audio_driver


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

    def __new__(cls, samples, sample_rate=defaults.sample_rate):

        if len(samples.shape) == 1:
            samples.shape = (samples.shape[0], 1)
        instance = np.asfortranarray(samples).view(cls)  # Copies samples
        instance.sample_rate = sample_rate
        return instance

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.sample_rate = getattr(obj, 'sample_rate', defaults.sample_rate)

    def __array_prepare__(self, out_arr, context=None):
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __reduce__(self):  # pickle additional attributes
        pickled_state = super(Signal, self).__reduce__()
        new_state = pickled_state[2] + (self.sample_rate,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.sample_rate = state[-1]
        super(Signal, self).__setstate__(state[0:-1])

    @property
    def num_channels(self):
        """
        Number of channels
        """
        return self.shape[1]

    @property
    def num_frames(self):
        """
        Number of frames (samples)
        """
        return self.shape[0]

    @property
    def duration(self):
        return float(self.num_frames) / self.sample_rate

    @property
    def time(self):
        return np.arange(self.num_frames) / self.sample_rate

    @property
    def left(self):
        return self[..., 0][..., np.newaxis].copy('K')

    @property
    def right(self):
        if self.num_channels > 1:
            return self[..., -1][..., np.newaxis].copy('K')
        else:
            raise AttributeError('Signal only has one channel')

    def is_mono(self):
        return self.num_channels == 1

    def is_stereo(self):
        return self.num_channels == 2

    def to_mono(self):
        return self.mean(-1)[..., np.newaxis]

    def to_stereo(self):
        if self.num_channels == 1:
            return np.tile(self, 2)
        else:
            return self[..., :2].copy('F')

    def zero_pad(self, start_frames=0, end_frames=0):
        """
        Pad with zeros at the start and/or end

        Parameters
        ----------
        start_frames: int
            Number of zeros at the start.
        end_frames: int
            Number of zeros at the end.

        """

        shape = list(self.shape)
        shape[0] += start_frames + end_frames
        data = np.zeros(shape, order='F', dtype=self.dtype)
        data[start_frames:start_frames + self.num_frames] = self

        return self.__class__(data,
                              self.sample_rate)

    def as_ndarray(self):
        """
        Return the data as ndarray again
        """
        return np.array(self)

    def plot(self,
             axes=None,
             xlabel='Time (s)',
             ylabel='Amplitude',
             color=None):
        """
        Plot the signal using matplotlib.
        """

        if axes is None:
            axes = plt.gca()

        axes.plot(self.time, self, color=color)

        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)

        return axes


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

    def __init__(self, samples, sample_rate=defaults.sample_rate):
        self.stream = None
        super(Wave, self).__new__(Wave, samples, sample_rate)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stream = getattr(obj, 'stream', None)
        self.sample_rate = getattr(obj, 'sample_rate', None)

    @classmethod
    def tone(cls, freq=1000, phase=0,
             duration=1, sample_rate=defaults.sample_rate):
        num_samples = conversion.nearest_sample(duration, sample_rate)
        t = np.arange(num_samples) / sample_rate
        return cls(np.sin(2 * np.pi * t * freq + phase), sample_rate)

    @classmethod
    def read(cls, filename):
        """
        Read an audio file using PySoundfile.

        Parameters
        ----------
        filename: string
            Path to the audio file.
        """
        samples, sample_rate = sf.read(filename)

        if len(samples.shape) == 1:
            samples = samples.reshape((-1, 1))

        instance = cls(samples, sample_rate)

        return instance

    def write(self, filename, desired_dtype='DOUBLE'):
        """
        Write the data to an audio file using PySoundfile.

        Parameters
        ----------
        filename: string
            Path to the audio file.
        desired_dtype: data type
            Desired data type, e.g. 'PCM_16'
        """

        if np.float == self.dtype:

            if np.max(self) >= 1 or np.min(self) < -1:
                print("Warning: Signal amplitude exceeds the interval [-1, 1)")

        sf.write(filename, self, self.sample_rate, desired_dtype)

    def with_duration(self, duration):

        frames = conversion.nearest_sample(duration,
                                           self.sample_rate)

        if self.num_frames < frames:
            return Wave(self.zero_pad(0, frames - self.num_frames),
                        self.sample_rate)

        elif self.num_frames > frames:
            return Wave(self[:frames], self.sample_rate)

        else:
            return Wave(self, self.sample_rate)

    def append(self, other):

        return Wave(np.r_[self, other], self.sample_rate)

    def __add__(self, other):

        if isinstance(other, self.__class__):

            max_frames = np.maximum(self.num_frames, other.num_frames)
            max_channels = np.maximum(self.num_channels, other.num_channels)

            result = Wave(np.zeros((max_frames, max_channels)),
                          self.sample_rate)

            result[:self.num_frames, :self.num_channels] = (
                self.reshape(self.num_frames, self.num_channels)
            )

            result[:other.num_frames, :other.num_channels] += (
                other.reshape(other.num_frames, other.num_channels)
            )

        else:

            result = Wave(np.add(self, other), self.sample_rate)

        return result

    @property
    def level(self):
        return conversion.power_to_db(
            float(np.mean(self.flatten() ** 2))
        )

    @level.setter
    def level(self, target_level):
        gain = conversion.db_to_amp(target_level - self.level)
        self *= gain

    @property
    def peak_level(self):
        return conversion.amp_to_db(
            float(np.max(np.abs(self)))
        )

    @peak_level.setter
    def peak_level(self, target_level):
        gain = conversion.db_to_amp(target_level - self.peak_level)
        self *= gain

    @property
    def loudness(self):
        from ..analysis import loudness
        ebur128 = loudness.EBUR128(sample_rate=self.sample_rate)
        return ebur128.process(self).P

    @loudness.setter
    def loudness(self, target_loudness):
        gain = conversion.db_to_amp(target_loudness - self.loudness)
        self *= gain

    def normalize(self):
        """
        Normalise by maximum amplitude.
        """
        return Wave(np.divide(self, np.max(np.abs(self), 0)), self.sample_rate)

    def play(self, stop_func=None):
        """
        Play the sound with the current audio driver.

        Parameters
        ----------
        stop_func: function
            Function to execute when the sound ends.
        """
        if audio_driver is not None and self.stream is None:
            self.stream = audio_driver.play(
                self, sr=self.sample_rate, stop_func=stop_func
            )

    def stop(self):
        """
        Stop playback if playing.
        """
        if audio_driver is not None:
            audio_driver.stop(self.stream)
            self.stream = None

    @classmethod
    def record(cls, max_seconds=10, num_channels=2,
               sr=defaults.sample_rate, stop_func=None):
        return audio_driver.record(max_seconds, num_channels, sr, stop_func)


class Spectrum(Signal):
    """
    Audio spectrum
    """

    def __new__(cls, samples, sample_rate, freqs=None):

        instance = Signal.__new__(cls, samples, sample_rate)
        instance.freqs = freqs

        return instance

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.freqs = getattr(obj, 'freqs', None)

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

    def plot_magnitude(self, log_mag=True, log_x=True):

        mag = self.magnitude()

        y_label = 'Magnitude'
        if log_mag:
            mag = conversion.amp_to_db(mag)
            y_label += ', dB'

        axes = plt.gca()

        if log_x:
            axes.semilogx(self.freqs, mag)
        else:
            axes.plot(self.freqs, mag)

        axes.set_xlabel('Frequency, Hz')
        axes.set_ylabel(y_label)

        return axes


class Spectrogram(Signal):
    """

    This class represents a general time-frequency representation of a signal.
    It is not necessarily tied to STFT, e.g. the output of the Gammatone
    processor is a Spectrogam instance. It is basically a matrix of an abitrary
    data type, where rows represent frequency and columns represent time.

    Parameters
    ----------
    samples: float, complex
    sample_rate: int
        Sample rate in samples / second of the original time domain signal.
    hop_size: int
         Hop size of the time-frequency transform (default is 1).
    freqs: float
        Centre frequencies of the filters used for the time-frequency
        transform. If None, frequencies are assumed to be linearly spaced and
        are determined from the shape of the input array.
    """

    def __new__(cls, samples, sample_rate=defaults.sample_rate,
                hop_size=1, freqs=None, freq_scale=defaults.freq_scale):

        # Ensure spectrogram is 3D
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim == 2:
            samples = np.expand_dims(samples, -1)

        instance = Signal.__new__(cls, samples, sample_rate)

        instance.hop_size = hop_size
        instance.freq_scale = freq_scale

        if freqs is None:
            spacing = sample_rate / 2.0 ** np.ceil(np.log2(samples.shape[0]))
            instance.freqs = np.arange(samples.shape[0]) * spacing
        else:
            instance.freqs = freqs

        return instance

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.sample_rate = getattr(obj, 'sample_rate', defaults.sample_rate)
        self.hop_size = getattr(obj, 'hop_size', None)
        self.freqs = getattr(obj, 'freqs', None)
        self.freq_scale = getattr(obj, 'freq_scale', defaults.freq_scale)

    @property
    def num_bands(self):
        return self.shape[0]

    @property
    def num_frames(self):
        return self.shape[1]

    @property
    def num_channels(self):
        return self.shape[2]

    @property
    def duration(self):
        return float(self.num_frames * self.hop_size) / self.sample_rate

    @property
    def time(self):
        hop_secs = float(self.hop_size) / self.sample_rate
        return np.arange(self.num_frames) * hop_secs

    def magnitude(self):
        return np.abs(self)

    def phase(self):
        return np.angle(self)

    def zero_pad(self, start_frames, end_frames=0):
        """
        Pad with zeros at the start and/or end

        Parameters
        ----------
        start_frames: int
            Number of zeros at the start.
        end_frames: int
            Number of zeros at the end.

        """

        shape = list(self.shape)
        shape[1] += start_frames + end_frames
        data = np.zeros(shape, order='F', dtype=self.dtype)
        data[:, start_frames:start_frames + self.num_frames] = self

        return Spectrogram(data,
                           self.sample_rate,
                           self.hop_size,
                           self.freqs,
                           self.freq_scale)

    def plot(self, **kwargs):
        return self.plot_magnitude(**kwargs)

    def plot_magnitude(self,
                       colormap="CMRmap",
                       min_time=None,
                       max_time=None,
                       min_freq=None,
                       max_freq=None,
                       axes=None,
                       xlabel="Time (s)",
                       ylabel=None,
                       colorbar=True,
                       title=None,
                       log_mag=True,
                       log_yscale=False):
        """
        Plot the magnitude spectrogram

        Parameters
        ----------
        colormap: string
            Matplotlib colormap.
        axes: matplotlib axes object
            Axes object for plotting on existing figure.
        min_time: float
            minimum time point in seconds
        max_time: float
            maximum time point in seconds
        min_freq: float
            minimum frequency
        max_freq: float
            maximum frequency
        xlabel: boolean
            Add labels to x axis.
        ylabel: boolean
            Add labels to y axis.
        colorbar: boolean
            Add a colorbar.
        title: string
            Plot title (overlaid on image).
        log_mag: boolean
            Plot log magnitude.
        log_yscale: boolean
            Plot on log-y axis.
        """

        # Average over channels for now
        mag = self.magnitude().mean(axis=-1)

        if log_mag:
            mag = 20. * np.log10((mag / np.max(mag)) + np.spacing(1))
            min_val = -80
        else:
            min_val = 0

        if axes is None:
            axes = plt.gca()

        img = axes.imshow(
            mag,
            cmap=colormap,
            aspect="auto",
            vmin=min_val,
            origin="low",
            interpolation='bilinear',
            extent=[self.time[0], self.time[-1],
                    self.freqs[0], self.freqs[-1]],
        )

        if colorbar:
            plt.colorbar(img, ax=axes)
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)
        elif ylabel is None:
            axes.set_ylabel(self.freq_scale)

        if log_yscale:
            axes.set_yscale('symlog')

        axes.set_xlim((min_time, max_time))
        axes.set_ylim((min_freq, max_freq))

        if title is not None:
            axes.text(0.8, 0.8, title, horizontalalignment='center',
                      bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5},
                      transform=axes.transAxes)

        return axes


class TFMask(Spectrogram):
    """
    Base time-frequency mask for multiplying with spectrograms.
    """

    def plot(self,
             mask_color=(1, 0, 0, 0.5),
             axes=None,
             **kwargs):
        """
        Plot the time-frequency mask.

        Parameters
        ----------
        mask_color: tuple
            Color specification (including alpha) for the mask
        axes: matplotlib axes object
            Axes object for plotting on existing figure.
        kwargs: key, value mappings
            keyword arguments are passed through to Spectrogram.plot_magnitude
        """
        if axes is None:
            colormap = LinearSegmentedColormap.from_list(
                "map", ["white", "black"])
        else:
            alpha_color = [mask_color[0], mask_color[1], mask_color[2], 0]
            colormap = LinearSegmentedColormap.from_list(
                "map", [alpha_color, mask_color])
        Spectrogram.plot_magnitude(self, colormap=colormap, **kwargs)


class BinaryMask(TFMask):
    """
    Binary Mask based on a comparison between target and background.
    If the threshold is 0, the mask is 1 when the target magnitude is larger
    than the background, and 0 otherwise.
    """

    def __new__(cls, target, background, threshold=0):
        tm = target.magnitude() + defaults.eps
        bm = background.magnitude() + defaults.eps
        mask = (20 * np.log10(tm / bm) > threshold).astype(_types.float_)
        instance = TFMask.__new__(cls, mask)
        instance.sample_rate = target.sample_rate
        instance.hop_size = target.hop_size
        instance.freqs = target.freqs
        instance.scale = target.freq_scale
        return instance


class RatioMask(TFMask):
    """
    Ratio Mask: soft mask based on ratio of target to background magnitude,
    with optional exponent p.
    """

    def __new__(cls, target, background, p=1):
        tm = target.magnitude() + defaults.eps
        bm = background.magnitude() + defaults.eps
        mask = (tm**p / (tm + bm)**p).astype(_types.float_)
        instance = TFMask.__new__(cls, mask)
        instance.sample_rate = target.sample_rate
        instance.hop_size = target.hop_size
        instance.freqs = target.freqs
        instance.scale = target.freq_scale
        return instance


class ComplexRatioMask(TFMask):
    """
    Complex Ratio Mask: mask based on the ratio of the (complex) target and
    mixture.
    Williamson, D. Wang, Y., and Wang, D. Complex Ratio Masking for Monaural
    Speech Seperation. IEEE/ACM transactions on audio, speech, and language
    processing. Vol 24, No. 3. 2016.
    """

    def __new__(cls, target, background):
        eps = complex(defaults.eps, defaults.eps)
        mask = ((target + eps) /
                (target + background + eps))
        instance = TFMask.__new__(cls, mask)
        instance.sample_rate = target.sample_rate
        instance.hop_size = target.hop_size
        instance.freqs = target.freqs
        instance.scale = target.freq_scale
        return instance

    def compress(self, k=10, c=0.1):
        exp = np.exp(-c * self)
        return k * (1 - exp) / (1 + exp)

    def uncompress(self, k=10, c=0.1):
        return -1/c * np.log((k - self) / (k + self))
