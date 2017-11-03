from __future__ import division, print_function
import numpy as np
from scipy import signal
from . import stft
from ..base import algorithms, defaults
from ..data import audio
from ..utilities import conversion
from . import meddis


class Gammatone(algorithms.Processor):
    '''
    Implementation of the gammatone filterbank according to:

    Slanley, M., 1993. An Efficient Implementation of the Patterson-Holdsworth
    Auditory Filter Bank. Apple Computer Technical Report #35. Perception
    Group—Advanced Technology Group. Apple Computer, Inc.
    '''

    def __init__(self,
                 lo_freq=50,
                 hi_freq=10000,
                 num_filters_per_erb=1,
                 centre_freqs=None,
                 erbs=None,
                 sample_rate=defaults.sample_rate):

        self.sample_rate = sample_rate
        dt = 1 / sample_rate

        if centre_freqs is None:
            if not hi_freq:
                hi_freq = conversion.cam_to_hz(
                    conversion.hz_to_cam(lo_freq) + 1)
            self.centre_freqs = conversion.cam_scale_centre_freqs(
                lo_freq, hi_freq, num_filters_per_erb)
        else:
            self.centre_freqs = centre_freqs


        if erbs is None:
            erbs = conversion.hz_to_cambridge_erb(self.centre_freqs)
        self.cams = conversion.hz_to_cam(self.centre_freqs)

        self.num_bands = len(self.cams)

        b_param = 1.019 * 2 * np.pi * erbs

        cf_arg = 2 * self.centre_freqs * np.pi * dt
        b_dt = b_param * dt
        e1 = -2 * np.exp(2j * cf_arg) * dt
        e2 = 2 * np.exp(-b_dt + 1j * cf_arg) * dt
        cos = np.cos(cf_arg)
        sin = np.sin(cf_arg)
        neg_sqrt = np.sqrt(3 - 2 ** 1.5)
        pos_sqrt = np.sqrt(3 + 2 ** 1.5)
        e_b_dt = np.exp(b_dt)

        self.inv_gain = 1 / np.abs(
            (e1 + e2 * (cos - neg_sqrt * sin)) *
            (e1 + e2 * (cos + neg_sqrt * sin)) *
            (e1 + e2 * (cos - pos_sqrt * sin)) *
            (e1 + e2 * (cos + pos_sqrt * sin)) /
            (-2 / np.exp(2 * b_dt) - 2 * np.exp(2j * cf_arg) +
             2 * (1 + np.exp(2j * cf_arg)) / e_b_dt) ** 4
        )

        # Feedfoward coefficents
        self.b_coefs = np.zeros((self.num_bands, 4, 2))
        self.b_coefs[:, :, 0] = dt
        self.b_coefs[:, 0, 1] = (-(2 * dt * cos / e_b_dt +
                                   2 * pos_sqrt * dt * sin / e_b_dt) / 2
                                 )
        self.b_coefs[:, 1, 1] = (-(2 * dt * cos / e_b_dt -
                                   2 * pos_sqrt * dt * sin / e_b_dt) / 2
                                 )
        self.b_coefs[:, 2, 1] = (-(2 * dt * cos / e_b_dt +
                                   2 * neg_sqrt * dt * sin / e_b_dt) / 2
                                 )
        self.b_coefs[:, 3, 1] = (-(2 * dt * cos / e_b_dt -
                                   2 * neg_sqrt * dt * sin / e_b_dt) / 2
                                 )

        # Feedback coefficents
        self.a_coefs = np.zeros((self.num_bands, 3))
        self.a_coefs[:, 0] = 1
        self.a_coefs[:, 1] = -2 * cos / e_b_dt
        self.a_coefs[:, 2] = np.exp(-2 * b_dt)

    @algorithms.check_mono
    def process(self, wave):

        if wave.sample_rate != self.sample_rate:
            raise Exception("Wrong sample rate")

        y = audio.Spectrogram(np.tile(wave, self.num_bands).T,
                              self.sample_rate,
                              1,
                              self.cams,
                              'cam')
        for chn, y_chn in enumerate(y):
            y_chn *= self.inv_gain[chn]
            for i in range(4):
                y_chn[:] = signal.lfilter(
                    self.b_coefs[chn, i], self.a_coefs[chn], y_chn)
        return y

    @algorithms.check_mono
    def process_generator(self, wave):

        for chn in range(self.num_bands):

            y = wave.ravel() * self.inv_gain[chn]

            for i in range(4):
                y = signal.lfilter(self.b_coefs[chn, i], self.a_coefs[chn], y)

            y.shape = (1, -1)
            y = y.view(audio.Spectrogram)
            y.freqs = self.cams[chn]
            y.freq_scale = 'cam'
            y.sample_rate = wave.sample_rate
            yield y


class MeddisHairCell(algorithms.Processor):
    '''
    Implementation of Meddis' inner hair cell model, following:

    Meddis, R., Hewitt, M., Shackleton, T. M., 1990. Implementation details of
    a computation model of the inner hair-cell auditory-nerve synapse. The
    Journal of the Acoustical Society of America 87(4):1813-1816.

    This processor is not strictly dependent on the configuation of the
    Spectrogram to be processed. Initialisation is thus independent of the
    input.  However, sampling frequecies < 1000Hz should not be used (refer to
    paper).

    '''

    def __init__(self,
                 medium_spont_rate=False,
                 sample_rate=defaults.sample_rate):
        self.medium_spont_rate = medium_spont_rate
        self.sample_rate = sample_rate

    def process(self, specgram):
        if len(specgram.shape) == 1:
            raise ValueError('Input Spectrogram must be 2D')

        out = meddis.process(specgram,
                             int(self.sample_rate),
                             int(self.medium_spont_rate))

        out = out.view(audio.Spectrogram)

        out.sample_rate = self.sample_rate
        out.freqs = specgram.freqs

        return out


class RatePattern(algorithms.Processor):
    '''
    Returns a cochleagram (as a Spectrogram instance) in the form of average
    firing rate over time per frequency channel.
    '''

    def __init__(self,
                 lo_freq=50,
                 hi_freq=10000,
                 num_filters_per_erb=1,
                 window_size=1024,
                 hop_size=512,
                 sample_rate=defaults.sample_rate):

        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self.gammatone = Gammatone(
            lo_freq, hi_freq, num_filters_per_erb, sample_rate=sample_rate)
        self.cams = self.gammatone.cams

        self.ihc = MeddisHairCell(False, sample_rate)
        self.framer = stft.Framer(window_size, hop_size, True, True, False)

    @algorithms.check_mono
    def process(self, wave):

        shape = (len(self.cams), self.framer.calc_num_frames(wave))

        frames = audio.Spectrogram(np.empty(shape),
                                   self.sample_rate,
                                   self.hop_size,
                                   self.cams,
                                   'cam')

        for chn, y_chn in enumerate(self.gammatone.process_generator(wave)):
            y_chn[:] = self.ihc.process(y_chn)
            frames[chn] = self.framer.process(y_chn).mean(axis=-1).T
        return frames
