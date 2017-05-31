from __future__ import division, print_function
from ..transforms import stft
from .. import filters
import numpy as np
from ..base import algorithms, defaults
from ..data import audio
from ..utilities import conversion
import collections


class EBUR128Loudness(algorithms.Processor):

    LoudnessDescriptors = collections.namedtuple('LoudnessDescriptors',
                                                 ['M', 'S', 'I'],
                                                 )

    def __init__(self,
                 hop_size=0.1,
                 sample_rate=defaults.sample_rate,
                 complete_blocks_only=True):

        window_size_ML = conversion.nearest_sample(0.4, sample_rate)
        self.window_size_STL = conversion.nearest_sample(3, sample_rate)
        hop_size = conversion.nearest_sample(hop_size, sample_rate)
        self.rate = sample_rate / hop_size
        self.complete_blocks_only = complete_blocks_only

        '''
        ITU-R BS.770 specify complete blocks only, so we limit input signals to
        a minimum of 3 s, so we limit input signals to a minimum of 3 s.
        '''

        if self.complete_blocks_only:
            pad_start_end = False
        else:
            pad_start_end = True

        self.framer_ML = stft.Framer(
            window_size_ML, hop_size, pad_start_end, pad_start_end)

        self.framer_STL = stft.Framer(
            self.window_size_STL, hop_size, pad_start_end, pad_start_end)

        self.gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41])

        self.offset_db = 0.691
        self.threshold = conversion.db_to_power(-70 + self.offset_db)

        self.k_filter = filters.loudness.K(sample_rate)

    def process(self, wave):

        if (self.complete_blocks_only and
                wave.num_frames < self.window_size_STL):

            raise ValueError('wave duration must be > 3 s')

        # Filter signal and then square
        wave = self.k_filter.process(wave)
        wave *= wave

        num_frames_ML = self.framer_ML.calc_num_frames(wave.left)
        energy_ML = np.zeros(num_frames_ML)

        num_frames_STL = self.framer_STL.calc_num_frames(wave.left)
        energy_STL = np.zeros(num_frames_STL)

        for channel, gain in zip(wave.T, self.gains):

            energy_ML += gain * self.framer_ML.process(channel).mean(1)

            energy_STL += gain * self.framer_STL.process(channel).mean(1)

        # Sum values > -70LUFS, drop by -10LU
        energy_ML[energy_ML < self.threshold] = 0.0
        divisor = np.cumsum(energy_ML > self.threshold)
        divisor[divisor == 0] = 1

        rel_thresh = 0.1 * np.cumsum(energy_ML) / divisor

        # Limit @ -70 LUFS?
        rel_thresh[rel_thresh < self.threshold] = self.threshold

        # Sum values > relative threshold
        energy_IL = np.zeros(num_frames_ML)

        for i in range(num_frames_ML):

            energy_so_far = energy_ML[:i+1]
            idx = energy_so_far > rel_thresh[i]

            if np.any(idx):
                energy_IL[i] = np.mean(energy_so_far[idx])

        # To LUFS
        for series in [energy_ML, energy_STL, energy_IL]:
            series[series < self.threshold] = self.threshold
            series[:] = conversion.power_to_db(series) - self.offset_db

        # As Signals
        descriptors = self.LoudnessDescriptors(energy_ML.view(audio.Signal),
                                               energy_STL.view(audio.Signal),
                                               energy_IL.view(audio.Signal))
        # Fix rate
        for series in descriptors:
            series.sample_rate = self.rate

        return descriptors
