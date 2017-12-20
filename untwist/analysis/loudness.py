from __future__ import division, print_function
from ..transforms import stft
from .. import filters
import numpy as np
from ..base import algorithms, defaults
from ..data import audio
from ..utilities import conversion
import collections


class EBUR128(algorithms.Processor):
    '''
    Calculates the follwing EBU R 128 loudness descriptors from an input Wave:
        - M: Momentary Loudness (time series)
        - S: Short-term loudness (time series)
        - I: Integrated loudness (time series)
        - MaxM: Maximum momentary loudness (scalar)
        - MaxS: Maximum short-term loudness (scalar)
        - P: Programme loudness (scalar). It is the final value of the
             integrated loudness.
        - LRA: Loudness range (scalar)

    The above descriptors are returned as a named tuple.

    Note that the momentary loudness is calculated from complete blocks only;
    no padding of the input Wave takes place. This is not the case for the STL
    measure: the end of the sigal is padded by about 1.5s. This is suggested in
    EBU Tech 3342, and also facilitates the measurement of relatively short
    signals.

    References:

    EBU R 128, 2014. Loudness normalisation and permitted maximum level of
    audio signals.

    EBU TECH 3341, 2016. 'EBU Mode' metering to supplement EBU R 128 loudness
    normalisation.

    EBU TECH 3342, 2016. Loudness range: A measure to supplement EBU R 128
    loudness normalisation.
    '''

    LoudnessDescriptors = collections.namedtuple('LoudnessDescriptors',
                                                 ['M',
                                                  'S',
                                                  'I',
                                                  'MaxM',
                                                  'MaxS',
                                                  'P',
                                                  'LRA']
                                                 )

    def __init__(self,
                 hop_size=0.1,
                 sample_rate=defaults.sample_rate):

        self.window_size_ML = conversion.nearest_sample(0.4, sample_rate)
        window_size_STL = conversion.nearest_sample(3, sample_rate)
        hop_size = conversion.nearest_sample(hop_size, sample_rate)
        self.rate = sample_rate / hop_size

        '''
        ITU-R BS.770 specify complete blocks only, so no padding of momentary
        loudness. However, pad the end of STL such that short signals can be
        processed. Padding is also suggested in tech 3342.
        '''

        self.framer_ML = stft.Framer(
            self.window_size_ML, hop_size, False, False)

        self.framer_STL = stft.Framer(
            window_size_STL, hop_size, False, True)

        self.gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41])

        self.offset_db = 0.691
        self.threshold = conversion.db_to_power(-70 + self.offset_db)

        self.k_filter = filters.loudness.KFilter(sample_rate)

    def process(self, wave):

        if (wave.num_frames < self.window_size_ML):
            raise ValueError('Wave duration must be > 400 ms')

        '''
        Calculate ML and STL
        '''

        # Filter signal and then square
        wave = self.k_filter.process(wave)
        wave *= wave

        num_frames_ML = self.framer_ML.calc_num_frames(wave)
        energy_ML = np.zeros(num_frames_ML)

        num_frames_STL = self.framer_STL.calc_num_frames(wave)
        energy_STL = np.zeros(num_frames_STL)

        for channel, gain in zip(wave.T, self.gains):

            energy_ML += gain * self.framer_ML.process(channel).mean(1)

            energy_STL += gain * self.framer_STL.process(channel).mean(1)

        '''
        Loudness Range (LRA)
        '''

        # Use abs thresh of -70LUFS, drop by 20LU
        mean_STL_thresh = np.mean(energy_STL[energy_STL > self.threshold])
        rel_thresh = 0.01 * mean_STL_thresh

        main_STL = energy_STL[energy_STL > rel_thresh]

        p95, p10 = np.percentile(main_STL, [95, 10])
        lra = conversion.power_to_db(p95 / p10)

        '''
        Integrated loudness
        '''

        # Sum values > -70LUFS, drop by 10LU
        energy_ML[energy_ML < self.threshold] = 0.0
        divisor = np.cumsum(energy_ML > self.threshold)
        divisor[divisor == 0] = 1

        rel_thresh = 0.1 * np.cumsum(energy_ML) / divisor

        # Limit @ -70 LUFS?
        # rel_thresh[rel_thresh < self.threshold] = self.threshold

        # Sum values > relative threshold
        energy_IL = np.zeros(num_frames_ML)

        for i in range(num_frames_ML):

            energy_so_far = energy_ML[:i+1]
            idx = energy_so_far > rel_thresh[i]

            if np.any(idx):
                energy_IL[i] = np.mean(energy_so_far[idx])

        '''
        To LUFS and configure output
        '''

        for series in [energy_ML, energy_STL, energy_IL]:
            series[series < self.threshold] = self.threshold
            series[:] = conversion.power_to_db(series) - self.offset_db

        # Container with time series as Signals
        descriptors = self.LoudnessDescriptors(energy_ML.view(audio.Signal),
                                               energy_STL.view(audio.Signal),
                                               energy_IL.view(audio.Signal),
                                               np.max(energy_ML),
                                               np.max(energy_STL),
                                               energy_IL[-1],
                                               lra)

        # Fix rate for time series descriptors
        for series in descriptors[:3]:
            series.sample_rate = self.rate

        return descriptors


class LDR(algorithms.Processor):

    LoudnessDescriptors = collections.namedtuple('LoudnessDescriptors',
                                                 ['Fast',
                                                  'Slow',
                                                  'LDR',
                                                  'ProgrammeLDR'],
                                                 )

    def __init__(self,
                 fast_dur=0.05,
                 slow_dur=3,
                 analysis_window_dur=3,
                 hop_size=0.004,
                 percentile=95,
                 sample_rate=defaults.sample_rate):

        # Force symmetric windows
        window_size_fast = conversion.nearest_sample(fast_dur, sample_rate)
        window_size_fast += (window_size_fast - 1) % 2
        window_size_slow = conversion.nearest_sample(slow_dur, sample_rate)
        window_size_slow += (window_size_slow - 1) % 2

        hop_size1 = conversion.nearest_sample(hop_size, sample_rate)
        self.rate = sample_rate / hop_size1

        # Pad start and end, windows are then centred at time zero
        self.framer_fast = stft.Framer(window_size_fast, hop_size1, True, True)
        self.framer_slow = stft.Framer(window_size_slow, hop_size1, True, True)

        # Time-varying LDR
        window_size_analysis = conversion.nearest_sample(analysis_window_dur,
                                                         self.rate)
        window_size_analysis += (window_size_analysis - 1) % 2

        self.framer_analysis = stft.Framer(window_size_analysis,
                                           1,
                                           True,
                                           True)

        # Remaining setup
        self.gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41])

        self.k_filter = filters.loudness.KFilter(sample_rate)
        self.perc = percentile

    def process(self, wave):

        # Filter signal and then square
        wave = self.k_filter.process(wave)
        wave *= wave

        # Fast and Slow calculation
        num_frames_fast = self.framer_fast.calc_num_frames(wave)
        energy_fast = np.zeros(num_frames_fast)
        energy_slow = np.zeros(num_frames_fast)

        for channel, gain in zip(wave.T, self.gains):

            energy_fast += gain * self.framer_fast.process(channel).mean(1)

            energy_slow += (gain *
                            self.framer_slow.process(
                                channel).mean(1)[:num_frames_fast]
                            )

        energy_fast = conversion.power_to_db(energy_fast)
        energy_slow = conversion.power_to_db(energy_slow)

        # Difference + percentile
        dif = audio.Signal(energy_fast - energy_slow,
                           sample_rate=self.rate)

        programme_ldr = np.percentile(dif, self.perc, axis=0)

        # Time-varying LDR
        frames = self.framer_analysis.process(dif)
        ldr = np.percentile(frames, self.perc, axis=1)

        # Container with time series as Signals
        descriptors = self.LoudnessDescriptors(energy_fast.view(audio.Signal),
                                               energy_slow.view(audio.Signal),
                                               ldr.view(audio.Signal),
                                               programme_ldr)

        # Fix rate for time series descriptors
        for series in descriptors[:3]:
            series.sample_rate = self.rate

        return descriptors
