from __future__ import division
import numpy as np
from ...data import audio
from .. import stft

'''
TODO: Complete tests for first-sample centred window?
STFT test is probably sufficient.
'''


def test_framer_pad_start():
    '''
    Stereo Wave, start padded such that window is centred on first sample.
    '''

    window_size = 1025
    hop_size = 512

    samples = np.random.normal(size=(44100, 2))
    signal = audio.Wave(samples)
    signal2 = signal.zero_pad(window_size//2)

    framer = stft.Framer(window_size, hop_size, True, False)

    num_frames = int((signal.num_frames + window_size//2 - window_size) //
                     hop_size) + 1

    frames = framer.process(signal)

    assert(num_frames == frames.shape[1])

    for i in range(num_frames):

        start = i * hop_size
        end = start + window_size

        expected = signal2[start:end]

        assert(np.array_equal(expected, frames[:, i]))


def test_framer_pad_end():
    '''
    Stereo Wave, end padded such that start of window passes entire signal.
    '''

    window_size = 1025
    hop_size = 512

    samples = np.random.normal(size=(44100, 2))
    signal = audio.Wave(samples)
    signal2 = signal.zero_pad(0, window_size)

    framer = stft.Framer(window_size, hop_size, False, True)

    num_frames = int(np.ceil(signal.num_frames / hop_size))

    frames = framer.process(signal)

    assert(num_frames == frames.shape[1])

    for i in range(num_frames):

        start = i * hop_size
        end = start + window_size

        expected = signal2[start:end]

        assert(np.array_equal(expected, frames[:, i]))


def test_framer_wave_full():
    '''
    Stereo Wave, full frames only.
    '''

    window_size = 1024
    hop_size = 512

    samples = np.random.normal(size=(44100, 2))
    signal = audio.Wave(samples)

    framer = stft.Framer(window_size, hop_size, False, False)

    num_frames = (signal.num_frames - window_size) // hop_size + 1

    frames = framer.process(signal)

    assert(num_frames == frames.shape[1])

    for i in range(num_frames):

        start = i * hop_size
        end = start + window_size

        expected = signal[start:end]

        assert(np.array_equal(expected, frames[:, i]))


def test_framer_spectrogram_full():
    '''
    Stereo spectrogram, full frames only.
    '''

    window_size = 1024
    hop_size = 512

    signal = audio.Spectrogram(np.random.normal(size=(513, 44100, 2)))
    framer = stft.Framer(window_size, hop_size, False, False)
    frames = framer.process(signal)

    num_frames = (signal.shape[1] - window_size) // hop_size + 1

    assert(num_frames == frames.shape[2])

    for i in range(num_frames):

        start = i * hop_size
        end = start + window_size

        expected = signal[:, start:end]

        assert(np.array_equal(expected, frames[:, :, i]))
