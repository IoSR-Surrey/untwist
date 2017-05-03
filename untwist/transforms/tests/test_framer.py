import numpy as np
from ...data import audio
from .. import stft

'''
TODO: Complete tests for first-sample centred window?
STFT test is probably sufficient.
'''


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

    for channel in signal.T:

        frames = framer.process(channel)

        assert(num_frames == frames.shape[0])

        for i, frame in enumerate(frames):

            start = i * hop_size
            end = start + window_size

            expected = channel[start:end]

            assert(np.array_equal(expected, frame))


def test_framer_spectrogram_full():
    '''
    Spectrogram, full frames only.
    '''

    window_size = 1024
    hop_size = 512

    signal = audio.Spectrogram(np.random.normal(size=(2, 44100)))
    framer = stft.Framer(window_size, hop_size, False, False)
    frames = framer.process(signal)

    num_frames = (signal.shape[1] - window_size) // hop_size + 1

    assert(num_frames == frames.shape[0])

    for i, frame in enumerate(frames):

        start = i * hop_size
        end = start + window_size

        expected = signal[:, start:end]

        assert(np.array_equal(expected, frame))
