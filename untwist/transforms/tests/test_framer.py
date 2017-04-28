import numpy as np
from ...data import audio
from .. import stft

'''
TODO: Complete tests for first-sample centred window?
STFT test is probably sufficient.
'''

def test_framer_wave_full():
    '''
    Mono Wave, full frames only.
    '''

    window_size = 1024
    hop_size = 512

    signal = audio.Wave.tone(freq=440, duration=1)
    framer = stft.Framer(window_size, hop_size, False, False)
    frames = framer.process(signal)

    num_frames = (signal.size - window_size) // hop_size + 1

    assert(num_frames == frames.shape[0])

    for i, frame in enumerate(frames):

        start = i * hop_size
        end = start + window_size

        expected = signal[start:end, 0]

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
        print(expected.shape, frame.shape)

        assert(np.array_equal(expected, frame))
