import numpy as np
from ...data import Wave
from ...transforms import STFT, ISTFT


def test_stft():
    signal = Wave(np.random.normal(size=(44100, 2)))
    samples = signal.num_frames
    spectrogram = STFT().process(signal)
    signal2 = ISTFT().process(spectrogram)
    assert(np.sum(np.abs(signal - signal2[:samples, :])) < 1e-10)
