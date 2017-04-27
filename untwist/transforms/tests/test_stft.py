import numpy as np
from ...data import Wave
from ...transforms import STFT, ISTFT


def test_stft():
    sine1 = Wave.tone(freq=440, duration=1)
    samples = sine1.shape[0]
    spectrogram = STFT().process(sine1)
    sine2 = ISTFT().process(spectrogram)
    assert(np.sum(np.abs(sine1 - sine2[:samples, :])) < 1e-10)
