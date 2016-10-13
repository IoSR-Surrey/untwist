import os
import numpy as np
from ...data import Wave
from ...transforms import STFT, ISTFT

def test_stft():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine1 = Wave.read(audio_dir + "sine440.wav")
    samples = sine1.shape[0]
    spectrogram = STFT().process(sine1)
    sine2 = ISTFT().process(spectrogram)
    assert(np.sum(np.abs(sine1 - sine2[:samples,:])) < 1e-10)

