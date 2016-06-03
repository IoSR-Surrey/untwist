""" Tests for the audio module"""
import os
import numpy as np
from ...data.audio import Wave, BinaryMask, RatioMask
from ...transforms.stft import STFT

def test_wave_io():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    fname = audio_dir+"noise.wav"    
    w1 = Wave(np.random.normal(0, 1, 44100), 44100)
    assert(len(w1.shape) == 2)    
    w1.write(fname)
    w2 = Wave.read(fname)
    assert(w2.sample_rate == 44100)
    assert(w1.shape == w2.shape)
    assert(np.sum(w1 - w2) == 0)
    os.remove(fname)
    
def test_normalize():
    w1 = Wave(np.random.normal(0, 0.5, 44100), 44100)
    w2 = w1.normalize()
    print np.max(np.abs(w2))
    assert(np.max(np.abs(w2)) > 0.99)

def test_masks():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine = Wave.read(audio_dir + "sine440.wav")
    silence = Wave(np.zeros(sine.shape), sine.sample_rate)
    spectrogram = STFT().process(sine)
    SINE = STFT().process(sine)
    SILENCE = STFT().process(silence)
    bm = BinaryMask(SINE, SILENCE)[:,:-1]
    assert(np.sum(bm) == bm.shape[0] * bm.shape[1])
    rm = BinaryMask(SINE, SILENCE)[:,:-1]
    assert(np.sum(rm) == rm.shape[0] * rm.shape[1])
    
    