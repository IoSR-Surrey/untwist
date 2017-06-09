""" Tests for the audio module"""
from __future__ import print_function
import os
import numpy as np
from ...data.audio import Wave, BinaryMask, RatioMask, ComplexRatioMask
from ...transforms.stft import STFT
from ...analysis import loudness
import tempfile


def test_wave_io():
    with tempfile.TemporaryDirectory() as tmp_dir:
        print('created temporary directory', tmp_dir)
        fname = os.path.join(tmp_dir, "noise.wav")
        w1 = Wave(np.random.normal(0, 1, 44100), 44100)
        assert(len(w1.shape) == 2)
        w1.write(fname)
        w2 = Wave.read(fname)
        assert(w2.sample_rate == 44100)
        assert(w1.shape == w2.shape)
        assert(np.sum(w1 - w2) == 0)


def test_wave_add():

    w1 = Wave(np.ones(10))
    w2 = Wave(np.ones(20) + 1).as_stereo()

    w3 = w1 + w2

    expected = np.ones((20, 2)) + 1
    expected[:10, 0] += 1

    assert(np.all(w3 == expected))


def test_normalize():
    w1 = Wave(np.random.normal(0, 0.5, 44100), 44100)
    w2 = w1.normalize()
    assert(np.max(np.abs(w2)) > 0.99)


def test_loudness():

    ebur128 = loudness.EBUR128(sample_rate=44100)

    w1 = Wave(np.random.normal(0, 0.5, 44100), 44100)

    target = -23.0
    w1.loudness = target

    measured_loudness = ebur128.process(w1).P

    assert(np.round(measured_loudness, 1) ==
           np.round(w1.loudness, 1) ==
           target)


def test_masks():
    sine = Wave.tone(freq=440, duration=1)
    silence = Wave(np.zeros(sine.shape), sine.sample_rate)
    SINE = STFT().process(sine)
    SILENCE = STFT().process(silence)
    bm = BinaryMask(SINE, SILENCE)[:, :-1]
    assert(np.sum(bm) == bm.shape[0] * bm.shape[1])
    rm = RatioMask(SINE, SILENCE)[:, :-1]
    assert(np.round(np.sum(rm)) == rm.shape[0] * rm.shape[1])
    crm = ComplexRatioMask(SINE, SILENCE)[:, :-1]
    assert(np.round(np.sum(crm)) == crm.shape[0] * crm.shape[1])
