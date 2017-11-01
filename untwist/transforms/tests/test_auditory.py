import numpy as np
from scipy.io import loadmat
from .. import auditory
from ...data import audio


def test_gammatone():
    # Load data and flip so frequencies are low to high
    amt = loadmat('./data/test_data/mat/gammatone.mat')
    out_amt = amt['outsig'][::-1]
    centre_freqs = amt['cf'].ravel()[::-1]
    erbs = amt['erb'].ravel()[::-1]

    sample_rate = 44100
    gt = auditory.Gammatone(centre_freqs=centre_freqs,
                            erbs=erbs,
                            sample_rate=sample_rate)

    impulse = np.zeros(8193)
    impulse[0] = 1
    impulse = audio.Wave(impulse, sample_rate)

    out_untwist = gt.process(impulse)

    assert(np.allclose(out_untwist, out_amt))


def test_meddishaircell():

    amt = loadmat('./data/test_data/mat/meddishaircell.mat')
    in_amt = amt['insig']
    out_amt = amt['outsig']
    sample_rate = 44100

    in_untwist = audio.Spectrogram(in_amt, sample_rate=sample_rate)

    meddis = auditory.MeddisHairCell(sample_rate=sample_rate)
    out_untwist = meddis.process(in_untwist)

    assert(np.allclose(out_untwist, out_amt))
