import numpy as np
from .. import base, loudness


def test_pre():

    ff = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    fb = np.array([1.0, -1.69065929318241, 0.73248077421585])
    target = base.Filter(ff, fb, 48000)
    test_freqs = np.arange(10, 20000)
    target_mag = target.response(test_freqs).magnitude()

    for fs in [44100, 48000, 96000]:

        # pre = biquad.HighShelf(1500, 1, 4, fs)
        pre = loudness.PreFilter(fs)

        mag = pre.response(test_freqs).magnitude()

        assert(np.max(np.abs(20 * np.log10(target_mag / mag))) < 0.01)


def test_rlb():

    ff = np.array([1.0, -2.0, 1.0])
    fb = np.array([1.0, -1.99004745483398, 0.99007225036621])
    target = base.Filter(ff, fb, 48000)
    test_freqs = np.arange(10, 20000)
    target_mag = target.response(test_freqs).magnitude()

    for fs in [44100, 48000, 96000]:

        # rlb = biquad.HighPass(38, 0.5, fs)
        rlb = loudness.RLBFilter(fs)

        mag = rlb.response(test_freqs).magnitude()

        assert(np.max(np.abs(20 * np.log10(target_mag / mag))) < 0.01)


def test_k():

    sos = np.array([
        np.r_[1.53512485958697, -2.69169618940638, 1.19839281085285,
              1.0, -1.69065929318241, 0.73248077421585],
        np.r_[1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
    ])

    target = base.SOS(sos, 48000)

    test_freqs = np.arange(10, 20000)
    target_mag = target.response(test_freqs).magnitude()

    for fs in [44100, 48000, 96000]:

        k = loudness.KFilter(fs)

        mag = k.response(test_freqs).magnitude()

        assert(np.max(np.abs(20 * np.log10(target_mag / mag))) < 0.01)
