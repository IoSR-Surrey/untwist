from ...data import audio
from .. import loudness
import numpy as np
import os


data_path = os.path.abspath(os.path.join(__file__,
                                         '../../../../data/test_data/ebu_r128')
                            )


def all_within(a, b, within=0.1):
    return np.all(np.abs(a - b) < within)


def test_ebu128loudness_case1():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-1-16bit.wav')
    )

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    for s in series:
        assert(all_within(s, expected, tol))


def test_ebu128loudness_case2():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-2-16bit.wav')
    )

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -33.0
    tol = 0.1

    for s in series:
        assert(all_within(s, expected, tol))


def test_ebu128loudness_case3():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-3-16bit-v02.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.I[-1], expected, tol))


def test_ebu128loudness_case4():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-4-16bit-v02.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.I[-1], expected, tol))


def test_ebu128loudness_case5():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-5-16bit-v02.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.I[-1], expected, tol))


def test_ebu128loudness_case6():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-6-5channels-16bit.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.I, expected, tol))


def test_ebu128loudness_case7():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-7_seq-3342-5-16bit.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.I[-1], expected, tol))


def test_ebu128loudness_case9():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-9-16bit.wav'))

    processor = loudness.EBUR128Loudness(sample_rate=wave.sample_rate)
    series = processor.process(wave)

    expected = -23.0
    tol = 0.1

    assert(all_within(series.S, expected, tol))
