from ...data import audio
from .. import loudness
from ...utilities import conversion
import numpy as np
import os

'''
Here, the Max STL is used for STL because of decay necessary for LRA
computation (and ease of use).
'''

data_path = os.path.abspath(os.path.join(__file__,
                                         '../../../../data/test_data/ebu_r128')
                            )


def all_within(a, b, within=0.1):
    return np.all(np.abs(a - b) < within)


def test_ebu128_tech3341_case1():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-1-16bit.wav')
    )

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))
    assert(all_within(out.M, expected, tol))
    assert(all_within(out.MaxS, expected, tol))


def test_ebu128_tech3341_case2():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-2-16bit.wav')
    )

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -33
    tol = 0.1

    assert(all_within(out.P, expected, tol))
    assert(all_within(out.M, expected, tol))
    assert(all_within(out.MaxS, expected, tol))


def test_ebu128_tech3341_case3():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-3-16bit-v02.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))


def test_ebu128_tech3341_case4():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-4-16bit-v02.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))


def test_ebu128_tech3341_case5():

    wave = audio.Wave.read(os.path.join(data_path, 'seq-3341-5-16bit-v02.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))


def test_ebu128_tech3341_case6():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-6-5channels-16bit.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.I, expected, tol))


def test_ebu128_tech3341_case7():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-7_seq-3342-5-16bit.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))


def test_ebu128_tech3341_case8():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-2011-8_seq-3342-6-16bit-v02.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.P, expected, tol))


def test_ebu128_tech3341_case9():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-9-16bit.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = -23
    tol = 0.1

    assert(all_within(out.MaxS, expected, tol))


'''
Loudness range
'''


def test_ebu128_tech3342_case1():

    wave = audio.Wave.tone(1000, duration=40, sample_rate=48000).to_stereo()
    wave.peak_level = -20
    drop_after_sample = conversion.nearest_sample(20, wave.sample_rate)
    wave[drop_after_sample:] *= conversion.db_to_amp(-10)

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 10
    tol = 1

    assert(all_within(out.LRA, expected, tol))


def test_ebu128_tech3342_case2():

    wave = audio.Wave.tone(1000, duration=40, sample_rate=48000).to_stereo()
    wave.peak_level = -20
    drop_after_sample = conversion.nearest_sample(20, wave.sample_rate)
    wave[drop_after_sample:] *= conversion.db_to_amp(-5)

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 5
    tol = 1

    assert(all_within(out.LRA, expected, tol))


def test_ebu128_tech3342_case3():

    wave = audio.Wave.tone(1000, duration=40, sample_rate=48000).to_stereo()
    wave.peak_level = -40
    drop_after_sample = conversion.nearest_sample(20, wave.sample_rate)
    wave[drop_after_sample:] *= conversion.db_to_amp(20)

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 20
    tol = 1

    assert(all_within(out.LRA, expected, tol))


def test_ebu128_tech3342_case4():

    wave = audio.Wave.tone(1000, duration=100, sample_rate=48000).to_stereo()
    wave.peak_level = -50

    for gain, time in zip([15, 15, -15, -15], [20, 40, 60, 80]):

        drop_after_sample = conversion.nearest_sample(time, wave.sample_rate)
        wave[drop_after_sample:] *= conversion.db_to_amp(gain)

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 15
    tol = 1

    assert(all_within(out.LRA, expected, tol))


def test_ebu128_tech3342_case5():

    wave = audio.Wave.read(os.path.join(data_path,
                                        'seq-3341-7_seq-3342-5-16bit.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 5
    tol = 1

    assert(all_within(out.LRA, expected, tol))


def test_ebu128_tech3342_case6():

    wave = audio.Wave.read(os.path.join(
        data_path, 'seq-3341-2011-8_seq-3342-6-16bit-v02.wav'))

    processor = loudness.EBUR128(sample_rate=wave.sample_rate)
    out = processor.process(wave)

    expected = 15
    tol = 1

    assert(all_within(out.LRA, expected, tol))
