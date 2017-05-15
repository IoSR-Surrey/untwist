from __future__ import division, print_function
import numpy as np
from ..base import defaults


def db_to_amp(db):
    return 10 ** (db / 20)


def db_to_power(db):
    return 10 ** (db / 10)


def amp_to_db(amp):
    return 20 * np.log10(amp + defaults.eps)


def power_to_db(power):
    return 10 * np.log10(power + defaults.eps)


def nearest_sample(time, sr):
    return int(np.round(time * sr))


def nearest_bin(freq, fft_size, sr):
    return int(np.round(freq * fft_size / sr))


def hz_to_cam(hz):
    return 21.366 * np.log10(4368e-6 * hz + 1.0)


def cam_to_hz(cam):
    return (10 ** (cam / 21.366) - 1) / 4368e-6


def hz_to_cambridge_erb(hz):
    return 24.673 * (4368e-6 * hz + 1)


def scale_to_hz(hz, scale):
    # More to come...
    if scale in ['hz', 'hertz']:
        return hz
    elif scale in ['cam']:
        return cam_to_hz(hz)


def hz_to_scale(hz, scale):
    # More to come...
    if scale in ['hz', 'hertz']:
        return hz
    elif scale in ['cam']:
        return hz_to_cam(hz)


def cam_scale_centre_freqs(lo_freq, hi_freq, num_filters_per_erb=1):
    '''
    Returns frequencies on the Cam scale and hertz frequency scale
    '''
    cam_lo = hz_to_cam(lo_freq)
    cam_hi = hz_to_cam(hi_freq)
    dif = np.round(cam_hi - cam_lo)
    num_channels = int(dif * num_filters_per_erb)
    cams = cam_lo + np.arange(num_channels) / num_filters_per_erb
    return cams, cam_to_hz(cams)
