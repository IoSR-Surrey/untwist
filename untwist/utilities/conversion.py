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
    Returns frequencies of linearly spaced components from lo_freq to hi_freq
    on the Cam scale.
    '''

    cam_lo = hz_to_cam(lo_freq)
    cam_hi = hz_to_cam(hi_freq)
    dif = np.round(cam_hi - cam_lo)
    num_channels = int(dif * num_filters_per_erb)
    cams = cam_lo + np.arange(num_channels) / num_filters_per_erb
    return cam_to_hz(cams)


def biquad_coefficients(ff_old, fb_old, fs_old, fs_new):

        if fs_new != fs_old:

            fc = ((fs_old/np.pi) *
                  np.arctan(np.sqrt((1 + fb_old[1] + fb_old[2]) /
                                    (1 - fb_old[1] + fb_old[2]))))

            q = (np.sqrt((fb_old[2] + 1) ** 2 - fb_old[1] ** 2) /
                 (2 * np.abs(1 - fb_old[2])))

            vl = ((ff_old[0] + ff_old[1] + ff_old[2]) /
                  (1 + fb_old[1] + fb_old[2]))

            vb = (ff_old[0] - ff_old[2]) / (1 - fb_old[2])

            vh = ((ff_old[0] - ff_old[1] + ff_old[2]) /
                  (1 - fb_old[1] + fb_old[2]))

            omega = np.tan(np.pi * fc / fs_new)

            omega_sqrd = omega * omega

            denom = omega_sqrd + omega / q + 1

            ff_new, fb_new = np.zeros((2, 3))

            ff_new[0] = (vl * omega_sqrd + vb * (omega / q) + vh) / denom
            ff_new[1] = 2 * (vl * omega_sqrd - vh) / denom
            ff_new[2] = (vl * omega_sqrd - (vb * omega / q) + vh) / denom

            fb_new[0] = 1.0
            fb_new[1] = 2 * (omega_sqrd - 1) / denom
            fb_new[2] = (omega_sqrd - (omega / q) + 1) / denom

            return ff_new, fb_new

        else:

            return ff_old, fb_old
