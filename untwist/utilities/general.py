from __future__ import division, print_function
import wave


def get_duration(wav_file):
    with wave.open(wav_file, 'rb') as f:
        return f.getnframes() / f.getframerate()
