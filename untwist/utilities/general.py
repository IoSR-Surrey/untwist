from __future__ import division, print_function
import wave
import shutil
import tempfile


class TemporaryDirectory(object):
    """
    Context manager for tempfile.mkdtemp().
    This class is available in python +v3.2.
    From: https://gist.github.com/cpelley/10e2eeaf60dacc7956bb
    """
    def __enter__(self):
        self.dir_name = tempfile.mkdtemp()
        return self.dir_name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.dir_name)


def get_duration(wav_file):
    with wave.open(wav_file, 'rb') as f:
        return f.getnframes() / f.getframerate()
