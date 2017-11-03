from ... import data
from .. import algorithms
import numpy as np
import pytest


def test_check_mono():

    class CheckMono(algorithms.Processor):

        @algorithms.check_mono
        def process(self, signal, signal2):
            pass

    x = data.audio.Wave(np.arange(10))

    processor = CheckMono()

    processor.process(signal=x, signal2=x)

    with pytest.raises(Exception):
        processor.process(signal=x, signal2=x.reshape(2, 5))

    with pytest.raises(Exception):
        processor.process(signal=x.reshape(2, 5), signal2=x)
