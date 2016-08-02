import os
import numpy as np
from ...data import Wave
from ...transforms.qerbt import QERBT, QERBFilter

def test_qerbt():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine1 = Wave.read(audio_dir + "sine440.wav")    
    spgm = QERBT().process(sine1).magnitude()
    assert np.sum(spgm[81,:]) < np.sum(spgm[82,:]) > np.sum(spgm[83,:])


def test_qerbtf():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine1 = Wave.read(audio_dir + "sine440.wav")
    spgm = QERBT().process(sine1).magnitude()
    filter = np.ones_like(spgm)
    filter[70:95,:] = 0
    qerbf = QERBFilter()
    result = qerbf.process(sine1, filter)
    assert( (np.sum(result**2) / np.sum(sine1**2)) < 0.01)