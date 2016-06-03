import os
import numpy as np
from ...data import Wave
from ...transforms.qerbt import QERBT

def test_qerbt():
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine1 = Wave.read(audio_dir + "sine440.wav")    
    spgm = QERBT().process(sine1).magnitude()
    print np.sum(spgm[81,:])
    print np.sum(spgm[82,:])
    print np.sum(spgm[83,:])
    print np.sum(spgm[84,:])
    print np.sum(spgm[85,:])
    assert np.sum(spgm[81,:]) < np.sum(spgm[82,:]) > np.sum(spgm[83,:])
