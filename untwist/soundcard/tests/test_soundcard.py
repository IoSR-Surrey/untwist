from ...data.audio import Wave
import numpy as np
import os

def test_playback():# a bit silly, but at least touches the code
    audio_dir = os.path.dirname(__file__) + "/" + ("../") * 3 + "audio/"
    sine1 = Wave.read(audio_dir + "sine440.wav")
    sine1.play()
    sine1.stop()
    
    