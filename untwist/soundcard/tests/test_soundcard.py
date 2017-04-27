from ...data.audio import Wave


def test_playback():  # a bit silly, but at least touches the code

    sine1 = Wave.tone(440)
    sine1.play()
    sine1.stop()
