try:
    import pyaudio_driver
    audio_driver = pyaudio_driver.PyAudioDriver()
except:
    audio_driver = None
    print "Sound card not available"

__all__ = ['audio_driver']