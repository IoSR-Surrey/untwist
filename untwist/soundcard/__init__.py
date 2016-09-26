from __future__ import print_function
try:
    from .pyaudio_driver import PyAudioDriver    
    audio_driver = PyAudioDriver()
except:
    audio_driver = None
    print("Sound card not available")

__all__ = ['audio_driver']
