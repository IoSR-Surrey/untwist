"""
Onset detection algorithms
"""
import numpy as np
from scipy import signal, interpolate
from scipy.stats import zscore
from untwist.base.algorithms import Processor

class OnsetDetector(Processor):
    """
    Onset detector based on comon time-frequency detection functions.
    Based on: Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., 
    Davies, M., & Sandler, M. B. (2005). 
    A tutorial on onset detection in music signals. 
    IEEE Transactions on speech and audio processing, 13(5), 1035-1047.
    """
    
    def __init__(self, func = "hfc", threshold = -0.5, 
        moving_size = 4, median_size = 401):
        self.func = getattr(self, func)
        self.median_kernel = (1, median_size)
        self.moving_avg_filter = np.ones(moving_size) / moving_size
        self.threshold = threshold
        
    def process(self, X):
        onset_func = zscore(self.func(X))
        onset_func = signal.filtfilt(self.moving_avg_filter, 1, onset_func)
        onset_func = onset_func - signal.medfilt(
            onset_func[:,np.newaxis], self.median_kernel
        )[:,0]
        peaks = signal.argrelmax(onset_func)
        onsets = peaks[0][np.where(onset_func[peaks[0]] >
            self.threshold)]
        return onsets

    def energy(self, X):
        return np.sum(X.magnitude()**2,0) / X.shape[0]

    def hfc(self, X):
        bin_hz = 0.5 * X.sample_rate / float(X.shape[0])
        weights = (bin_hz * np.arange(X.shape[0]))**2
        return np.sum(X.magnitude() * weights[:,np.newaxis],0)

    def _diff(self, X):
        return np.insert(np.abs(np.sum(np.diff(X),0)),0,0)

    def mag_diff(self, X):
        return self._diff(X.magnitude())   
    
    def phase_diff(self, X):
        return self._diff(X.magnitude())   

    def complex_diff(self, X):
        return self._diff(X)