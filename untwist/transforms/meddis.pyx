import numpy as np
cimport numpy as np
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process(np.ndarray[DTYPE_t, ndim=2] sig, int sample_rate, int medium=False):

    # Assume signal is in pa
    cdef double gain = 1581.1388300841895

    cdef double dt = 1.0 / sample_rate
    cdef double m = 1
    cdef double y = 5.05
    cdef double l = 2500
    cdef double r = 6580
    cdef double x = 66.31
    cdef double h = 50000 # firing rate factor

    cdef double a = 5
    cdef double b = 300
    cdef double g = 2000

    if medium:
        a = 10
        b = 3000
        g = 1000

    cdef double gdt = g * dt
    cdef double ydt = y * dt
    cdef double ldt = l * dt
    cdef double rdt = r * dt
    cdef double xdt = x * dt
    cdef double hdt = h * dt #  Use h instead, so output appox firing rate in spikes/s?
    cdef double kt = g * a / (a + b)
    cdef double c = m * y * kt / (l * kt + y * (l + r))
    cdef double w = c * r / x
    cdef double q = c * (l + r) / kt

    cdef int num_channels = sig.shape[0]
    cdef int num_frames = sig.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] prob = np.zeros((num_channels, num_frames),
                                                     dtype=DTYPE)

    cdef double q_i, c_i, w_i, xa, kt_i, eject, loss, reuptake, reprocess
    for chn in range(num_channels):
        q_i, c_i, w_i = q, c, w
        for i in range(num_frames):

            xa = gain * sig[chn, i] + a
            kt_i = (xa > 0) * gdt * xa / (xa + b)

            if m >= q_i:
                replenish = ydt * (m - q_i)
            else:
                replenish = 0

            eject = kt_i * q_i
            loss = ldt * c_i
            reuptake = rdt * c_i
            reprocess = xdt * w_i

            q_i += replenish - eject + reprocess
            c_i += eject - loss - reuptake
            w_i += reuptake - reprocess

            prob[chn, i] = c_i * h
    return prob
