import numpy as np
import matplotlib.pyplot as plt
from untwist.data import Wave, RatioMask
from untwist.transforms import STFT, ISTFT
from untwist.factorizations import RPCA

stft = STFT()
istft = ISTFT()
rpca = RPCA(iterations=100)

# Try with vocals over repetitive music background
x = Wave.read("mixture.wav")
X = stft.process(x[:, 0])

# this will take some time
(L, S) = rpca.process(X.magnitude())

M = RatioMask(S, L)
v = istft.process(X * M)
v.write("vocal_estimate.wav")

plt.subplot(4, 1, 1)
X.plot(label_x=False, title="mixture")
plt.subplot(4, 1, 2)
L.plot(label_x=False, title="L")
plt.subplot(4, 1, 3)
S.plot(label_x=False, title="S")
plt.subplot(4, 1, 4)
M.plot(title="estimated mask")
plt.show()
