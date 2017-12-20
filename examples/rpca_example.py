import numpy as np
import matplotlib.pyplot as plt
from untwist import factorizations
from untwist import data
from untwist import transforms
from utils import get_stems


stems = get_stems(song_idx=2,
                  path_to_dsd100_subset='/scratch/DSD100subset',
                  mono=True)

stft = transforms.STFT()
istft = transforms.ISTFT()
rpca = factorizations.RPCA(iterations=100)

# Try with vocals over repetitive music background
mixture_spec = stft.process(stems.mixture)

# this will take some time
low_rank_l, sparse_s = rpca.process(mixture_spec.magnitude())

# Calculate binary mask and synthesise vocals and accompaniment
mask = data.audio.BinaryMask(sparse_s, low_rank_l, 0)
istft.process(mixture_spec * mask).normalize().write(
    "vocal_estimate_mask.wav")
istft.process(mixture_spec * (1 - mask)).normalize().write(
    "accomp_estimate_mask.wav")
stems.mixture.write("mixture.wav")

plt.subplot(3, 1, 1)
mixture_spec.plot(xlabel=False, title="Mixture")
plt.subplot(3, 1, 2)
low_rank_l.plot(xlabel=False, title="Low-rank matrix (accompaniment)")
plt.subplot(3, 1, 3)
sparse_s.plot(xlabel=False, title="Sparse matrix (vocals)")
plt.show()
