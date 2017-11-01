import numpy as np
import matplotlib.pyplot as plt
import theano
from untwist import data
from untwist import transforms
from untwist import neuralnetworks
from utils import get_stems
floatX = theano.config.floatX

# Load the stems
stems = get_stems(song_idx=0,
                  path_to_dsd100_subset='/scratch/DSD100subset',
                  mono=True)

train_frames = 2000
fft_size = 1024
n_bins = fft_size // 2 + 1

stft = transforms.STFT(fft_size=fft_size)
istft = transforms.ISTFT(fft_size=fft_size)

# Compute spectrograms of the mixture, vocals and accompaniment
mix_spec = stft.process(stems.mixture)
vocals_spec = stft.process(stems.vocals)
accompaniment_spec = stft.process(stems.accompaniment)

# The network and training algorithm
mlp = neuralnetworks.MLP(input_size=n_bins,
                         output_size=n_bins,
                         hidden_sizes=[n_bins, n_bins])

sgd = neuralnetworks.SGD(mlp,
                         learning_rate=0.05,
                         momentum=0.2,
                         batch_size=200,
                         iterations=100)

# Ideal mask for the vocals
ideal_mask = data.audio.BinaryMask(
    vocals_spec.magnitude(), accompaniment_spec.magnitude())

# Create dataset for adding training data
ds = data.dataset.Dataset(x_width=n_bins,
                          x_type=floatX,
                          y_width=n_bins,
                          y_type=np.bool_)

'''
Mixture spectrogram as input, ideal mask for vocal as output
Need to transpose so each column is a frequency bin
'''
Xtrain = mix_spec[:, :train_frames].magnitude().T
Ytrain = ideal_mask[:, :train_frames].T


# Add to the dataset and standardise
ds.add(Xtrain, Ytrain)
ds.standardize()
ds.shuffle()

# Go on then...
sgd.train(ds)

# Predict using remaining spectra
test_data = mix_spec[:, train_frames:].magnitude().T
test_data = ds.standardize_points(test_data)
test_prediction = sgd.predict(test_data)

# Predict the mask, threshold to get binary (> 0.5)
estimated_mask = ideal_mask[:, train_frames:].copy()
estimated_mask[:] = test_prediction.T > 0.5

# Example audio
ideal_vocal = mix_spec[:, train_frames:] * ideal_mask[:, train_frames:]
istft.process(ideal_vocal).write('ideal_masked_vocal.wav')
predicted_vocal = mix_spec[:, train_frames:] * estimated_mask
istft.process(predicted_vocal).write('predicted_vocal.wav')

# Transpose back to bin x time and plot
plt.subplot(4, 1, 1)
test_data.T.plot(title="mixture", xlabel=False)
plt.subplot(4, 1, 2)
vocals_spec[:, train_frames:].plot(title="target signal", xlabel=False)
plt.subplot(4, 1, 3)
ideal_mask[:, train_frames:].plot(title="ideal mask", xlabel=False)
plt.subplot(4, 1, 4)
estimated_mask.plot(title="estimated mask")
plt.show()
