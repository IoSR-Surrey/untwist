import numpy as np
import matplotlib.pyplot as plt
import theano
from untwist.data import Wave, Dataset, BinaryMask
from untwist.transforms import STFT, ISTFT
from untwist.neuralnetworks import MLP, SGD
floatX = theano.config.floatX

n_bins = 513
train_frames = 10000

target = Wave.read("target.wav")[:, 0]
background = Wave.read("background.wav")[:, 0]
mix = target + background

stft = STFT()
istft = ISTFT()
mlp = MLP(n_bins, n_bins, [n_bins, n_bins])
sgd = SGD(mlp,
          learning_rate=0.05, momentum=0.2,
          batch_size=200, iterations=100)

X = stft.process(mix)
T = stft.process(target)
B = stft.process(background)

ideal_mask = BinaryMask(T.magnitude(), B.magnitude())
ds = Dataset(n_bins, floatX, n_bins, np.bool_)
Xtrain = X[:, :train_frames].magnitude().T
Ytrain = ideal_mask[:, :train_frames].T
print(Xtrain.shape, Ytrain.shape, n_bins)

ds.add(Xtrain, Ytrain)
ds.standardize()
ds.shuffle()
sgd.train(ds)

Xtest = X[:, train_frames:].magnitude().T
Xtest = ds.standardize_points(Xtest)
pred = sgd.predict(Xtest)
estimated_mask = np.empty_like(Xtest.T)
estimated_mask[:] = pred.T > 0.5
plt.subplot(4, 1, 1)
Xtest.T.plot(label_x=False, title="mixture")
plt.subplot(4, 1, 2)
T[:, train_frames:].plot(label_x=False, title="target signal")
plt.subplot(4, 1, 3)
ideal_mask[:, train_frames:].plot(label_x=False, title="ideal mask")
plt.subplot(4, 1, 4)
estimated_mask.plot(title="estimated mask")
plt.show()
