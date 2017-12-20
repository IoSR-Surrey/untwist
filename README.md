# untwist

Untwist is python library for audio source separation. It provides a
self-contained object-oriented framework including common source separation
algorithms as well as input/output functions, data management utilities and
time-frequency transforms.

# Installation

To install untwist, clone the github repository and run 
```
python setup.py install
```
from the root directory. Alternatively, you can use [pip](https://pip.pypa.io/en/stable/installing/):
```
pip install git+https://github.com/IoSR-Surrey/untwist
````

These commands will install all dependencies for you and build the required
extension modules for improved performance.

# Usage

A typical usage scenario involves loading an audio file, computing a
time-frequency transform to obtain a spectrogram, and using some of the source
separation algorithms to obtain a time-frequency mask. The mask is then
multiplied with the spectrogram and the result is converted back to the time
domain.  See the [API documentation](http://iosr-surrey.github.io/untwist/) for
reference.  See the `examples` folder for some code examples (you will need to
download [this audio dataset.](https://www.loria.fr/~aliutkus/DSD100subset.zip))

# License

The library is under development at the Centre for Vision, Speech and Signal
Processing, University of Surrey. The source code is published under the MIT
license.
