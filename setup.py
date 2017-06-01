from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np


setup(name='untwist',
      version='0.1.dev0',
      author='Gerard Roma',
      author_email='g.roma@surrey.ac.uk',
      packages=[
        'untwist',
        'untwist.base',
        'untwist.data',
        'untwist.analysis',
        'untwist.utilities',
        'untwist.soundcard',
        'untwist.transforms',
        'untwist.filters',
        'untwist.factorizations',
        'untwist.neuralnetworks'
        ],
      ext_modules=cythonize(
          Extension('*', ['./untwist/transforms/meddis.pyx'])),
      include_dirs=[np.get_include()]
      )
