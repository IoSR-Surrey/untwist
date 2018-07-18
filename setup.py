from setuptools import setup, Extension
import subprocess
import pkg_resources
import sys

# Let pip handle setup dependencies
# https://bitbucket.org/dholth/setup-requires
sys.path[0:0] = ['setup-requires']
pkg_resources.working_set.add_entry('setup-requires')

def missing_requirements(specifiers):
    for specifier in specifiers:
        try:
            pkg_resources.require(specifier)
        except pkg_resources.DistributionNotFound:
            yield specifier


def install_requirements(specifiers):
    to_install = list(specifiers)
    if to_install:
        cmd = [sys.executable, "-m", "pip", "install",
            "-t", "setup-requires"] + to_install
        subprocess.call(cmd)

requires = ['numpy', 'cython']
install_requirements(missing_requirements(requires))

def extensions():
    from Cython.Build import cythonize
    import numpy as np
    ext = Extension("untwist.transforms.meddis",
                    ["untwist/transforms/meddis.pyx"],
                    include_dirs=[np.get_include()])
    return cythonize([ext])

setup(name='untwist',
      version='0.1.dev1',
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
      install_requires=[
          'cython',
          'h5py',
          'numpy',
          'soundfile',
          'scipy',
          'six',
          'theano',
          'matplotlib'
      ],
      #ext_modules=lazy_cythonize(extensions),
      ext_modules=extensions(),
)
