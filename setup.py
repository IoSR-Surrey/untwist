from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import setuptools.command.install


class build_ext(_build_ext):
    def finalize_options(self):
        import numpy
        from Cython.Build.Dependencies import cythonize
        _build_ext.finalize_options(self)
        self.include_dirs.append(numpy.get_include())
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
        )

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
        'untwist.hpss'
        ],
      setup_requires=['cython', 'numpy'],
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
      ext_modules=[
        Extension(
            "untwist.transforms.meddis", ["untwist/transforms/meddis.pyx"])],
      cmdclass={'build_ext': build_ext},
      )
