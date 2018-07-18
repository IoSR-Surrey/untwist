from setuptools import setup, Extension

'''
Handle Cython build after handling dependencies
https://stackoverflow.com/questions/11010151/distributing-a-shared-library-and-some-c-code-with-a-cython-extension-module
'''
class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())


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
      ext_modules=lazy_cythonize(extensions),
)
