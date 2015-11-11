from distutils.core import setup


setup(name='untwist',
      version='0.1.dev0',
      author = 'Gerard Roma',
      author_email='g.roma@surrey.ac.uk',
      packages=[
        'untwist', 
        'untwist.base',
        'untwist.data',
        'untwist.soundcard',
        'untwist.transforms', 
        'untwist.factorizations', 
        'untwist.neuralnetworks'
        ],
      )
