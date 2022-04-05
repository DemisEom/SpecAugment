from setuptools import setup, find_packages

setup(
   name='specAugment',
   version='1.2.3',
   description='A implementation of "SpecAugment"',
   url              = 'https://github.com/seriousran/SpecAugment',
   packages         = find_packages(exclude = ['docs', 'tests*']),
   install_requires=['tensorflow', 'librosa', 'matplotlib', 'torch'], #external packages as dependencies
)
