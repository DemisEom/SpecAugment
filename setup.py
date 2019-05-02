from setuptools import setup, find_packages

setup(
   name='SpecAugment',
   version='1.2',
   description='A implementation of "SpecAugment"',
   url              = 'https://github.com/shelling203/SpecAugment',
   packages         = find_packages(exclude = ['docs', 'tests*']),
   install_requires=['tensorflow', 'librosa', 'matplotlib', 'torch'], #external packages as dependencies
)