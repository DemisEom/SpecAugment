from setuptools import setup, find_packages

setup(
   name='SpecAugment',
   version='1.0',
   description='A implementation of "SpecAugment"',
   url              = 'https://github.com/shelling203/SpecAugment',
   packages=find_packages(exclude=['tests']),
   install_requires=['tensorflow', 'librosa'], #external packages as dependencies
)