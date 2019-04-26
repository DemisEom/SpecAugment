from setuptools import setup

setup(
   name='SpecAugment',
   version='1.0',
   description='A implementation of "SpecAugment"',
   author='Demis Eom',
   author_email='shelling203@gmail.com',
   packages=['SpecAugment'],  #same as name
   install_requires=['tensorflow', 'librosa'], #external packages as dependencies
)