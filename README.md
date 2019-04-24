# SpecAugment
This is a implementation of SpecAugment with Tensorflow, introduced by Google Brain[1]. 


... 

## How to use

First, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

Next, you need to install some audio libraries work properly. To install the requirement packages. Run the following command:

> pip install -r requirements.txt


And then, run the specAugment.py program. It modifies the spectrogram by warping it in the time direction, masking blocks of consecutive frequency channels, and masking blocks of utterances in time.  

> python specAugment.py

<p align="center">
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_1.png" alt="Example result of base spectrogram"/ width=600>
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_2.png" alt="Example result of base spectrogram"/ width=600>
</p> 


# Reference

1. https://arxiv.org/pdf/1904.08779.pdf





---
## About
In paper(SpecAugment) proposed 3 augmentation polcy

1. Time Warping(To do...)
- In paper using Tensorflow's "sparse image warp ". 
- This i implementing now.

2. Frequency masking
- 

3. Time masking
- 

## In Implementation
In this Implementation, I define funtion "spec_augment"


## How to use
1. install packages
```python
 pip install -r requirements.txt
```

2. run 
```python
python specAugment.py
```

3. result
run this code you can see the result of masked audio mel-spectrogram

<p align="center">
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_1.png" alt="Example result of base spectrogram"/ width=600>
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_2.png" alt="Example result of base spectrogram"/ width=600>
</p> 


## Notes
- now I implementing time warp part. but, sparse image warp of tensorflow don't have specific reference.
