# specAugment
This repository is implementation of SpecAugment.  
For Tensorflow.

This is not include LAS or other network.  
I Just Implementing include time warping and masking part.  


ref : https://arxiv.org/pdf/1904.08779.pdf
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

---
## In Implementation
In this Implementation, I define funtion "spec_augment"


---
## How to use
1. install packages
'''python
 pip install -r requirements.txt
'''

2. run 
'''python
python specAugment.py
'''

3. result
run this code you can see the result of masked audio mel-spectrogram

