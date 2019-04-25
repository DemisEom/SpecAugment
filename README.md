# SpecAugment

This is a implementation of SpecAugment that speech data augmentation method which directly process the spectrogram with Tensorflow, introduced by Google Brain[1]. This is currently under the Apache 2.0, Please feel free to use for your project. Enjoy!

## How to use

First, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

Next, you need to install some audio libraries work properly. To install the requirement packages. Run the following command:

```bash
pip install -r requirements.txt
```

And then, run the specAugment.py program. It modifies the spectrogram by warping it in the time direction, masking blocks of consecutive frequency channels, and masking blocks of utterances in time.  

```bash
python spec_augment_test.py
```

<p align="center">
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_1.png" alt="Example result of base spectrogram"/ width=600>
  <img src="https://github.com/shelling203/specAugment/blob/master/images/Figure_2.png" alt="Example result of base spectrogram"/ width=600>
</p>


# Reference

1. https://arxiv.org/pdf/1904.08779.pdf

