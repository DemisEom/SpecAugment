# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.contrib.image.python.ops import sparse_image_warp
import tensorflow as tf
import numpy as np
import librosa
import random
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.python.framework import constant_op

def spec_augment(input, time_warping_para, frequency_masking_para, time_masking_para, num_mask):
  """Compute spec augmentation.
  The Parameters, "Augmentation parameters for policies", refer to the 'Tabel 1' in the paper.
  Args:
    input: Extracted mel-spectrogram, numpy array.
    time_warping_para: "Augmentation parameters for policies W", LibriSpeech is 80.
    frequency_masking_para: "Augmentation parameters for policies F", LibriSpeech is 27.
    time_masking_para: "Augmentation parameters for policies T", LibriSpeech is 100.
    num_mask : number of masking lines.
  Returns:
    raw : warped and masked mel spectrogram.
  """
  raw = input
  ta = 128

  ## todo : time warping
  # raw = raw.reshape([1, raw.shape[0], raw.shape[1], 1])
  # w = random.randint(0, time_warping_para)
  # warped = tf.contrib.image.sparse_image_warp(raw,
  #                                             source_control_point_locations = [1, time_warping_para, 2],
  #                                             dest_control_point_locations = [1, ta - time_warping_para, 2],
  #                                             interpolation_order=2,
  #                                             regularization_weight=0.0,
  #                                             num_boundary_points=0,
  #                                             name='sparse_image_warp'
  #                                             )

  # repeat number of mask lines
  for i in range(num_mask):
    """Frequency masking
    In paper Frequency masking written as follows. 'Frequency masking is applied so that f consecutive mel frequency
    channels [f0, f0 + f) are masked, where f is first chosen from a uniform distribution from 0 to the frequency mask parameter F,
    and f0 is chosen from [0, ν − f). ν is the number of mel frequency channels.'
    """
    f = np.random.uniform(low=0.0, high = frequency_masking_para)
    f = int(f)
    v = 128  # Now hard coding but I will improve soon.
    f0 = random.randint(0, v - f)
    raw[f0:f0+f, :] = 0


    """Time masking
    In paper Frequency masking written as follows. 'Time masking is applied so that t consecutive time steps
    [t0, t0 + t) are masked, where t is first chosen from a uniform distribution from 0 to the time mask parameter T,
    and t0 is chosen from [0, τ − t).'
    """
    t = np.random.uniform(low=0.0, high = time_masking_para)
    t = int(t)
    t0 = random.randint(0, ta-t)
    raw[:, t0:t0+t] = 0

  return raw

# First, we need to load sample audio file
# we use one of the LibriSpeech data
audio_path = "./data"
audio_file = "./data/61-70968-0002.wav"
y, sr = librosa.load(audio_file)

# for extracting mel-spectrogram feature, we using LibSosa.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# we can see extracted mel-spectrogram just simple method using 'specshow'
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# Doing spec augment spectrogram
masked_spec = spec_augment(S, time_warping_para=80,
                           time_masking_para=100,
                           frequency_masking_para=27,
                           num_mask=1)

# Show time warped & masked spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(masked_spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram MASKED')
plt.tight_layout()
plt.show()

