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

import tensorflow as tf
import numpy as np
import librosa
import random
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def spec_augment(input, time_warping_para, time_masking_para, frequency_masking_para, num_mask):
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

    #Frequency masking
    f = np.random.uniform(low=0.0, high = frequency_masking_para)
    f = int(f)
    v = 128
    f0 = random.randint(0, v - f)

    raw[f0:f0+f, :] = 0

    #time masking
    t = np.random.uniform(low=0.0, high = time_masking_para)
    t = int(t)
    t0 = random.randint(0, ta-t)

    raw[:, t0:t0+t] = 0

  return raw

# load audio file
audio_path = "./data"
audio_file = "./data/61-70968-0002.wav"

# extract melspectrogram
y, sr = librosa.load(audio_file)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

## show base mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# doing specaugment spectrogram
masked_spec = spec_augment(S, time_warping_para=80,
                           time_masking_para=100,
                           frequency_masking_para=27,
                           num_mask=1)

# show time waped & masked spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(masked_spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram MASKED')
plt.tight_layout()
plt.show()

