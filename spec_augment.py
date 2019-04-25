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

def spec_augment(mel_spectrogram, time_warping_para, frequency_masking_para, time_masking_para, num_mask):
  """Spec augmentation.
  Related paper : https://arxiv.org/pdf/1904.08779.pdf
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
  tau = 128

  """(TO DO)Time warping
  In paper Time warping written as follows. 'Given a log mel spectrogram with τ time steps,
  we view it as an image where the time axis is horizontal and the frequency axis is vertical.
  A random point along the horizontal line passing through the center of the image within the time steps (W, τ − W)
  is to be warped either to the left or right by a distance w chosen from a uniform distribution
  from 0 to the time warp parameter W along that line.'
  In paper Using Tensorflow's 'sparse-image-warp'.
  """
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

  """ Masking line loop """
  for i in range(num_mask):
    """Frequency masking
    In paper Frequency masking written as follows. 'Frequency masking is applied so that f consecutive mel frequency
    channels [f0, f0 + f) are masked, where f is first chosen from a uniform distribution from 0 to the frequency mask parameter F,
    and f0 is chosen from [0, ν − f). ν is the number of mel frequency channels.'
    In this code, ν was written with v. ands For choesing 'f' uniform distribution, I using random package.
    """
    f = np.random.uniform(low=0.0, high=frequency_masking_para)
    f = int(f)
    v = 128  # Now hard coding but I will improve soon.
    f0 = random.randint(0, v - f)
    mel_spectrogram[f0:f0+f, :] = 0


    """Time masking
    In paper Time masking written as follows. 'Time masking is applied so that t consecutive time steps
    [t0, t0 + t) are masked, where t is first chosen from a uniform distribution from 0 to the time mask parameter T,
    and t0 is chosen from [0, τ − t).'
    In this code, τ(tau) was written with tau. and For choesing 't' uniform distribution, I using random package.
    """
    t = np.random.uniform(low=0.0, high=time_masking_para)
    t = int(t)
    t0 = random.randint(0, tau-t)
    mel_spectrogram[:, t0:t0+t] = 0

  return mel_spectrogram

# First, we need to load sample audio file
# For the test, I use one of the 'Libiri Speech' data.
audio_path = "./data"
audio_file = "./data/61-70968-0002.wav"
audio, sampling_rate = librosa.load(audio_file)

# For extracting mel-spectrogram feature, I use 'LibSosa' python audio package.
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128, fmax=8000)

# we can see extracted mel-spectrogram using simple method 'specshow'
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()

# Augmentation using 'SpecAugment(Spectrogram augmentation)"
masked_spectrogram = spec_augment(mel_spectrogram=mel_spectrogram,
                                  time_warping_para=80,
                                  time_masking_para=100,
                                  frequency_masking_para=27,
                                  num_mask=1)

# Show time warped & masked spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(masked_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Masked Mel Spectrogram')
plt.tight_layout()
plt.show()

