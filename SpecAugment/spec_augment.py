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
# ==============================================================================
"""SpecAugment Implementation.
Related paper : https://arxiv.org/pdf/1904.08779.pdf

In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa.display
import tensorflow as tf
from tensorflow.contrib.image import sparse_image_warp
from tensorflow.python.framework import constant_op
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def spec_augment(mel_spectrogram, time_warping_para, frequency_masking_para, time_masking_para, num_mask):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      input(numpy array): Extracted mel-spectrogram.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, dafault = 80
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, dafault = 100
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, dafault = 27
      num_mask(float): number of masking lines.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """

    # Step 1 : Time warping (TO DO)
    tau = mel_spectrogram.shape[1]

    # Image warping control point setting
    control_point_locations = np.asarray([[64, 64], [64, 80]])
    control_point_locations = constant_op.constant(
        np.float32(np.expand_dims(control_point_locations, 0)))

    control_point_displacements = np.ones(
        control_point_locations.shape.as_list())
    control_point_displacements = constant_op.constant(
        np.float32(control_point_displacements))

    # mel spectrogram data type convert to tensor constant for sparse_image_warp
    mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
    mel_spectrogram_op = constant_op.constant(np.float32(mel_spectrogram))
    w = random.randint(0, time_warping_para)

    warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_op,
                                                     source_control_point_locations=control_point_locations,
                                                     dest_control_point_locations=control_point_locations + control_point_displacements,
                                                     interpolation_order=2,
                                                     regularization_weight=0,
                                                     num_boundary_points=0
                                                     )

    # Change data type of warp result to numpy array for masking step
    with tf.Session() as sess:
        warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op)

    warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                             warped_mel_spectrogram.shape[2]])

    # loop Masking line number
    for i in range(num_mask):
        # Step 2 : Frequency masking
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        v = 128  # Now hard coding but I will improve soon.
        f0 = random.randint(0, v - f)
        warped_mel_spectrogram[f0:f0 + f, :] = 0

        # Step 3 : Time masking
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_mel_spectrogram[:, t0:t0 + t] = 0

    return warped_mel_spectrogram


def visualization_melspectrogram(mel_spectrogram, title):
    """visualizing result of specAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

