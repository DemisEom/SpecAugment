import tensorflow as tf
import numpy as np
import librosa
import random
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
from tensorflow.keras.preprocessing import image as kp_image

def load_sounds_in_folder(foldername):
    """ Loads all sounds in a folder"""
    sounds = []
    for filename in os.listdir(foldername):
        X, sr = librosa.load(os.path.join(foldername,filename))
        sounds.append(X)
    return sounds

def spec_augment(input, time_warping_para, time_masking_para, frequency_masking_para, num_mask):
  raw = input
  ta = 128
  raw = raw.reshape([1, raw.shape[0], raw.shape[1], 1])

  # time warping
  w = random.randint(0, time_warping_para)
  warped = tf.contrib.image.sparse_image_warp(raw,
                                              source_control_point_locations = [1, time_warping_para, 2],
                                              dest_control_point_locations = [1, ta - time_warping_para, 2],
                                              interpolation_order=2,
                                              regularization_weight=0.0,
                                              num_boundary_points=0,
                                              name='sparse_image_warp'
                                              )

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

## mel spectrogram test
audio_file = "/Users/demis/Workspace/deep_speech/deepspeech_tensorflow/data/librispeech_data/test-clean/LibriSpeech/test-clean-wav/61-70968-0002.wav"
audio_path = "/Users/demis/Workspace/deep_speech/deepspeech_tensorflow/data/librispeech_data/test-clean/LibriSpeech/test-clean-wav"
img_path = "/Users/demis/Workspace/deep_speech/data/warp_test_img.jpg"

test_img = Image.open(img_path)
img = kp_image.img_to_array(test_img)
# DATA = load_sounds_in_folder(audio_path)

y, sr = librosa.load(audio_file)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

## show base mel-spectrogram
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()

# show masked spectrogram
masked_spec = spec_augment(S, time_warping_para=80,
                           time_masking_para=100,
                           frequency_masking_para=27,
                           num_mask=1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(masked_spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram MASKED')
plt.tight_layout()
plt.show()

print(S)



