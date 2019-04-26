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

"""Tests for specaugment"""
import librosa
from SpecAugment import spec_augment

if __name__ == "__main__":

    # First, we need to load sample audio file
    # For the test, I use one of the 'Libiri Speech' data.
    audio_path = "./data"
    audio_file = "./data/61-70970-0007.wav"

    audio, sampling_rate = librosa.load(audio_file)

    # For extracting mel-spectrogram feature, I use 'LibSosa' python audio package.
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128, fmax=8000)

    # we can see extracted mel-spectrogram using simple method 'specshow'
    spec_augment.visualization_melspectrogram(mel_spectrogram=mel_spectrogram, title="Mel Spectrogram")

    # Augmentation using 'SpecAugment(Spectrogram augmentation)"
    warped_masked_mel_spectrogram, warped_mel_spectrogram = spec_augment.spec_augment(mel_spectrogram=mel_spectrogram,
                                                                                      time_warping_para=80,
                                                                                      time_masking_para=100,
                                                                                      frequency_masking_para=27,
                                                                                      num_mask=1)

    # Show time warped & masked spectrogram
    spec_augment.visualization_melspectrogram(mel_spectrogram=warped_masked_mel_spectrogram, title="Warped & Masked Mel Spectrogram")


