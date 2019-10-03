''' RAVDESS Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import scipy
import librosa
import torch.utils.data as data

def extract_features(S, sr):
    
    S_spectrogram = librosa.feature.melspectrogram(y=S, sr=sr, n_mels=128)  # spectrogram features
    S = librosa.power_to_db(S_spectrogram)
    S = scipy.fftpack.dct(S, axis=0, type=2, norm='ortho')[:12]  # MFCC features
    
    S_delta = librosa.feature.delta(S)
    S_delta2 = librosa.feature.delta(S, order=2)
    
    S = np.vstack([S_spectrogram, S_delta, S_delta2])     
    
    return S

class COMMANDS(data.Dataset):
    """`COMMANDS Dataset.
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('data/COMMANDS_data.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_feature']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((51768, 16000))

        else:
            self.PrivateTest_data = self.data['Test_feature']
            self.PrivateTest_labels = self.data['Test_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((12959, 16000))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            voice, target = self.train_data[index], self.train_labels[index]
        else:
            voice, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        if self.transform is not None:
            voice = self.transform(samples=voice, sample_rate=16000)
            voice = extract_features(voice,16000)
        else:
            voice = extract_features(voice,16000)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        return voice, target
#         return voice.T, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        else:
            return len(self.PrivateTest_data)
