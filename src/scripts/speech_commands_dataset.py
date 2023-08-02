"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import os
import numpy as np
from pprint import pprint

import librosa


from torch.utils.data import Dataset

__all__ = ['CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset']

CLASSES = ['right', 'eight', 'cat', 'tree', 'bed', 'happy', 'go', 'dog', 'no', 'wow',
           'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero', 'seven', 'up',
           'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 'five', 'off', 'four']


class SpeechCommandsDataset(Dataset):

    def __init__(self, folder, transform=None, classes=CLASSES):

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        data = []
        for c in classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data


class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    train_dir = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/test'
    train_dataset = SpeechCommandsDataset(train_dir)
    pprint(len(train_dataset.data))
    print(10000 / 40)
