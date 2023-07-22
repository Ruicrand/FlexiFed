import os
from random import random

import librosa
import numpy as np
import torch
import torchvision
import torchaudio
from pprint import pprint
from torch.utils.data import Dataset
from torchvision import datasets

from process.transforms_stft import *
from process.transforms_wav import *
from scripts.speech_commands_dataset import SpeechCommandsDataset, BackgroundNoiseDataset
import matplotlib.pyplot as plt


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        return self.dataset[self.idxs[item]]

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = 12
        count = np.zeros(nclasses)
        for item in self.idxs:
            count[self.dataset[item]['target']] += 1

        N = len(self.idxs)
        weight_per_class = N * 1.0 / count
        weight = np.zeros(len(self.idxs))

        for idx, item in enumerate(self.idxs):
            weight[idx] = weight_per_class[self.dataset[item]['target']]
        return weight


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)  # 分配给每个user的数据量
    dict_users = {}  # user编号->user分配的数据
    all_idx = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idx, num_items, replace=False))  # 从剩余数据中随机选择
        all_idx = list(set(all_idx) - dict_users[i])  # 从剩余数据中删除已选数据
    return dict_users


def get_cifar(device_num):
    data_dir = '/kaggle/input/cifar10-python'

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.RandomCrop(32, padding=4),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=False)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=False)

    # 按照独立同分布将数据分成device_num组
    user_groups = cifar_iid(train_dataset, device_num)
    user_groups_test = cifar_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


"""
train/
    train/airplane
    train/automobile
    train/...
valid/
    valid/...
test/
    test/...
"""


def get_cinic(device_num):
    data_dir = '/path/to/cinic/directory'
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.RandomCrop(32, padding=4),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir + '/train'),
        transform=train_transform)

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir + '/test'),
        transform=train_transform)

    user_groups = cifar_iid(train_dataset, device_num)
    user_groups_test = cifar_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_speech(device_num):
    n_mels = 32
    background_noise = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/train/_background_noise_'
    train_dataset = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/train'
    test_dataset = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/test'

    data_aug_transform = torchvision.transforms.Compose(
        [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
         TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(background_noise, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)

    train_transform = torchvision.transforms.Compose(
        [ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_dataset = SpeechCommandsDataset(train_dataset,
                                          torchvision.transforms.Compose([LoadAudio(),
                                                                          data_aug_transform,
                                                                          add_bg_noise,
                                                                          train_transform]))

    test_transform = torchvision.transforms.Compose(
        [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    test_dataset = SpeechCommandsDataset(test_dataset,
                                         torchvision.transforms.Compose([LoadAudio(),
                                                                         FixAudioLength(),
                                                                         test_transform]))

    user_groups = cifar_iid(train_dataset, device_num)
    user_groups_test = cifar_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_dataset(dataset_name, device_num):
    if dataset_name == 'cifar10':
        return get_cifar(device_num)

    elif dataset_name == 'cinic10':
        return get_cinic(device_num)

    elif dataset_name == 'SpeechCommands':
        return get_speech(device_num)


def get_common_base_layers(model_list):
    min_idx = 0
    min_len = 1000000

    # 找到参数最少的一个模型
    for i in range(0, len(model_list)):
        if len(model_list[i]) < min_len:
            min_idx = i
            min_len = len(model_list[i])

    commonList = [s for s in model_list[min_idx].keys()]

    # 找到common base layers
    for i in range(0, len(model_list)):
        weight_name_list = [s for s in model_list[i].keys()]
        for j in range(len(commonList)):
            if commonList[j] == weight_name_list[j]:
                continue
            else:
                del commonList[j:len(commonList) + 1]  # 从哪一层开始不同，删去该层后所有层
                break
    return commonList


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()

#
# if __name__ == '__main__':
#     data_dir = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/train/bed/0a7c2a8d_nohash_0.wav'
#
#     wave, _ = torchaudio.load(data_dir)
#
#     melspec = torchaudio.transforms.MelSpectrogram(n_mels=32, n_fft=1024)(wave)
#
#     plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
