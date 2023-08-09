import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from scripts.speech_commands_dataset import SpeechCommandsDataset
from scripts.agnews_dataset import AGNewsDataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        return self.dataset[self.idxs[item]]


def split_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)  # 分配给每个user的数据量
    dict_users = {}  # user编号->user分配的数据
    all_idx = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idx, num_items, replace=False))  # 从剩余数据中随机选择
        all_idx = list(set(all_idx) - dict_users[i])  # 从剩余数据中删除已选数据
    return dict_users


def get_cifar(device_num, data_dir):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=False)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=False)

    user_groups = split_iid(train_dataset, device_num)
    user_groups_test = split_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_cinic(device_num, data_dir):
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    user_groups = split_iid(train_dataset, device_num)
    user_groups_test = split_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_speech(device_num, data_dir):
    dataset = SpeechCommandsDataset(data_path=data_dir)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    user_groups = split_iid(train_dataset, device_num)
    user_groups_test = split_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_agnews(device_num, data_dir, is_VDCNN):
    train_dataset = os.path.join(data_dir, 'train.csv')
    test_dataset = os.path.join(data_dir, 'test.csv')

    train_dataset = AGNewsDataset(train_dataset, is_VDCNN=is_VDCNN)
    test_dataset = AGNewsDataset(test_dataset, is_VDCNN=is_VDCNN)

    user_groups = split_iid(train_dataset, device_num)
    user_groups_test = split_iid(test_dataset, device_num)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_dataset(dataset_name, device_num, data_dir, is_VDCNN=0):
    if dataset_name == 'cifar10':
        return get_cifar(device_num, data_dir)

    elif dataset_name == 'cinic10':
        return get_cinic(device_num, data_dir)

    elif dataset_name == 'speechcommands':
        return get_speech(device_num, data_dir)

    # if train on VDCNN, is_VDCNN = 1
    elif dataset_name == 'agnews':
        return get_agnews(device_num, data_dir, is_VDCNN)


def get_common_base_layers(model_list):
    min_idx = 0
    min_len = 1000000

    for i in range(0, len(model_list)):
        if len(model_list[i]) < min_len:
            min_idx = i
            min_len = len(model_list[i])

    commonList = [s for s in model_list[min_idx].keys()]

    for i in range(0, len(model_list)):
        weight_name_list = [s for s in model_list[i].keys()]
        for j in range(len(commonList)):
            if commonList[j] == weight_name_list[j]:
                continue
            else:
                del commonList[j:len(commonList) + 1]
                break
    return commonList

# def get_speech(device_num):
#
#     n_mels = 32
#     train_dataset = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/train'
#     test_dataset = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/test'
#
#     # data_aug_transform = torchvision.transforms.Compose(
#     #     [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
#     #      TimeshiftAudioOnSTFT(), FixSTFTDimension()])
#     # bg_dataset = BackgroundNoiseDataset(background_noise, data_aug_transform)
#     # add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
#
#     # train_transform = torchvision.transforms.Compose(
#     #     [ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
#     train_transform = torchvision.transforms.Compose(
#         [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
#     train_dataset = SpeechCommandsDataset(train_dataset,
#                                           torchvision.transforms.Compose([LoadAudio(),
#                                                                           FixAudioLength(),
#                                                                           # data_aug_transform,
#                                                                           # add_bg_noise,
#                                                                           train_transform]))
#
#     test_transform = torchvision.transforms.Compose(
#         [ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
#     test_dataset = SpeechCommandsDataset(test_dataset,
#                                          torchvision.transforms.Compose([LoadAudio(),
#                                                                          FixAudioLength(),
#                                                                          test_transform]))
#
#     # train_transform = torchvision.transforms.Compose([
#     #     torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=1024),
#     #     torchaudio.transforms.TimeMasking(3),
#     #     torchaudio.transforms.FrequencyMasking(3),
#     # ])
#     # train_dataset = SpeechCommandsDataset(train_dataset, train_transform)
#     #
#     # test_transform = torchvision.transforms.Compose([torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=1024)])
#     # test_dataset = SpeechCommandsDataset(test_dataset, test_transform)
#
#     user_groups = split_iid(train_dataset, device_num)
#     user_groups_test = split_iid(test_dataset, device_num)
#
#     return train_dataset, test_dataset, user_groups, user_groups_test
