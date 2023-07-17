import numpy as np
import torch
import os
import librosa
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchaudio


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


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

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomCrop(32, padding=4),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomCrop(32, padding=4),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
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
    data_dir = ""

    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset='training')
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset='testing')

    return None


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


if __name__ == '__main__':

    data_dir = "/Users/chenrui/Desktop/课件/REPO/edge computing/project/FlexiFed/data"

    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset='training')
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, download=True, subset='testing')

    print(len(train_dataset))
    print(len(test_dataset))
