import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

__all__ = [
    'VDCNN',
    'VDCNN_9',
    'VDCNN_17',
    'VDCNN_29',
    'VDCNN_49'
]

cfg = {
    'A': [1, 1, 1, 1],
    'B': [2, 2, 2, 2],
    'C': [5, 5, 2, 2],
    'D': [8, 8, 5, 3]
}

arch = [64, 128, 256, 512]


class ConvBlock(nn.Module):

    def __init__(self, in_channels, n_filters, shortcut=False, downsample=None):
        super(ConvBlock, self).__init__()
        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(n_filters)

        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        residual = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            output += residual

        output = F.relu(output)
        return output


def get_block(in_channels, n_filters, num_block, shortcut=False, downsample=None):
    layers = [ConvBlock(in_channels, n_filters, shortcut, downsample)]
    layers += [ConvBlock(n_filters, n_filters, shortcut)] * (num_block - 1)
    layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
    return nn.Sequential(*layers)


def get_downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(out_channels)
    )


class VDCNN(nn.Module):

    def __init__(self, num_blocks, num_embedding=69, num_classes=4, shortcut=False):
        super(VDCNN, self).__init__()

        self.embed = nn.Embedding(num_embedding, 16, padding_idx=0)
        self.first_conv = nn.Conv1d(16, 64, kernel_size=3, padding=1)

        self.block0 = get_block(in_channels=64, n_filters=64,
                                num_block=num_blocks[0], shortcut=shortcut)
        self.block1 = get_block(in_channels=64, n_filters=128,
                                num_block=num_blocks[1], shortcut=shortcut,
                                downsample=get_downsample(64, 128))
        self.block2 = get_block(in_channels=128, n_filters=256,
                                num_block=num_blocks[2], shortcut=shortcut,
                                downsample=get_downsample(128, 256))
        self.block3 = get_block(in_channels=256, n_filters=512,
                                num_block=num_blocks[3], shortcut=shortcut,
                                downsample=get_downsample(256, 512))

        self.maxPool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.maxPool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.maxPool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, padding=1),
            self.block0,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self.block1,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self.block2,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self.block3,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveMaxPool1d(8),
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, num_classes), nn.ReLU(),
        )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        output = self.embed(x)
        output = output.transpose(1, 2)
        output = self.features(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def VDCNN_9():
    return VDCNN(cfg['A'])


def VDCNN_17():
    return VDCNN(cfg['B'])


def VDCNN_29():
    return VDCNN(cfg['C'])


def VDCNN_49():
    return VDCNN(cfg['D'])


# model = VDCNN_9()
# y = model(torch.zeros(1, 1024, dtype=torch.long))
