import torch
import torch.nn as nn
import warnings
from torch.nn.init import kaiming_normal_

warnings.filterwarnings("ignore")

__all__ = [
    'CharCNN',
    'CharCNN_3',
    'CharCNN_4',
    'CharCNN_5',
    'CharCNN_6'
]


def first_stage(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3),
        nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=7),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3)
    )


def second_stage(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        in_channels = out_channels

    return nn.Sequential(*layers)


class CharCNN(nn.Module):
    def __init__(self, version, m=70, l0=1014, num_classes=4):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(m + 1, m, padding_idx=0)
        self.embedding.weight.data[1:].copy_(torch.eye(m))
        self.embedding.weight.requires_grad = False

        self.block1 = first_stage(in_channels=m, out_channels=256)
        size = ((l0 - 6) / 3 - 6) / 3

        if version == 3:
            self.size = int((size - 2) / 3)
            self.block2 = None

        elif version == 4:
            self.size = int((size - 2 * 1 - 2) / 3)
            self.block2 = second_stage(1, 256, 256)

        elif version == 5:
            self.size = int((size - 2 * 2 - 2) / 3)
            self.block2 = second_stage(2, 256, 256)

        elif version == 6:
            self.size = int((size - 2 * 3 - 2) / 3)
            self.block2 = second_stage(3, 256, 256)

        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.size * 256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.05)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.block1(x)
        if self.block2 is not None:
            x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def CharCNN_3(m=70, l0=1014, num_classes=4):
    return CharCNN(3, m=m, l0=l0, num_classes=num_classes)


def CharCNN_4(m=70, l0=1014, num_classes=4):
    return CharCNN(4, m=m, l0=l0, num_classes=num_classes)


def CharCNN_5(m=70, l0=1014, num_classes=4):
    return CharCNN(5, m=m, l0=l0, num_classes=num_classes)


def CharCNN_6(m=70, l0=1014, num_classes=4):
    return CharCNN(6, m=m, l0=l0, num_classes=num_classes)


if __name__ == '__main__':
    model = CharCNN_6()
    out = model(torch.zeros((1, 1014), dtype=torch.long))
