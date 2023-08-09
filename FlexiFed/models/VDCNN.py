import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VDCNN',
    'VDCNN_9',
    'VDCNN_17',
    'VDCNN_29',
    'VDCNN_49'
]

class VDCNN(nn.Module):
    def __init__(self, block, layers, m=69, l0=1024, num_classes=4):
        super(VDCNN, self).__init__()
        self.embedding = nn.Embedding(m, 16, padding_idx=0)
        self.conv0 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64)
        )
        # layer1
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.pooling1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1 = ConvBlock(64, 128, downsample=nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
        ))
        # layer2
        self.layer2 = self.make_layer(block, 128, layers[1])
        self.pooling2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(128, 256, downsample=nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(256),
        ))
        # layer3
        self.layer3 = self.make_layer(block, 256, layers[2])
        self.pooling3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(256, 512, downsample=nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
        ))
        # layer4
        self.layer4 = self.make_layer(block, 512, layers[3])
        self.pooling4 = nn.AdaptiveMaxPool1d(8)
        # full connection
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )
        '''Initialize the weights of conv layers'''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv0(x)

        x = self.layer1(x)
        x = self.pooling1(x)
        x = self.conv1(x)

        x = self.layer2(x)
        x = self.pooling2(x)
        x = self.conv2(x)

        x = self.layer3(x)
        x = self.pooling3(x)
        x = self.conv3(x)

        x = self.layer4(x)
        x = self.pooling4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(planes, planes))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def VDCNN_9():
    return VDCNN(block=ConvBlock, layers=[0, 0, 0, 0])


def VDCNN_17():
    return VDCNN(block=ConvBlock, layers=[1, 1, 1, 1])


def VDCNN_29():
    return VDCNN(block=ConvBlock, layers=[4, 4, 1, 1])


def VDCNN_49():
    return VDCNN(block=ConvBlock, layers=[7, 7, 4, 2])

# cfg = {
#     'A': [1, 1, 1, 1],
#     'B': [2, 2, 2, 2],
#     'C': [5, 5, 2, 2],
#     'D': [8, 8, 5, 3]
# }
#
# arch = [64, 128, 256, 512]
#
#
# class ConvBlock(nn.Module):
#
#     def __init__(self, in_channels, n_filters, shortcut, downsample=None):
#         super(ConvBlock, self).__init__()
#         self.shortcut = shortcut
#         self.downsample = downsample
#
#         self.conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1, stride=1)
#         self.bn1 = nn.BatchNorm1d(n_filters)
#
#         self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1, stride=1)
#         self.bn2 = nn.BatchNorm1d(n_filters)
#
#     def forward(self, x):
#         residual = x
#         output = F.relu(self.bn1(self.conv1(x)))
#         output = self.bn2(self.conv2(output))
#
#         if self.shortcut:
#             if self.downsample is not None:
#                 residual = self.downsample(x)
#             output += residual
#
#         output = F.relu(output)
#         return output
#
#
# def get_block(in_channels, n_filters, num_block, shortcut=False, downsample=None):
#     layers = [ConvBlock(in_channels, n_filters, shortcut, downsample)]
#     for i in range(num_block - 1):
#         layers.append(ConvBlock(n_filters, n_filters, shortcut))
#     return nn.Sequential(*layers)
#
#
# def get_downsample(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#         nn.BatchNorm1d(out_channels)
#     )
#
#
# class VDCNN(nn.Module):
#
#     def __init__(self, num_blocks, num_embedding=69, num_classes=4, shortcut=True):
#         super(VDCNN, self).__init__()
#
#         self.embed = nn.Embedding(num_embedding, 16, padding_idx=0)
#         self.features = make_layers(num_blocks, shortcut)
#         self.first_conv = nn.Conv1d(16, 64, kernel_size=3, padding=1)
#
#         self.block0 = get_block(in_channels=64, n_filters=64,
#                                 num_block=num_blocks[0], shortcut=shortcut)
#         self.block1 = get_block(in_channels=64, n_filters=128,
#                                 num_block=num_blocks[1], shortcut=shortcut,
#                                 downsample=get_downsample(64, 128))
#         self.block2 = get_block(in_channels=128, n_filters=256,
#                                 num_block=num_blocks[2], shortcut=shortcut,
#                                 downsample=get_downsample(128, 256))
#         self.block3 = get_block(in_channels=256, n_filters=512,
#                                 num_block=num_blocks[3], shortcut=shortcut,
#                                 downsample=get_downsample(256, 512))
#
#         self.features = nn.Sequential(
#             self.first_conv,
#             self.block0, nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
#             self.block1, nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
#             self.block2, nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
#             self.block3,
#             nn.AdaptiveMaxPool1d(8),
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 2048), nn.ReLU(),
#             nn.Linear(2048, 2048), nn.ReLU(),
#             nn.Linear(2048, num_classes)
#         )
#
#         self.__init_weights()
#
#     def __init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#
#     def forward(self, x):
#         output = self.embed(x)
#         output = output.transpose(1, 2)
#         output = self.features(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc(output)
#         return output
#
#
# def make_layers(num_blocks, shortcut):
#     layers = [nn.Conv1d(16, 64, kernel_size=3, padding=1), ConvBlock(64, 64, shortcut)]
#
#     # block1
#     for i in range(num_blocks[0] - 1):
#         layers.append(ConvBlock(64, 64, shortcut))
#     layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
#
#     # block2
#     layers.append(ConvBlock(64, 128, shortcut,
#                             downsample=nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False),
#                                                      nn.BatchNorm1d(128))))
#     for i in range(num_blocks[1] - 1):
#         layers.append(ConvBlock(128, 128, shortcut))
#     layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
#
#     # block3
#     layers.append(ConvBlock(128, 256, shortcut,
#                             downsample=nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False),
#                                                      nn.BatchNorm1d(256))))
#     for i in range(num_blocks[2] - 1):
#         layers.append(ConvBlock(256, 256, shortcut))
#     layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
#
#     # block4
#     layers.append(ConvBlock(256, 512, shortcut,
#                             downsample=nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False),
#                                                      nn.BatchNorm1d(512))))
#     for i in range(num_blocks[3] - 1):
#         layers.append(ConvBlock(512, 512, shortcut))
#
#     layers.append(nn.AdaptiveMaxPool1d(8))
#     return nn.Sequential(*layers)
#
#
# def VDCNN_9(shortcut=True):
#     return VDCNN(cfg['A'], shortcut=shortcut)
#
#
# def VDCNN_17(shortcut=True):
#     return VDCNN(cfg['B'], shortcut=shortcut)
#
#
# def VDCNN_29(shortcut=True):
#     return VDCNN(cfg['C'], shortcut=shortcut)
#
#
# def VDCNN_49(shortcut=True):
#     return VDCNN(cfg['D'], shortcut=shortcut)
