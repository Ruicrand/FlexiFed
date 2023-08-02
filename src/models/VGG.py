import math
from pprint import pprint

import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary

__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

from scripts.utils import get_common_base_layers

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(in_channels=3, num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], in_channels=in_channels), num_classes=num_classes)


def vgg11_bn(in_channels=3, num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg13(in_channels=3, num_classes=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], in_channels=in_channels), num_classes=num_classes)


def vgg13_bn(in_channels=3, num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg16(in_channels=3, num_classes=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], in_channels=in_channels), num_classes=num_classes)


def vgg16_bn(in_channels=3, num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg19(in_channels=3, num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], in_channels=in_channels), num_classes=num_classes)


def vgg19_bn(in_channels=3, num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


# if __name__ == '__main__':
#     model = [vgg11_bn().state_dict(), vgg13_bn().state_dict(), vgg16_bn().state_dict(), vgg19_bn().state_dict()]
#
#     pprint(get_common_base_layers(model))