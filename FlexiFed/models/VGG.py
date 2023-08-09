import math
import torch.nn as nn

__all__ = [
    'VGG',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
]

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
        mul = 1
        if num_classes == 30:
            mul = 4
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * mul, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

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


def vgg11_bn(in_channels=3, num_classes=10):
    return VGG(make_layers(cfg['A'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg13_bn(in_channels=3, num_classes=10):
    return VGG(make_layers(cfg['B'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg16_bn(in_channels=3, num_classes=10):
    return VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)


def vgg19_bn(in_channels=3, num_classes=10):
    return VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels), num_classes=num_classes)