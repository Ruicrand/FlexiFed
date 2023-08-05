import argparse
from scripts.GlobalTrain import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--strategy', type=str)
    parser.add_argument("-d", '--data_set', type=str)
    parser.add_argument("-t", '--save_period', type=int, default=10)
    parser.add_argument("-lr", '--lr', type=float, default=0.01)
    parser.add_argument("-w", '--weight_decay', type=float, default=5e-4)
    parser.add_argument("-epochs", '--epochs', type=int, default=252)
    parser.add_argument("-visual", '--visual', type=int, default=1)
    parser.add_argument("-i", '--in_channel', type=int, default=3)
    parser.add_argument("-classes", '--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args


def train(opt):
    global_train(opt)


class OPT:
    def __init__(self, opt):
        self.strategy = opt['strategy']
        self.data_set = opt['data_set']
        self.save_period = opt['save_period']
        self.lr = opt['lr']
        self.weight_decay = opt['weight_decay']
        self.visual = opt['visual']
        self.epochs = opt['epochs']
        self.in_channel = opt['in_channel']
        self.num_classes = opt['num_classes']


if __name__ == "__main__":
    # opt = get_args()
    opt = {
        'strategy': 'Basic',
        'data_set': 'speechcommands',
        'save_period': 1,
        'lr': 0.0001,
        'weight_decay': 1e-2,
        'visual': 1,
        'epochs': 252,
        'in_channel': 1,
        'num_classes': 30
    }
    train(OPT(opt))

    opt['strategy'] = 'Clustered'
    train(OPT(opt))

    opt['strategy'] = 'Max'
    train(OPT(opt))
