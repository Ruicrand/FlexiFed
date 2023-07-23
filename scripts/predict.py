import torch
import numpy as np
from scripts.utils import DatasetSplit
from torch.utils.data import DataLoader


def predict(net, dataset, idxs, device):
    net.to(device)

    total, correct = 0.0, 0.0
    test_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=False)

    net.eval()

    with torch.no_grad():

        for _, batch in enumerate(test_loader):

            # inputs = batch['input']
            # inputs = torch.unsqueeze(inputs, 1)
            # targets = batch['target']
            inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predict = torch.max(outputs, 1)

            predict = predict.view(-1)  # 可能无用
            correct += predict.eq(targets).sum().item()  # 预测正确个数
            total += targets.size(0)

    acc = correct * 100.0 / total
    acc = round(acc, 2)

    return acc