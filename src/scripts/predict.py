import torch
from utils import DatasetSplit
from torch.utils.data import DataLoader


def predict(net, dataset, idxs, device):
    net.to(device)
    net.eval()

    total, correct = 0.0, 0.0
    test_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=False)

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # inputs = batch['input']
            # inputs = torch.unsqueeze(inputs, 1)
            # targets = batch['target']
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predict_ = torch.max(outputs, 1)

            predict_ = predict_.view(-1)
            correct += predict_.eq(targets).sum().item()
            total += targets.size(0)

    acc = correct * 1.0 / total
    acc = round(acc, 4)

    return acc
