import torch
from scripts.utils import DatasetSplit
from torch.utils.data import DataLoader


def predict(net, dataset, idxs, device):

    total, correct = 0.0, 0.0
    test_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=False)

    net.to(device)
    net.eval()

    with torch.no_grad():
        for images, targets in test_loader:

            images, targets = images.to(device), targets.to(device)

            outputs = net(images)
            _, predict = torch.max(outputs, 1)
            predict = predict.view(-1)
            
            correct += predict.eq(targets).sum().item()
            total += len(targets)

    acc = correct / total
    return round(acc, 2)
