import copy
import torch
import torch.nn as nn
from utils import DatasetSplit
from torch.utils.data import DataLoader


# for speech commands, lr = 1e-4, weight_decay=1e-2
def client_local_train(net, dataset, idxs, device, lr=0.01, weight_decay=5e-4, epochs=10):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)

    train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=True)

    for epoch in range(epochs):

        net.train()
        correct, total = 0, 0

        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predict = outputs.max(1)
            total += targets.size(0)
            correct += predict.eq(targets).sum().item()

        # scheduler.step()

    print('Local : |  Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    return net.state_dict()
