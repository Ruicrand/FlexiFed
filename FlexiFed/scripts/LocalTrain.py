import torch
import torch.nn as nn
from tqdm import tqdm
from scripts.utils import DatasetSplit
from torch.utils.data import DataLoader


def client_local_train(net, dataset, idxs, device, lr=0.01, weight_decay=5e-4, epochs=10):

    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    train_dataset = DatasetSplit(dataset, idxs)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):

        for batch in train_loader:

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return net.state_dict()


# for speech commands, lr = 1e-4, weight_decay=1e-2
def client_local_train_1(net, dataset, idxs, device, lr=0.03, weight_decay=5e-4, epochs=10):

    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    train_dataset = DatasetSplit(dataset, idxs)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    pbar = tqdm(range(epochs), unit='epoch', unit_scale=1)

    for epoch in pbar:

        correct, total = 0, 0

        for batch in train_loader:

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs).to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predict = outputs.max(1)
            total += targets.size(0)
            correct += predict.eq(targets).sum().item()

        # update the progress bar
        pbar.set_postfix({'acc': "%.02f%%" % (100 * correct / total)})

    return net.state_dict()
