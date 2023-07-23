import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.utils import DatasetSplit, plot_spectrogram
from torch.utils.data import DataLoader, WeightedRandomSampler


# for speech commands, lr = 1e-4, weight_decay=1e-2
def client_local_train(net, dataset, idxs, device, lr=0.01, weight_decay=5e-4, epochs=10):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    train_sample = set(np.random.choice(list(idxs), 1000, replace=False))
    train_dataset = DatasetSplit(dataset, train_sample)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    pbar = tqdm(range(epochs), unit='epoch', unit_scale=1)

    for epoch in pbar:

        net.train()
        correct, total = 0, 0

        for batch in train_loader:

            # inputs = batch['input']
            # inputs = torch.unsqueeze(inputs, 1)
            # targets = batch['target']
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
