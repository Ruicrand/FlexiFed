import copy
import os
import numpy as np
import torch
import warnings

from tqdm import tqdm
from models.CharCNN import *
from models.ResNet import *
from models.VDCNN import *
from models.VGG import *
from scripts.draw import plot_2d_list
from scripts.FedStrategy import common_cluster, common_basic, common_max
from scripts.predict import predict
from scripts.LocalTrain import client_local_train, client_local_train_1
from scripts.utils import get_dataset

warnings.filterwarnings("ignore")

Data_dir = {
    'cifar10': '/root/autodl-tmp/cifar10',
    'cinic10': '/root/autodl-tmp/cinic10',
    'speechcommands': '/root/autodl-tmp/SpeechCommands.h5',
    'agnews': '/root/autodl-tmp/agnews'
}


def global_train(opt):
    device = torch.device('cuda')
    model = 'VDCNN'

    save_period = opt.save_period
    strategy = opt.strategy
    data_set = opt.data_set
    visual = opt.visual

    title = strategy + '-Common'
    res_dir = os.path.join('',
                           data_set, model, title)

    predict_period = 1
    device_num = 8
    one_group = device_num // 4

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(dataset_name=data_set,
                                                                             device_num=device_num,
                                                                             data_dir=Data_dir[data_set],
                                                                             is_VDCNN=model == 'VDCNN')

    # Training
    lr = opt.lr
    weight_decay = opt.weight_decay
    epochs = opt.epochs
    in_channel = opt.in_channel
    num_classes = opt.num_classes

    modelAccept = {_: None for _ in range(device_num)}  # client的参数列表
    local_acc = [[] for _ in range(device_num)]

    st = 0
    data_client = len(user_groups[0])
    local_dataLength = data_client // 10

    for epoch in range(epochs):

        ed = st + local_dataLength

        for idx in range(device_num):

            if idx < 2:
                # model = vgg11_bn(in_channels=in_channel, num_classes=num_classes)
                # model = resnet20(in_channels=in_channel, num_classes=num_classes)
                model = VDCNN_9()
                # model = CharCNN_3()

            elif 2 <= idx < 4:
                # model = vgg13_bn(in_channels=in_channel, num_classes=num_classes)
                # model = resnet32(in_channels=in_channel, num_classes=num_classes)
                model = VDCNN_17()
                # model = CharCNN_4()

            elif 4 <= idx < 6:
                # model = vgg16_bn(in_channels=in_channel, num_classes=num_classes)
                # model = resnet44(in_channels=in_channel, num_classes=num_classes)
                model = VDCNN_29()
                # model = CharCNN_5()

            else:
                # model = vgg19_bn(in_channels=in_channel, num_classes=num_classes)
                # model = resnet56(in_channels=in_channel, num_classes=num_classes)
                model = VDCNN_49()
                # model = CharCNN_6()

            if epoch != 0:
                model.load_state_dict(modelAccept[idx])

            idx_train_all = list(user_groups[idx])
            idx_train_batch = set(idx_train_all[st:ed])

            modelAccept[idx] = copy.deepcopy(client_local_train_1(model, train_dataset,
                                                                  idx_train_batch, device,
                                                                  lr=lr, weight_decay=weight_decay))

            acc = predict(model, test_dataset, user_groups_test[idx], device)
            local_acc[idx].append(acc)

        if strategy == 'Basic':
            _, modelAccept = common_basic(modelAccept)

        elif strategy == 'Clustered':
            modelAccept = common_cluster(modelAccept)

        else:
            modelAccept = common_max(modelAccept)

        if epoch % save_period == 0:

            start = 0
            res = np.zeros((4, len(local_acc[0])))

            for i in range(4):
                res[i] = np.array(local_acc[start:start + one_group]).sum(axis=0)
                (res[i] / one_group).round(4)
                start += one_group

            if visual:
                plot_2d_list(res, predict_period=predict_period, title=title)

            np.save(res_dir, res)
