import copy
import os
from pprint import pprint

import numpy as np
import torch
import warnings

from tqdm import tqdm

from models.CharCNN import *
from models.ResNet import *
from models.VDCNN import *
from models.VGG import *
from scripts.draw import plot_2d_list
from scripts.fed import common_cluster, common_basic, common_max
from scripts.predict import predict
from scripts.train import client_local_train, client_local_train_1
from scripts.utils import get_dataset

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    device = torch.device('mps')

    # 设备数量
    device_num = 40

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset('agnews', device_num, is_VDCNN=0)

    # Training
    epochs = 500

    uid = [_ for _ in range(device_num)]
    modelAccept = {_: None for _ in range(device_num)}  # client的参数列表
    local_acc = [[] for _ in range(device_num)]

    st = 0
    data_client = len(user_groups[0])
    local_dataLength = data_client / 10
    predict_period = 10
    title = 'Basic-common'
    res_dir = '/root/result/cifar/VGG/Clustered-common.np'

    in_channel = 3
    num_classes = 10

    for epoch in tqdm(range(epochs)):

        ed = st + local_dataLength

        for idx in range(device_num):

            if idx < 10:
                # model = vgg11_bn(in_channels=in_channel, num_classes=num_classes)
                model = resnet20(in_channels=in_channel, num_classes=num_classes)
                # model = VDCNN_9()
                # model = CharCNN_3()
            elif 10 <= idx < 20:
                # model = vgg13_bn(in_channels=in_channel, num_classes=num_classes)
                model = resnet32(in_channels=in_channel, num_classes=num_classes)
                # model = VDCNN_17()
                # model = CharCNN_4()
            elif 20 <= idx < 30:
                # model = vgg16_bn(in_channels=in_channel, num_classes=num_classes)
                model = resnet44(in_channels=in_channel, num_classes=num_classes)
                # model = VDCNN_29()
                # model = CharCNN_5()
            else:
                # model = vgg19_bn(in_channels=in_channel, num_classes=num_classes)
                model = resnet56(in_channels=in_channel, num_classes=num_classes)
                # model = VDCNN_49()
                # model = CharCNN_6()

            if epoch != 0:
                model.load_state_dict(modelAccept[idx])

            idx_train_all = list(user_groups[idx])
            idx_train_batch = set(idx_train_all[int(st):int(ed)])
            modelAccept[idx] = copy.deepcopy(client_local_train_1(model, train_dataset, idx_train_batch, device))

            if epoch % predict_period == 0:
                acc = predict(model, test_dataset, user_groups_test[idx], device)
                local_acc[idx].append(acc)

        st = ed % data_client

        # modelAccept = common_max(modelAccept)
        # modelAccept = common_cluster(modelAccept)
        _, modelAccept = common_basic(modelAccept)

        if epoch % predict_period == 0:

            # 输出4种不同模型的正确率
            res = [[] for _ in range(4)]
            for i in range(4):
                res[i] = [0 for _ in range(len(local_acc[0]))]

            for j in range(len(local_acc[0])):
                start = 0
                for arch in range(4):
                    sum = 0.0
                    for i in range(start, start + 10):
                        sum += local_acc[i][j]
                    res[arch][j] = round(sum / 10.0, 2)

                    start += 10

            plot_2d_list(res, predict_period=predict_period, title=title)

            tmp = np.array(res)
            np.save(res_dir, tmp)


