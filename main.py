import copy
import torch
import warnings

from models.VDCNN import *
from models.VGG import *
from scripts.fed import common_cluster, common_basic, common_max
from scripts.predict import predict
from scripts.train import client_local_train
from scripts.utils import get_dataset

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    device = torch.device('mps')

    # 设备数量
    device_num = 40

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset('agnews', device_num)

    # Training
    epochs = 35
    num_classes = 30
    lr = 0.01

    uid = [_ for _ in range(device_num)]
    modelAccept = {_: None for _ in range(device_num)}  # client的参数列表
    local_acc = [[] for _ in range(device_num)]

    for epoch in range(epochs):

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for idx in range(device_num):

            if idx < 10:
                # model = vgg11_bn(in_channels=1, num_classes=num_classes)
                model = VDCNN_9()
            elif 10 <= idx < 20:
                # model = vgg13_bn(in_channels=1, num_classes=num_classes)
                model = VDCNN_17()
            elif 20 <= idx < 30:
                # model = vgg16_bn(in_channels=1, num_classes=num_classes)
                model = VDCNN_29()
            else:
                # model = vgg19_bn(in_channels=1, num_classes=num_classes)
                model = VDCNN_49()
            if epoch != 0:
                model.load_state_dict(modelAccept[idx])

            # if epoch == 2:
            #     lr = 0.0001

            # 保存client进行local_train后的模型参数
            modelAccept[idx] = copy.deepcopy(client_local_train(model, train_dataset, user_groups[idx], device))

            # 预测
            acc = predict(model, test_dataset, user_groups_test[idx], device)
            local_acc[idx].append(acc)
            print(local_acc[idx])

        # modelAccept = common_max(modelAccept)
        # modelAccept = common_cluster(modelAccept)
        _, modelAccept = common_basic(modelAccept)

        # 输出4种不同模型的正确率
        res = [[] for i in range(4)]
        for i in range(4):
            res[i] = [0 for i in range(len(local_acc[0]))]

        for j in range(len(local_acc[0])):
            start = 0
            for type in range(4):
                sum = 0.0
                for i in range(start, start + 10):
                    sum += local_acc[i][j]
                res[type][j] = round(sum / 10.0, 2)

                start += 10

        print(res)
