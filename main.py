import copy
import torch
import warnings
from models.VGG import *
from scripts import predict, train, utils, fed

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设备数量
    device_num = 40

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, user_groups_test = utils.get_dataset("cifar10", device_num)

    # Training
    epochs = 32

    uid = [_ for _ in range(device_num)]
    modelAccept = {_: None for _ in range(device_num)}
    local_acc = [[] for _ in range(device_num)]

    # 不同的设备采用不同架构的网络
    for idx in uid:
        if idx < 10:
            modelAccept[idx] = vgg11_bn()

        elif 10 <= idx < 20:
            modelAccept[idx] = vgg13_bn()

        elif 20 <= idx < 30:
            modelAccept[idx] = vgg16_bn()

        else:
            modelAccept[idx] = vgg19_bn()

    for epoch in range(epochs):

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for idx in range(device_num):

            train_all = user_groups[idx]

            if epoch == 0:
                model = modelAccept[idx]

            else:
                if idx < 10:
                    model = vgg11_bn()

                elif 10 <= idx < 20:
                    model = vgg13_bn()

                elif 20 <= idx < 30:
                    model = vgg16_bn()

                else:
                    model = vgg19_bn()

                model.load_state_dict(modelAccept[idx])

            modelAccept[idx] = copy.deepcopy(train.client_local_train(model, train_dataset, train_all, device))

            # 预测
            acc = predict.predict(model, test_dataset, user_groups_test[idx], device)
            local_acc[idx].append(acc)
            print(local_acc[idx])

        for idx in uid:

            train_all = user_groups[idx]
            test_all = user_groups_test[idx]

            if epoch == 0:
                model = modelAccept[idx]

            else:
                if idx < 10:
                    model = vgg11_bn()

                elif 10 <= idx < 20:
                    model = vgg13_bn()

                elif 20 <= idx < 30:
                    model = vgg16_bn()

                else:
                    model = vgg19_bn()

                model.load_state_dict(modelAccept[idx])

            modelAccept[idx] = copy.deepcopy(train.client_local_train(model, train_dataset, train_all, device))

            # 预测
            acc = predict.predict(model, test_dataset, test_all, device)
            local_acc[idx].append(acc)
            print(local_acc[idx])

        # modelAccept = common_max(modelAccept)
        # modelAccept = common_cluster(modelAccept)
        _, modelAccept = fed.common_basic(modelAccept)

        # 输出4种不同模型的正确率
        res = [[] for _ in range(4)]
        for i in range(4):
            res[i] = [0 for _ in range(len(local_acc[0]))]

        for j in range(len(local_acc[0])):
            start = 0
            for type in range(4):
                sum = 0.0
                for i in range(start, start + 10):
                    sum += local_acc[i][j]
                res[type][j] = round(sum / 10, 2)

                start += 10

        print(res)
