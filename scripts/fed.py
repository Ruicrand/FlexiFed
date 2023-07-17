import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.utils import get_common_base_layers


def skip_batchnorm_params(name):
    skip_keywords = ['running_mean', 'running_var', 'num_batches_tracked', 'classifier']
    for keyword in skip_keywords:
        if keyword in name:
            return True
    return False


def common_basic(model_list):
    commonList = get_common_base_layers(model_list)

    # 聚合local models
    for k in range(len(commonList)):

        name = commonList[k]

        # 跳过bachNormal层
        if skip_batchnorm_params(name):
            continue
        if k + 1 < len(commonList) and skip_batchnorm_params(commonList[k + 1]):
            continue
        if k + 2 < len(commonList) and skip_batchnorm_params(commonList[k + 2]):
            continue

        comWeight = copy.deepcopy(model_list[0][name])
        for i in range(1, len(model_list)):
            comWeight += model_list[i][name]
        comWeight = torch.div(comWeight, len(model_list))

        for i in range(0, len(model_list)):
            model_list[i][name] = comWeight

    return len(commonList), model_list


def common_cluster(model_list):
    common_base_len, model_list = common_basic(model_list)

    group = 4
    num = int(len(model_list) / 4)
    start = 0

    # 按照模型架构分类
    for type in range(group):

        weight_name_list = [s for s in model_list[start].keys()]

        for k in range(common_base_len, len(weight_name_list)):
            name = weight_name_list[k]

            # 跳过bachNormal层和classifer
            if skip_batchnorm_params(name):
                continue
            if k + 1 < len(weight_name_list) and skip_batchnorm_params(weight_name_list[k + 1]):
                continue
            if k + 2 < len(weight_name_list) and skip_batchnorm_params(weight_name_list[k + 2]):
                continue

            comWeight = copy.deepcopy(model_list[start][name])
            for idx in range(start + 1, start + num):
                comWeight += model_list[idx][name]

            comWeight = torch.div(comWeight, num)

            for idx in range(start, start + num):
                model_list[idx][name] = comWeight

        start += num

    return model_list


def common_max(model_list):
    backup = copy.deepcopy(model_list)

    count = [[] for _ in range(len(model_list))]

    for i in range(len(model_list)):
        weight_name_list = [s for s in model_list[i].keys()]
        count[i] = [1 for _ in range(len(weight_name_list))]

    for i in range(0, len(model_list)):

        weight_name_list1 = [s for s in model_list[i].keys()]  # 第i个模型

        for j in range(i + 1, len(model_list)):

            if i == j:
                continue

            weight_name_list2 = [s for s in model_list[j].keys()]
            # 能共享就共享
            for k in range(0, len(weight_name_list1)):

                if weight_name_list2[k] == weight_name_list1[k]:
                    name = weight_name_list1[k]

                    # 跳过bachNormal层和classifer
                    if skip_batchnorm_params(name):
                        continue
                    if k + 1 < len(weight_name_list1) and skip_batchnorm_params(weight_name_list1[k + 1]):
                        continue
                    if k + 2 < len(weight_name_list1) and skip_batchnorm_params(weight_name_list1[k + 2]):
                        continue

                    model_list[i][name] += backup[j][name]
                    model_list[j][name] += backup[i][name]
                    count[i][k] += 1
                    count[j][k] += 1
                else:
                    break

    for c in range(0, len(model_list)):
        weight_name_list = [s for s in model_list[c].keys()]
        for k in range(0, len(weight_name_list)):
            model_list[c][weight_name_list[k]] = model_list[c][weight_name_list[k]].cpu() / count[c][k]

    return model_list