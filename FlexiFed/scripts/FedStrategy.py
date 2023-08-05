import copy
from scripts.utils import get_common_base_layers


def skip_params(name):
    keywords = ['classifier', 'fc']
    if keywords[0] in name or keywords[1] in name:
        return True
    else:
        return False


def common_basic(model_list):
    commonList = get_common_base_layers(model_list)
    device_num = len(model_list)
    for name in commonList:

        if skip_params(name):
            continue

        comWeight = copy.deepcopy(model_list[0][name])
        for i in range(1, device_num):
            comWeight += model_list[i][name]
        comWeight = comWeight / device_num

        for i in range(device_num):
            model_list[i][name] = comWeight

    return len(commonList), model_list


def common_cluster(model_list):
    common_base_len, model_list = common_basic(model_list)

    groups = 4
    start = 0
    one_group = len(model_list) // groups

    for _ in range(groups):

        weight_name_list = [s for s in model_list[start].keys()]

        for k in range(common_base_len, len(weight_name_list)):
            name = weight_name_list[k]

            if skip_params(name):
                continue

            comWeight = copy.deepcopy(model_list[start][name])
            for idx in range(start + 1, start + one_group):
                comWeight += model_list[idx][name]
            comWeight = comWeight / one_group

            for idx in range(start, start + one_group):
                model_list[idx][name] = comWeight

        start += one_group

    return model_list


def common_max(model_list):
    device_num = len(model_list)
    backup = copy.deepcopy(model_list)
    count = [[] for _ in range(device_num)]

    for i in range(device_num):
        weight_name_list = [s for s in model_list[i].keys()]
        count[i] = [1] * len(weight_name_list)

    for i in range(device_num):

        weight_name_list1 = [s for s in model_list[i].keys()]  # 第i个模型

        for j in range(i + 1, device_num):

            if i == j:
                continue

            weight_name_list2 = [s for s in model_list[j].keys()]

            for k in range(len(weight_name_list1)):

                if weight_name_list2[k] == weight_name_list1[k]:
                    name = weight_name_list1[k]

                    if skip_params(name):
                        continue

                    model_list[i][name] += backup[j][name]
                    model_list[j][name] += backup[i][name]
                    count[i][k] += 1
                    count[j][k] += 1
                else:
                    break

    for c in range(device_num):
        weight_name_list = [s for s in model_list[c].keys()]
        for k in range(len(weight_name_list)):
            model_list[c][weight_name_list[k]] = model_list[c][weight_name_list[k]].cpu() / count[c][k]

    return model_list
