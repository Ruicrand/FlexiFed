# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import sys
import csv
import torch
from torch.utils.data import Dataset

csv.field_size_limit(sys.maxsize)

ALPHABET = [
    "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
    "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
]
MAX_LEN = [1014, 1024]


class AGNewsDataset(Dataset):
    def __init__(self, data_path, is_VDCNN=0):
        self.data_path = data_path
        self.vocabulary = list(ALPHABET[is_VDCNN])
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = MAX_LEN[is_VDCNN]
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        raw_text = self.texts[index]
        data = torch.zeros(self.max_length, dtype=torch.long)

        for idx, char in enumerate(raw_text):
            if idx == self.max_length:
                break
            if char in self.vocabulary:
                data[idx] = self.vocabulary.index(char) + 1
        label = torch.tensor(self.labels[index])

        return data, label
