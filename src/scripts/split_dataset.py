"""Splits the google speech commands into train, validation and test sets.
"""

import os
import shutil
from torchaudio.datasets import SPEECHCOMMANDS


def move_files(src_folder, to_folder, list_file):
    with open(list_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            dirname = os.path.dirname(line)
            dest = os.path.join(to_folder, dirname)
            if not os.path.exists(dest):
                os.mkdir(dest)
            shutil.move(os.path.join(src_folder, line), dest)


if __name__ == '__main__':
    # train = SPEECHCOMMANDS(root='/Users/chenrui/Desktop/课件/REPO/edge computing/project/data', url='speech_commands_v0.01', download=True, subset='training')

    audio_folder = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands/speech_commands_v0.01'
    root = '/Users/chenrui/Desktop/课件/REPO/edge computing/project/data/SpeechCommands'

    validation_path = os.path.join(audio_folder, 'validation_list.txt')
    test_path = os.path.join(audio_folder, 'testing_list.txt')

    test_folder = os.path.join(root, 'test')
    train_folder = os.path.join(root, 'train')
    os.mkdir(test_folder)

    move_files(audio_folder, test_folder, test_path)
    move_files(audio_folder, test_folder, validation_path)
    os.rename(audio_folder, train_folder)
