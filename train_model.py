import os
import time

import numpy as np
from sklearn.model_selection import KFold

from dataloaders.image2image_dataset import image2imagedataset

from vl_framework.runners import train_resvit, test_resvit


def train_resvit_vl(dataroot, inputs, target):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = timestr + ' ' + '_'.join(inputs) + '-' + target
    checkpoints = r'./results'
    target_images = np.array(os.listdir(os.path.join(dataroot, target)))

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for train_index, test_index in kf.split(target_images):

        fold_folder = experiment_name
        save_dir = os.path.join(f'{checkpoints}', f'{fold_folder}')
        os.makedirs(save_dir, exist_ok=True)

        train_dataset = image2imagedataset(dataroot, inputs, target, target_images[train_index])
        test_dataset = image2imagedataset(dataroot, inputs, target, target_images[test_index], isTrain=False)

        log = open(os.path.join(save_dir, "trainset.txt"), "w")
        log.write('\n'.join(target_images[train_index]))

        log = open(os.path.join(save_dir, "testset.txt"), "w")
        log.write('\n'.join(target_images[test_index]))
        log.flush()

        train_resvit(fold_folder, checkpoints, train_dataset, len(inputs))
        test_resvit(fold_folder, checkpoints, test_dataset, len(inputs))

        break
    return experiment_name
