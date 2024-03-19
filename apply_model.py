import os

import numpy as np

from dataloaders.image2image_dataset import image2imagedataset
from vl_framework.runners import test_resvit

def apply_model(dataroot, experiment):
    markers = experiment.split(' ')[-1]
    inputs, target = markers.split('-')
    inputs = inputs.split('_')

    checkpoints = r'./results'
    target_images = np.array(os.listdir(os.path.join(dataroot, inputs[0])))

    save_dir = os.path.join(checkpoints, experiment)
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = image2imagedataset(dataroot, inputs, target, target_images, isTrain=False)
    test_resvit(experiment, checkpoints, test_dataset, len(inputs), deployment=True)
