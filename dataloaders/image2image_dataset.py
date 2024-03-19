import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data


class image2imagedataset(data.Dataset):
    def __init__(self, dataroot, inputs, target, image_names, isTrain=True):

        self.dataroot = dataroot
        self.inputs = inputs
        self.target = target
        self.isTrain = isTrain

        self.img_names = image_names

    def transform(self, image, mask):
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if self.isTrain:

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        image = TF.normalize(image, (0.5,), (0.5,))
        mask = TF.normalize(mask, (0.5,), (0.5,))

        return image, mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        image_name = self.img_names[idx]

        in_chs = []
        for ch_name in self.inputs:
            x_ch = cv2.imread(os.path.join(self.dataroot, ch_name, image_name), cv2.IMREAD_GRAYSCALE)
            in_chs.append(x_ch)

        x = np.stack(in_chs, axis=2)
        y = cv2.imread(os.path.join(self.dataroot, self.target, image_name), cv2.IMREAD_GRAYSCALE)

        # if eval not target:
        if not self.isTrain and y is None:
            y = x[:, :, 0]

        x, y = self.transform(x, y)

        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)

        return {'A': x, 'B': y, 'img': self.img_names[idx]}
