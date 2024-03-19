import os
import cv2
import numpy as np


def convert_folder_to_8bits(path_to_folder, clip_percentile=0.1):
    histogram = np.zeros(2 ** 16)
    for img in os.listdir(path_to_folder):
        full_img_path = os.path.join(path_to_folder, img)
        img16 = cv2.imread(full_img_path, -1)
        histogram = np.add(histogram, cv2.calcHist(img16, [0], None, [2 ** 16], [0, 2 ** 16]).flatten())

    a = 0
    b = 255
    c = percentile(histogram, clip_percentile)
    d = percentile(histogram, 100 - clip_percentile)

    print(f'a{a}, b{b}, c{c}, d{d}')
    process_im(path_to_folder, a, b, c, d)


def percentile(histogram, q):
    total = 0
    qindex = sum(histogram) * q/100
    for i, value in enumerate(histogram):
        total = total + value
        if total > qindex:
            return i


def process_im(path, a, b, c, d):
    for img in os.listdir(path):
        full_img_path = os.path.join(path, img)
        img16 = cv2.imread(full_img_path, -1).astype(np.float32)
        img8 = (img16 - c) * ((b - a) / (d - c)) + a
        rescale = np.clip(img8, 0, 255, out=None)
        rescale = rescale.astype('uint8')
        cv2.imwrite(full_img_path, rescale)


