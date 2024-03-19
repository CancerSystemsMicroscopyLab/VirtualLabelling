import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from models import create_model


def mse(a, b):
    return np.square(np.subtract(a,b)).mean()


def L1(a, b):
    return np.abs(np.subtract(a,b)).mean()


def test(opt, dataset, deployment):
    model = create_model(opt)

    test_results_dir = os.path.join(opt.checkpoints_dir, opt.name)
    predictions_dir = 'predictions'
    test_dir = 'test_images'

    if not os.path.exists(os.path.join(test_results_dir, predictions_dir)):
        os.mkdir(os.path.join(test_results_dir, predictions_dir))

    if not os.path.exists(os.path.join(test_results_dir, test_dir)):
        os.mkdir(os.path.join(test_results_dir, test_dir))

    log = open(os.path.join(test_results_dir, f"results.csv"), "w")
    log.write('img,mean_intensity,L1,L2,SSIM,PCC\n')

    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()

        img_name = data['img']

        visuals = model.get_current_visuals()

        if deployment:
            fake_B = visuals['fake_B'][:, :, 0]
            cv2.imwrite(os.path.join(test_results_dir, predictions_dir, img_name), fake_B)

        else:
            if not os.path.exists(os.path.join(test_results_dir, test_dir)):
                os.mkdir(os.path.join(test_results_dir, test_dir))

            img_root, img_ext = img_name.split(".")

            for in_ch in range(opt.input_nc):
                real_A = visuals['real_A'][:,:,in_ch]
                real_A = real_A.astype(np.uint8)
                cv2.imwrite(os.path.join(test_results_dir, test_dir, f'{img_root}_input{in_ch}.{img_ext}'), real_A)

            real_B = visuals['real_B'][:, :, 0]
            fake_B = visuals['fake_B'][:, :, 0]

            cv2.imwrite(os.path.join(test_results_dir, test_dir, f'{img_root}_target.{img_ext}'), real_B)
            cv2.imwrite(os.path.join(test_results_dir, test_dir, f'{img_root}_predicted.{img_ext}'), fake_B)

            real_mean_intensity = np.mean(real_B.flatten())
            l1_loss = L1(real_B, fake_B)
            mse_loss = mse(real_B, fake_B)
            ssim_loss = ssim(real_B, fake_B)
            pcc = np.corrcoef(real_B.flatten(), fake_B.flatten())[0][1]

            corrected_name = img_name
            log.write(f"{corrected_name},{real_mean_intensity},{l1_loss},{mse_loss},{ssim_loss},{pcc}\n")


        print('%04d: process image... %s' % (i, img_name))

