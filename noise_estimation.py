#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-07 14:36:55


import numpy as np
import cv2 as cv
from cv2 import imread
from utils import im2patch, im2double
import time

def noise_estimate(image, patch_s=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if image.ndim == 3:
    	# H * W * 3 -> 3 x H x W
        image = image.transpose((2, 0, 1))
    else:
    	# [1,2] -> [[1,2]]
        image = np.expand_dims(image, axis=0)

    # image to patch
    patch = im2patch(image, patch_s, 3)  # C x pch_size x pch_size x num_pch tensor
    n_patch = patch.shape[3]
    patch = patch.reshape((-1, n_patch))  # d x num_pch matrix
    d = patch.shape[0]

    mu = patch.mean(axis=1, keepdims=True)  # d x 1
    X = patch - mu
    sigma_X = np.matmul(X, X.transpose()) / n_patch

    eigenvalues, _ = np.linalg.eigh(sigma_X)
    eigenvalues.sort()

    for i in range(-1, -d - 1, -1):
        tau = np.mean(eigenvalues[:i])
        if np.sum(eigenvalues[:i] > tau) == np.sum(eigenvalues[:i] < tau):
            return np.sqrt(tau)


if __name__ == '__main__':
    image = imread('./lena.png')
    image = im2double(image)
    #cv.imshow("Image", image)
    #cv.waitKey(0)

    noise_level = [5, 15, 20, 30, 40]

    for level in noise_level:
        sigma = level / 255

        im_noise = image + np.random.randn(*image.shape) * sigma

        #if level == 40:
        #    cv.imshow('Noisy',im_noise)
        #    cv.waitKey(0)

        start = time.time()
        est_level = noise_estimate(im_noise)
        end = time.time()
        time_elapsed = end - start

        str_p = "Time: {0:.4f}, Ture Level: {1:6.4f}, Estimated Level: {2:6.4f}"
        print(str_p.format(time_elapsed, level, est_level * 255))


