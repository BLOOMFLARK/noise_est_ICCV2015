#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-07 15:13:19

import numpy as np
import sys
from math import floor

def im2patch(image, patch_s, stride=1):
    '''
    Transform image to patches.
    Input:
        image: 3 x H x W or 1 X H x W image, numpy format
        patch_s: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(patch_s, tuple):
        patch_H, patch_W = patch_s
    elif isinstance(patch_s, int):
        patch_H = patch_W = patch_s
    else:
        sys.exit('The input of patch_s must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    
    C, H, W = image.shape
    num_H = len(range(0, H - patch_H + 1, stride_H))
    num_W = len(range(0, W - patch_W + 1, stride_W))
    n_patch = num_H * num_W
    patch = np.zeros((C, patch_H * patch_W, n_patch), dtype=image.dtype)
    k = 0
    for i in range(patch_H):
        for j in range(patch_W):

            temp = image[:, i: H - patch_H + i + 1 : stride_H, j: W - patch_W + j + 1: stride_W]
            # в k строку записываем patch
            patch[:, k, :] = temp.reshape((C, n_patch))
            k += 1

    return patch.reshape((C, patch_H, patch_W, n_patch))

def im2double(im):
    '''
    Input:
        im: numpy uint format image, RGB or Gray, 
    '''

    im = im.astype(np.float)
    # flatten()
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())

    out = (im - min_val) / (max_val - min_val)

    return out
