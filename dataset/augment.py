import cv2
import random
import torch
import torchvision
import numpy as np
import torchvision.transforms as T
from PIL import Image
from easydict import EasyDict as dic


class RandomAugment:
    def __init__(self, params):
        self.aug = aug = dic(params.augment_parameters)
        self.cj = aug.colorjitter
        items = self.cj.items
        self.colorjitter = T.ColorJitter(
            brightness=items.brightness,
            contrast=items.contrast,
            saturation=items.saturation,
            hue=items.hue,
        )
        self.to_gray = T.Grayscale()
        self.src_img_mode = params.src_img_mode
        self.shift = dic(aug.shift)

    def _colorjitter(self, img):
        if np.random.rand() >= self.cj.probability:
            return img

        img = Image.fromarray(img)
        img = self.colorjitter(img)
        img = self.to_gray(img)
        return img

    def _warp_img(self, img, src, dx, dy):
        np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])

    def _shift(self, data, camera, img, aug_pts=None):
        if np.random.rand() >= self.shift.probability:
            return img, aug_pts

        src_img = data[self.src_img_mode][camera]
        h, w = img.shape[:2]
        rx = self.shift.w_ratio * w
        ry = self.shift.h_ratio * h
        mx = max(int(w * rx), 2)
        my = max(int(h * ry), 2)
        dx = random.randrange(1, mx, step=1)
        dy = random.randrange(1, my, step=1)
        dx = dx if random.random() < 0.5 else -dx
        dy = dy if random.random() < 0.5 else -dy
        img = self._warp_img(img, src_img, dx, dy)
        offset = torch.tensor([dx, dy]).repeat(4, 1).reshape(1, 4, 2)
        aug_pts = aug_pts + offset if aug_pts is not None else offset
        return img, aug_pts

    def _rotate(self, img, aug_pts=None):
        '''yaw pitch roll'''
        # TODO
        return img, aug_pts

    def __call__(self, img, data, camera):
        aug_pts = None
        img, aug_pts = self._shift(data, camera, img, aug_pts)
        img = self._colorjitter(img)

        return img, aug_pts
