import os
import cv2
import ipdb
import numpy as np
from skimage import morphology, img_as_ubyte


class FilterNonTextrueFrame(object):
    def __init__(self, coef1=5000, coef2=10, coef3=60):
        super().__init__()
        self.sobel_pixels_threshshold = coef1
        self.obj_min_pixel_filter = coef2
        self.binary_threshshold = coef3

    def __call__(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        gradx = cv2.convertScaleAbs(grad_x)
        grady = cv2.convertScaleAbs(grad_y)
        add_image = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
        _, thresh = cv2.threshold(
            src=add_image,
            thresh=self.binary_threshshold,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        thresh_bool = thresh > 0
        mask = morphology.remove_small_objects(thresh_bool, self.obj_min_pixel_filter)
        pix_sum = mask.sum()
        is_filter = False
        if mask.sum() < self.sobel_pixels_threshshold:
            is_filter = True
        return is_filter, pix_sum


def sobel_demo1(image, sv_dir):
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    # cv2.imshow('sobel_demo1_gradient_x', gradx)
    # cv2.imshow('sobel_demo1_gradient_y', grady)
    # 合并x, y两个梯度
    add_image = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # cv2.imshow('sobel_demo1_addWeighted', add_image)
    _, thresh = cv2.threshold(
        src=add_image, thresh=60, maxval=255, type=cv2.THRESH_BINARY
    )
    thresh_bool = thresh > 0
    thresh_img = add_image * thresh_bool
    min_pixel = 10
    mask = morphology.remove_small_objects(thresh_bool, min_pixel)
    res_img = img_as_ubyte(add_image * mask)
    # ipdb.set_trace()
    cv2.imwrite(f'{sv_dir}/1_x.jpg', gradx)
    cv2.imwrite(f'{sv_dir}/1_y.jpg', grady)
    cv2.imwrite(f'{sv_dir}/1_add.jpg', add_image)
    cv2.imwrite(f'{sv_dir}/1_thresh.jpg', thresh_img)
    cv2.imwrite(f'{sv_dir}/1_res.jpg', res_img)
    print(thresh.sum())
    print(mask.sum())


def sobel_demo2(image, sv_dir):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    # cv2.imshow('sobel_demo2_gradient_x', gradx)
    # cv2.imshow('sobel_demo2_gradient_y', grady)
    # 合并x, y两个梯度
    add_image = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # cv2.imshow('sobel_demo2_addWeighted', add_image)
    _, thresh = cv2.threshold(
        src=add_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY
    )
    thresh_bool = thresh > 0
    thresh_img = add_image * thresh_bool
    min_pixel = 10
    mask = morphology.remove_small_objects(thresh_bool, min_pixel)
    res_img = img_as_ubyte(add_image * mask)
    cv2.imwrite(f'{sv_dir}/2_x.jpg', gradx)
    cv2.imwrite(f'{sv_dir}/2_y.jpg', grady)
    cv2.imwrite(f'{sv_dir}/2_add.jpg', add_image)
    cv2.imwrite(f'{sv_dir}/2_thresh.jpg', thresh_img)
    cv2.imwrite(f'{sv_dir}/2_res.jpg', res_img)
    print(thresh.sum())
    print(mask.sum())


if __name__ == '__main__':
    sv_dir = 'dataset/data/test'
    img_path = f'{sv_dir}/front.png'
    # img_path = f'{sv_dir}/00107.jpg'
    # img_path = f'{sv_dir}/00044.jpg'
    # img_path = f'model/vis/45.jpg'
    src = cv2.imread(img_path)
    sobel_demo1(src, sv_dir)
    sobel_demo2(src, sv_dir)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
