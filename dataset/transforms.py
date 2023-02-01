import os
import cv2
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image


class RandomHSV(object):
    def __init__(self, hsv_prob=1.0, hgain=5, sgain=30, vgain=30):
        super().__init__()
        self.hsv_prob = hsv_prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, target):
        """modify Hue, Saturation and Value of image randomly"""
        if np.random.rand() < self.hsv_prob:
            hsv_augs = np.random.uniform(-1, 1, 3) * [
                self.hgain,
                self.sgain,
                self.vgain,
            ]  # random gains
            hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
            hsv_augs = hsv_augs.astype(np.int16)
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
            image = cv2.cvtColor(img_hsv.astype(image.dtype), cv2.COLOR_HSV2BGR)
        return image, target


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):

        if np.random.randint(8):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if np.random.randint(2):
            # delta = np.random.uniform(-self.delta, self.delta)
            # image += delta
            image = np.power(image, 1.11)
        else:
            image = np.power(image, 0.8)
        return image, target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if np.random.randint(8):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return image, target


class RandomHue(object):  #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if np.random.randint(8):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if np.random.randint(8):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, target):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if np.random.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomColorJitter(object):
    def __init__(self, brightness=0.9, contrast=0.5, saturation=0.5, hue=0.3):
        super().__init__()
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.jitter(img)
        return img, target


if __name__ == '__main__':

    aug_mode = 'photometric'
    # aug_mode = 'geometric'

    if aug_mode == 'photometric':
        photo_aug = PhotometricDistort()
        hsv_aug = RandomHSV()
        jitter_aug = RandomColorJitter()
        img_path = 'dataset/data/bev/fev2bev/front.png'
        img = cv2.imread(img_path)
        num = 8
        tgt1, _ = photo_aug([img] * num, None)
        tgt2, _ = photo_aug([img] * num, None)
        augs = []
        for i in range(num):
            tgt3, _ = hsv_aug(img, None)
            tgt4, _ = jitter_aug(img, None)
            aug = np.concatenate([img, tgt1[i], tgt2[i], tgt3, tgt4], axis=1)
            augs.append(aug)
        vs = np.concatenate(augs, axis=0)
        vs_gray = cv2.cvtColor(vs, cv2.COLOR_BGR2GRAY)
        sv_dir = 'dataset/data/test'
        os.makedirs(sv_dir, exist_ok=True)
        sv_path1 = f'{sv_dir}/front_aug_bgr.jpg'
        sv_path2 = f'{sv_dir}/front_aug_gray.jpg'
        cv2.imwrite(sv_path1, vs)
        cv2.imwrite(sv_path2, vs_gray)

    elif aug_mode == 'geometric':
        pass

    else:
        pass
