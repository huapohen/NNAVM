import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms as T
from torch.utils.data import Dataset

class BEVDataset(Dataset):
    def __init__(self, params, mode="train"):
        super(BEVDataset, self).__init__()
        self.params = params
        self.mode = mode
        if mode == "train":
            self.data_ratio = params.train_data_ratio
        else:
            self.data_ratio = params.test_data_ratio
        random.seed(params.seed)
        self.data_sample = []
        for data_ratio in self.data_ratio:
            set_name = data_ratio[0]
            base_path = os.path.join(params.data_dir, set_name, self.mode)
            camera_f = os.path.join(base_path, 'generate', 'front')
            name_list = os.listdir(camera_f)
            name_list = [ele.split('.')[0] for ele in name_list]
            name_list = list(set(name_list))
            percentage = int(len(name_list) * data_ratio[1])
            if params.enable_random and self.mode != "test":
                name_list = random.sample(name_list, percentage)
            else:
                name_list = name_list[:percentage]
            file_list = [(base_path, i) for i in name_list]
            self.data_sample += file_list
        if params.enable_random and self.mode != "test":
            random.shuffle(self.data_sample)

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, index):
        '''
        img_w: 512->1024,
        img_h: 160->320,
        wh_fblr = [(1078, 336), (1078, 336), (1172, 439), (1172, 439)]
        '''
        crop_area = {"front_back": [27, 16, 1024, 320],
                     "others": [119, 74, 1024, 320]}
        params = self.params
        base_path, name = self.data_sample[index]
        img_fblr, lab_fblr = [], []
        for camera in ['front', 'back', 'left', 'right']:
            img_path = os.path.join(
                base_path,
                'generate',
                camera,
                f'{name}.{params.train_image_type}',
            )
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if camera in ['front', 'back']:
                box = crop_area["front_back"]
            else:
                box = crop_area["others"]
            # crop
            img = img[box[1]: box[1]+box[3], box[0]:box[2]]
            img = cv2.resize(img, (params.img_w, params.img_h), interpolation = cv2.INTER_AREA)
            input = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)) / 255.
            img_fblr.append(input)

            lab = cv2.imread(img_path)
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)
            lab = np.transpose(lab, (2, 0, 1))
            lab_fblr.append(torch.from_numpy(lab))

        # assemble
        data = {"image": img_fblr, "label": lab_fblr, "name": name, "path": base_path}

        return data

    def data_aug(self, data):
        return data

if __name__ == "__main__":
    import ipdb
    from easydict import EasyDict
    ipdb.set_trace()
    param = EasyDict({"train_data_ratio":[["v1", 1]],
                    "test_data_ratio":[["v1", 1]],
                    "data_dir": "/data/xingchen/dataset/AVM/dybev",
                    "enable_random": False,
                    "seed": 1,
                    "train_image_type": "jpg",
                    "img_w": 512,
                    "img_h":160})
    bev_dataset = BEVDataset(param)
    for data in bev_dataset:
        print(data)