import os
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")


class DatasetPipeline(Dataset):
    def __init__(self, params, mode="train"):
        super().__init__()
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
            name_list = [ele.split('_')[0] for ele in name_list]
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
        params = self.params
        base_path, name = self.data_sample[index]
        img_fblr, lab_fblr = [], []
        for camera in ['front', 'back', 'left', 'right']:
            img_path = os.path.join(
                base_path,
                'generate',
                camera,
                f'{name}_{params.calc_homo_device}.{params.train_image_type}',
            )
            img = Image.open(img_path)
            img = img.convert("RGB")
            if camera in ['front', 'back']:
                img = img.crop((27, 16, 27 + 1024, 16 + 320))
            else:
                img = img.crop((119, 74, 119 + 1024, 74 + 320))
            img_input = img.resize((params.img_w, params.img_h), Image.BILINEAR)
            img_fblr.append(ToTensor()(img_input))
            lab = cv2.imread(img_path)
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)
            lab = np.transpose(lab, (2, 0, 1))
            lab_fblr.append(torch.from_numpy(lab))
        # assemble
        data = {"image": img_fblr, "label": lab_fblr, "name": name, "path": base_path}

        return data

    def data_aug(self, data):
        return data


def collate_fn(batch):
    batch_out = {}
    # image, label, name
    for key in batch[0].keys():
        batch_out[key] = []
    # batch_out = {'image':[], 'label':[], 'name':[], "path":[]}

    # batch_size = 32
    for x in batch:
        for k, v in x.items():
            if k in ['image', 'label']:
                batch_out[k].extend(v)
            else:
                batch_out[k].append(v)
    # batch_out = {'image':[from 1,2,3,..., to 32], ... }
    # [chw1, chw2, chw3, ..., chw32]
    for k, v in batch_out.items():
        # stack[] -> BCHW
        # if k in ["image", "label"]:
        if k in ["image"]:
            batch_out[k] = torch.stack(v)

    return batch_out


def fetch_dataloader(params):
    assert params.dataset_type in ["basic", "train", "test"]
    ds_type_list = []

    if params.dataset_type == "basic":
        train_ds = DatasetPipeline(params, mode="train")
        eval_ds = DatasetPipeline(params, mode="test")
        ds_type_list.extend(["train", "test"])
    elif params.dataset_type == "train":
        train_ds = DatasetPipeline(params, mode="train")
        ds_type_list.append("train")
    else:
        eval_ds = DatasetPipeline(params, mode="test")
        ds_type_list.append("test")

    dataloaders = {}

    if "train" in ds_type_list:
        dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=3,
        )
        dataloaders["train"] = dl

    if "test" in ds_type_list:
        dl = DataLoader(
            eval_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers_eval,
            pin_memory=params.cuda,
            collate_fn=collate_fn,
            prefetch_factor=3,
        )
        dataloaders["test"] = dl

    return dataloaders
