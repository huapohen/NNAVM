import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from preprocess import *


class DatasetPipeline(Dataset):
    def __init__(self, params, mode="train"):
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
            name_list_path = os.path.join(base_path, "image_name.txt")
            try:
                with open(name_list_path, 'r') as f:
                    name_list = f.readlines()
            except:
                raise f"Error! checkout dataset image_name.txt path: \n{name_list_path}"
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
        params = self.params
        # image
        base_path, name = self.data_sample[index]
        img_path = os.path.join(base_path, "image", name, params.train_image_type)
        try:
            img = Image.open(img_path)
        except:
            raise f"Error! checkout dataset image path: \n{img_path}"
        img = img.convert("L") if params.is_input_gray else img.convert("RGB")
        img = img.resize((params.img_w, params.img_h), Image.BILINEAR)
        # label
        lab_path = lab_path.replace(base_path, "label", name, params.test_image_type)
        try:
            lab = Image.open(lab_path)
        except:
            raise f"Error! checkout dataset label path: \n{lab_path}"
        lab = lab.convert("L") if params.is_input_gray else lab.convert("RGB")
        lab = lab.resize((params.img_w, params.img_h), Image.BILINEAR)
        # annotation
        ## ann_path = lab_path.replace(base_path, "annotations.json")
        ## with open(ann_path, 'r') as f:
        ##     ann_json = json.load(f)
        # assemble
        data = {
            "image": img,
            "label": lab,
            "name": name,
            "annotation": [],
        }
        # data augment
        data = self.data_aug(data)
        # to tensor
        ## transform = T.Compose(
        ##     [
        ##         T.ToTensor(),
        ##         T.Resize(params.img_h, params.img_w),
        ##         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ##     ]
        ## )
        data['image'] = ToTensor()(data['image'])
        data['label'] = ToTensor()(data['label'])

        return data

    def data_aug(self, data):
        return data


def collate_fn(batch):
    batch_out = {}
    # image, label, name
    for key in batch[0].keys():
        batch_out[key] = []
    # batch_out = {'image':[], 'label':[], 'name':[], "annotations":[]}

    # batch_size = 32
    for x in batch:
        for k, v in x.items():
            batch_out[k].append(v)
    # batch_out = {'image':[from 1,2,3,..., to 32], ... }
    # [chw1, chw2, chw3, ..., chw32]
    for k, v in batch_out.items():
        # stack[] -> BCHW
        if k in ["image", "label"]:
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
