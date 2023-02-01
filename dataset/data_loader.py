import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from dataset.bev_dataset import BEVDataset
from dataset.data_set import DatasetPipeline
import warnings

warnings.filterwarnings("ignore")


def collate_fn(batch):
    batch_out = {}
    # image, label, name
    for key in batch[0].keys():
        batch_out[key] = []
    # batch_out = {'image':[], 'label':[], 'name':[], "path":[]}

    # batch_size = 32
    for x in batch:
        for k, v in x.items():
            if k in [
                'image',
                'fev',
                'bev_origin',
                'undist',
                'offset',
                'bev_perturbed',
                'coords_undist',
                'coords_bev_origin',
                'coords_bev_perturbed',
            ]:
                batch_out[k].extend(v)
            elif k == 'camera_list':
                continue
            else:
                batch_out[k].append(v)
    batch_out['camera_list'] = camera_list = batch[0]['camera_list']
    num_cam = len(camera_list)
    # batch_out = {'image':[from 1,2,3,..., to 32], ... }
    # [chw1, chw2, chw3, ..., chw32]
    for k, v in batch_out.items():
        # stack[] -> BCHW
        # if k in ["image", "label"]:
        if k in [
            "image",
            'coords_undist',
            'coords_bev_origin',
            'offset',
            'coords_bev_perturbed',
            'fev',
            'undist',
            'bev_origin',
            'bev_perturbed',
        ]:
            fblr = []
            for i, _ in enumerate(camera_list):
                fblr.append(torch.stack(v[i::num_cam]))
            if k in [
                'image',
                'offset',
                'coords_undist',
                'coords_bev_origin',
                'coords_bev_perturbed',
            ]:
                batch_out[k] = torch.cat(fblr, dim=0)
            else:
                batch_out[k] = fblr
        elif k in ['name', 'path']:
            fblr = []
            for i, _ in enumerate(camera_list):
                fblr.append(v[i::num_cam])
            batch_out[k] = fblr
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

    if "train" in ds_type_list or 'basic' in ds_type_list:
        dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            collate_fn=collate_fn,
            drop_last=False,
            prefetch_factor=3,
        )
        dl.sample_number = train_ds.sample_number
        dataloaders["train"] = dl

    if "test" in ds_type_list or 'basic' in ds_type_list:
        dl = DataLoader(
            eval_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers_eval,
            pin_memory=params.cuda,
            collate_fn=collate_fn,
            drop_last=False,
            prefetch_factor=3,
        )
        dl.sample_number = eval_ds.sample_number
        dataloaders["test"] = dl

    return dataloaders
