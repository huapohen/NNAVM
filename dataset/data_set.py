import os
import sys
import cv2
import json
import torch
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class DatasetPipeline(Dataset):
    def __init__(self, params, mode="train"):
        super().__init__()
        self.params = params
        self.mode = mode
        if mode == "train":
            self.data_ratio = params.train_data_ratio
        else:
            self.data_ratio = params.test_data_ratio
        self.camera_list = params.camera_list
        random.seed(params.seed)
        self.data_sample = []
        self.perturbed_pts = {}
        self.sample_number = {}
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
                name_list = sorted(name_list)
                name_list = name_list[:percentage]
            file_list = [(base_path, i) for i in name_list]
            self.data_sample += file_list
            self.sample_number[set_name] = {
                'ratio': float(data_ratio[1]),
                "samples": len(file_list),
            }
            # perturbed points offset
            perturbed_pts_path = os.path.join(base_path, 'perturbed_points.json')
            with open(perturbed_pts_path, 'r') as f:
                self.perturbed_pts[set_name] = json.load(f)
        if params.enable_random and self.mode != "test":
            random.shuffle(self.data_sample)
        self.sample_number["total_samples"] = len(self.data_sample)

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, index):
        '''
        img_w: 512->1024,
        img_h: 160->320,
        wh_fblr = [(1078, 336), (1078, 336), (1172, 439), (1172, 439)]
        dh_fblr = [16, 16, 119, 119]
        dw_fblr = [27, 27, 74, 74]
        '''
        params = self.params
        base_path, name = self.data_sample[index]
        # input bev: crop & resize & normalize
        img_fblr = []
        for camera in self.camera_list:
            img_path = os.path.join(
                base_path,
                'generate',
                camera,
                f'{name}.{params.train_image_type}',
            )
            ori = cv2.imread(img_path)
            ori = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
            dhdw_mode = 'fb' if camera in ['front', 'back'] else 'lr'
            dh, dw = params.dh_fblr[dhdw_mode], params.dw_fblr[dhdw_mode]
            _h = params.img_h * 2
            _w = params.img_w * 2
            img = ori[dh : dh + _h, dw : dw + _w]
            img = cv2.resize(
                img, (params.img_w, params.img_h), interpolation=cv2.INTER_AREA
            )
            img = img[:, :, np.newaxis]
            img = ToTensor()(img)
            img_fblr.append(img)

        data = {}
        data['image'] = img_fblr
        task_mode = params.dataloader_task_mode
        if 'offset' in task_mode:
            data['offset'] = self.get_offset(base_path, name)
        if 'coords' in task_mode:
            pts1, pts2, pts3 = self.get_coords(base_path, name)
            data['coords_undist'] = pts1
            data['coords_bev_origin'] = pts2
            data['coords_bev_perturbed'] = pts3
        if 'bev_origin' in task_mode:
            data['bev_origin'] = self.get_bev_origin(base_path)
        if 'undist' in task_mode:
            data['undist'] = self.get_undist(base_path)
        if 'bev_perturbed' in task_mode:
            data['bev_perturbed'] = self.get_bev_perturbed(base_path, name)
        if 'name' in task_mode:
            data['name'] = name
        if 'path' in task_mode:
            data['path'] = base_path
        data['camera_list'] = self.camera_list

        return data

    def get_offset(self, base_path, name):
        set_name = base_path.split(os.sep)[-2]
        pert_pts = self.perturbed_pts[set_name]
        offset_fblr = []
        for camera in self.camera_list:
            offset = pert_pts[camera]['offset_list'][name]
            offset = np.asarray(offset)
            offset_fblr.append(torch.from_numpy(offset).float())
        return offset_fblr

    def get_bev_origin(self, base_path):
        bev_ori_fblr = []
        for camera in self.camera_list:
            img = cv2.imread(f'{base_path}/bev/{self.params.bev_mode}/{camera}.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img).unsqueeze(0)
            bev_ori_fblr.append(img)
        return bev_ori_fblr

    def get_undist(self, base_path):
        undist_fblr = []
        for camera in self.camera_list:
            img = cv2.imread(f'{base_path}/undist/{camera}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img).unsqueeze(0)
            undist_fblr.append(img)
        return undist_fblr

    def get_bev_perturbed(self, base_path, name):
        bev_perturbed = []
        for camera in self.camera_list:
            img_path = os.path.join(
                base_path,
                'generate',
                camera,
                f'{name}.{self.params.train_image_type}',
            )
            ori = cv2.imread(img_path)
            ori = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
            ori = torch.from_numpy(ori).unsqueeze(0)
            bev_perturbed.append(ori)
        return bev_perturbed

    def get_coords(self, base_path, name):
        with open(f'{base_path}/detected_points.json', 'r') as f:
            pts = json.load(f)
        src_coords, dst_coords = [], []
        for camera in self.params.camera_list:
            index = [0, 3, 4, 7]
            pt_src = pts["detected_points"][camera]
            pt_dst = pts["corner_points"][camera]
            pt_src = [[pt_src[i * 2], pt_src[i * 2 + 1]] for i in index]
            pt_dst = [[pt_dst[i * 2], pt_dst[i * 2 + 1]] for i in index]
            pt_src = torch.Tensor(pt_src).reshape(-1, 2)
            pt_dst = torch.Tensor(pt_dst).reshape(-1, 2)
            src_coords.append(pt_src)
            dst_coords.append(pt_dst)
        src_coords = torch.stack(src_coords)
        dst_coords = torch.stack(dst_coords)
        #
        with open(f'{base_path}/perturbed_points.json', 'r') as f:
            pert = json.load(f)
        perturbed_coords = []
        for camera in self.params.camera_list:
            pt = pert[camera]['perturbed_points_list'][name]
            pt = torch.Tensor(pt).reshape(-1, 2)
            perturbed_coords.append(pt)
        return src_coords, dst_coords, perturbed_coords

    def data_aug(self, data):
        return data
