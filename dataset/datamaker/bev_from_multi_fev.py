import os
import sys
import cv2
import math
import json
import time
import ipdb
import math
import torch
import shutil
import random
import threading
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from skimage import morphology
from easydict import EasyDict
from itertools import permutations
from einops import rearrange


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.backends.cuda.matmul.allow_tf32 = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class DataMakerTorch(nn.Module):
    '''
    if enable cuda, the result caculated by homo_torch_op is error, surprisely!
    if disable cuda (use cpu), the result is right.
    Why ?
    cuda core operation mechanism?
    '''

    def __init__(self, enable_cuda=True):
        super().__init__()
        self.dataset_root = '/home/data/lwb/data'
        self.dataset_name = 'dybev'
        self.dataset_version = version = 'v3'
        self.dataset_sv_dir = os.path.join(self.dataset_root, self.dataset_name)
        self.dataset_fev_video_dir_name = 'AVM_record_ocr'
        self.generate_dir_name = 'generate'
        self.perturb_pts_json_name = 'perturbed_points.json'
        self.detected_pts_json_name = 'detected_points.json'
        self.file_record_txt_name = 'image_names.txt'
        self.base_dir = base_dir = os.path.join('dataset', 'data')  # in code root dir
        self.threads_num = 32
        self.batch_size = bs = 32
        self.batch_size = self._init_make_batch_divisible(bs, self.threads_num)
        self.mtp = MultiThreadsProcess()
        self.src_num_mode_key_name = EasyDict(
            multi='multiple_driving_images',
            single='single_calibrate_image',
        )
        self.src_num_mode = self.src_num_mode_key_name.multi
        # self.src_num_mode = self.src_num_mode_key_name.single
        self.enable_cuda = enable_cuda
        self.offset_pix_range = 15
        self.perturb_mode = 'uniform'  # randint
        # the idx=0 image don't perturb, and the idx=1 to idx=1000 images are perturbed
        self.train_img_num = 1
        self.test_img_num = 2
        self.num_total_images = -1
        self.method = 'Axb'
        self.index = [0, 3, 4, 7]  # four corners
        self.camera_fblr = ["front", "back", "left", "right"]
        self.align_fblr = True  # fblr four cameras
        bev_mode = 'undist2bev'  # indirectly
        # bev_mode = 'fev2bev' # directly
        self.perturbed_image_type = 'bev'
        self.perturbed_pipeline = bev_mode
        self.bev_dir = os.path.join(base_dir, 'bev', bev_mode)
        self.pts_path = os.path.join(base_dir, self.detected_pts_json_name)
        self.dataset_dir = os.path.join(self.dataset_sv_dir, version)
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self._init_json_key_names()
        self.homo_torch_op, self.warp_torch_op = self._init_torch_operation()
        self._init_undistored_parameters()
        self._init_avm_calibrate_paramters()
        self.pt_src_fblr, self.pt_dst_fblr = self.read_points(self.index)
        self._init_warp_to_bev()
        self.batch_list, self.iteration = self._init_batch_and_iteration()

    def _init_torch_operation(self):
        self.wh_bev_fblr = wh_fblr = {
            "front": [1078, 336],
            "back": [1078, 336],
            "left": [1172, 439],
            "right": [1172, 439],
        }
        bs, is_gpu = self.batch_size, self.enable_cuda
        self.homo_torch_op = HomoTorchOP(method=self.method)
        self.warp_torch_op = {}
        for x in self.camera_fblr:
            self.warp_torch_op[x] = WarpTorchOP(bs, *wh_fblr[x], 0, is_gpu)

        return self.homo_torch_op, self.warp_torch_op

    def _init_warp_to_bev(self):
        self.f2u_dir_name = 'fev2undist'  # indirectly
        # self.f2b_dir_name = 'fev2bev' # directly
        self.src_img_path_record_txt = os.path.join(
            self.dataset_sv_dir, self.f2u_dir_name, self.file_record_txt_name
        )
        if self.src_num_mode == self.src_num_mode_key_name.multi:
            self.multiple_undist_dir = os.path.join(
                self.dataset_sv_dir, self.f2u_dir_name, 'undist'
            )
            assert self.align_fblr == True, 'only support four cameras'
            # all cameras have the same number of images
            self.any_camera = 'front'
            self.cam_img_dir_path = os.path.join(
                self.multiple_undist_dir, self.any_camera
            )
            self.cam_img_name_list = os.listdir(self.cam_img_dir_path)
            self.num_img_per_camera = len(self.cam_img_name_list)
        else:
            self.num_img_per_camera = 1
            self.single_fev_dir = os.path.join(self.base_dir, 'fev')
            self.single_undist_dir = os.path.join(self.base_dir, 'undist')

    def _init_avm_calibrate_paramters(self):
        self.calibrate = {
            "shift_w": 310,
            "shift_h": 170,
            "spread_w": 460,
            "spread_h": 740,
            "inn_shift_w": 30,
            "inn_shift_h": 20,
            "rectangle_length": 100,
            "detected_points_json_name": "detected_points.json",
            "detected_points_write_flag": False,
        }
        self.bev_wh_fblr = {
            "front": {"w": 1078, "h": 336},
            "back": {"w": 1078, "h": 336},
            "left": {"w": 1172, "h": 439},
            "right": {"w": 1172, "h": 439},
        }
        self.avm_resolution = {"w": 616, "h": 670, "scale": 1.75}

    def _init_undistored_parameters(self):
        scale = 1.0  # previous 0.5, current 1.0
        scale = self.new_scale_for_undist = 1.0
        fish = {"scale": scale, "width": 1280, "height": 960}
        hardware = {"focal_length": 950, "dx": 3, "dy": 3, "cx": 640, "cy": 480}
        distort = {
            "Opencv_k0": 0.117639891128,
            "Opencv_k1": -0.0264845591715,
            "Opencv_k2": 0.0064761037844,
            "Opencv_k3": -0.0012833025037,
            "undis_scale": 3.1,
        }
        focal_len = hardware["focal_length"]
        self.dx = dx = hardware["dx"] / fish["scale"]
        self.dy = dy = hardware["dy"] / fish["scale"]
        self.fish_width = distort_width = int(fish["width"] * fish["scale"])
        self.fish_height = distort_height = int(fish["height"] * fish["scale"])
        undis_scale = distort["undis_scale"]
        self.distort = distort
        self.center_w = center_w = distort_width / 2
        self.center_h = center_h = distort_height / 2
        self.intrinsic = intrinsic = [
            [focal_len / dx, 0, center_w],
            [0, focal_len / dy, center_h],
            [0, 0, 1],
        ]
        self.intrinsic_undis = intrinsic_undis = [
            [focal_len / dx, 0, center_w * undis_scale],
            [focal_len / dy, 0, center_h * undis_scale],
            [0, 0, 1],
        ]
        self.undist_w = u_w = int(distort_width * undis_scale)
        self.undist_h = u_h = int(distort_height * undis_scale)
        check1 = u_w == 1984 and u_h == 1488 and scale == 0.5
        check2 = u_w == 3968 and u_h == 2976 and scale == 1.0
        assert check1 or check2

        self.d = math.sqrt(
            pow(intrinsic_undis[0][2] / intrinsic[0][0], 2)
            + pow(intrinsic_undis[1][2] / intrinsic[1][1], 2)
        )

    def _init_make_batch_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _init_batch_and_iteration(self):
        size = self.num_img_per_camera
        batch = min(self.batch_size, size)
        iteration = int(size / batch)
        batch_list = [batch] * iteration
        if size % batch != 0:
            batch_list.append(size % batch)
            iteration += 1
        return batch_list, iteration

    def _init_json_key_names(self):
        self.key_name_pts_perturb = EasyDict(
            dict(
                dataset_version="dataset_version",
                perturbed_image_type="perturbed_image_type",
                perturbed_pipeline="perturbed_pipeline",
                dlt_homography_method="dlt_homography_method",
                make_date="make_date",
                random_mode="random_mode",
                train_img_num="train_img_num",
                test_img_num="test_img_num",
                camera_list="camera_list",
                offset_pixel_range="offset_pixel_range",
                original_points="original_points",
                perturbed_points_list="perturbed_points_list",
                offset_list="offset_list",
            )
        )
        self.key_name_pts_detect = EasyDict(
            dict(
                detected_points='detected_points',
                corner_points='corner_points',
            )
        )
        return

    def _init_mode(self, mode='train'):
        if mode in ['train', 'gt_bev']:
            self.num_generated_points = self.train_img_num
            self.delta_num = 0
        elif mode == 'test':
            self.num_generated_points = self.test_img_num
            self.delta_num = self.train_img_num
        else:
            raise ValueError('support mode in [`train`. `test`]')
        self.dataset_mode_dir = set_dir = os.path.join(self.dataset_dir, mode)
        os.makedirs(set_dir, exist_ok=True)
        self.perturb_pts_path = os.path.join(set_dir, self.perturb_pts_json_name)
        self.generate_dir = os.path.join(set_dir, self.generate_dir_name)

    def get_src_images(self, idx=None, batch_size=None, mode=None, align_fblr=None):

        if mode is None:
            mode = self.src_num_mode
        if align_fblr is None:
            align_fblr = self.align_fblr
        src_fblr = None
        nam_fblr = None

        # for multiprocessing
        def _read_image_kernel(img_path):
            img = cv2.imread(img_path)
            if self.new_scale_for_undist == 1.0:
                wh = tuple([int(x / 2) for x in img.shape[:2]][::-1])
                img = cv2.resize(img, wh, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img).unsqueeze(0)
            return img

        if align_fblr:
            if mode == self.src_num_mode_key_name.single:
                src_fblr = self.read_image_fblr()
            elif mode == self.src_num_mode_key_name.multi:
                src_fblr = dict(zip(self.camera_fblr, [[]] * len(self.camera_fblr)))
                nam_fblr = dict(zip(self.camera_fblr, [[]] * len(self.camera_fblr)))
                for camera in self.camera_fblr:
                    if batch_size is not None and idx is not None:
                        cam_dir = os.path.join(self.multiple_undist_dir, camera)
                        name_list = self.cam_img_name_list[
                            idx * batch_size : (idx + 1) * batch_size
                        ]
                        file_name_list = [
                            e.replace(self.any_camera, camera) for e in name_list
                        ]
                        # multiprocessing
                        threads_num = self.threads_num
                        imgs_num = len(file_name_list)
                        if imgs_num < threads_num:
                            loop_num = 1
                            threads_num = max(1, imgs_num)
                        else:
                            assert imgs_num % threads_num == 0
                            loop_num = int(imgs_num / threads_num)
                        # pipeline
                        for i in range(loop_num):
                            threads = []
                            for j in range(threads_num):
                                # preprocess
                                name = file_name_list[i * threads_num + j]
                                file_path = os.path.join(cam_dir, name)
                                # input
                                thr = ThreadsOP(_read_image_kernel, args=(file_path,))
                                threads.append(thr)
                            # process
                            for thr in threads:
                                thr.start()
                            for thr in threads:
                                thr.join()
                            for thr in threads:
                                # postprocess
                                img = thr.get_result()
                                # output
                                src_fblr[camera].append(img)
                                nam_fblr[camera].append(name)
                    else:
                        # read all images in one batch at once
                        cam_dir = os.path.join(self.multiple_undist_dir, camera)
                        file_name_list = os.listdir(cam_dir)
                        if len(file_name_list) > 64:
                            raise Exception("Too many images to feed into list")
                        for name in file_name_list:
                            file_path = os.path.join(cam_dir, name)
                            img = _read_image_kernel(file_path)
                            src_fblr[camera].append(img)
                            nam_fblr[camera].append(name)
                    img_batch = torch.stack(src_fblr[camera], dim=0)
                    if self.enable_cuda:
                        img_batch = img_batch.cuda()
                    src_fblr[camera] = img_batch
            else:
                raise ValueError
        else:
            # not four cameras
            with open(self.src_img_path_record_txt, 'r') as f:
                img_path_list = f.readlines()
            for x in img_path_list:
                name, path = x.split(': ')
                img = cv2.imread(path)
            raise ValueError('not support!')

        return src_fblr, nam_fblr

    def perturb_func(self, pts, random_mode='uniform'):
        assert self.offset_pix_range > 0
        _pix = int(self.offset_pix_range)
        pts = np.asarray(pts)
        gen = np.copy(pts)
        offset = np.zeros_like(pts)
        gen_list = {}
        off_list = {}
        for i in range(self.delta_num, self.delta_num + self.num_generated_points):
            np.random.seed(i)
            if random_mode == 'randint':
                offset = np.random.randint(-1 * _pix, _pix, pts.shape)
            elif random_mode == 'uniform':
                offset = np.random.uniform(-1 * _pix, _pix, pts.shape)
            else:
                raise ValueError
            if i == self.delta_num:
                offset = np.zeros_like(pts)
            new_pt = gen + offset
            idx = f'{i:04}'
            gen_list[idx] = new_pt.tolist()
            off_list[idx] = offset.tolist()
        return gen_list, off_list

    def generate_perturbed_points(self, index=[0, 3, 4, 7]):
        '''
        The location of the key-names should not be placed here.
        Because, in other places, it will still be called in the
        form of a string, which will be wrong and inconsistent.
        So, change the string to key of dict
        '''
        with open(self.pts_path, "r") as f:
            pts = json.load(f)
            kn2_ = self.key_name_pts_detect
        kn1_ = self.key_name_pts_perturb
        pts_perturb = {
            kn1_.dataset_version: self.dataset_version,
            kn1_.perturbed_image_type: self.perturbed_image_type,
            kn1_.perturbed_pipeline: self.perturbed_pipeline,
            kn1_.dlt_homography_method: self.method,
            kn1_.make_date: time.strftime('%z %Y-%m-%d %H:%M:%S', time.localtime()),
            kn1_.random_mode: self.perturb_mode,
            kn1_.train_img_num: self.train_img_num,
            kn1_.test_img_num: self.test_img_num,
            kn1_.camera_list: self.camera_fblr,
        }
        if index is None:
            index = self.index
        for camera in self.camera_fblr:
            # default points: bev
            pt_ori = pts[kn2_.corner_points][camera]
            pt_ori = [[pt_ori[i * 2], pt_ori[i * 2 + 1]] for i in index]
            gen_list, offset_list = self.perturb_func(pt_ori, self.perturb_mode)
            pts_perturb[camera] = {
                kn1_.offset_pixel_range: self.offset_pix_range,
                kn1_.original_points: pt_ori,
                kn1_.perturbed_points_list: gen_list,
                kn1_.offset_list: offset_list,
            }
        with open(self.perturb_pts_path, 'w') as f:
            json.dump(pts_perturb, f)

        return pts_perturb

    def read_points(self, index=[0, 3, 4, 7]):
        with open(self.pts_path, "r") as f:
            pts = json.load(f)
            kn_ = self.key_name_pts_detect
        pt_src_fblr = {}
        pt_dst_fblr = {}
        if index is None:
            index = self.index
        for camera in self.camera_fblr:
            pt_src = pts[kn_.detected_points][camera]
            pt_dst = pts[kn_.corner_points][camera]
            pt_src = [[[pt_src[i * 2], pt_src[i * 2 + 1]] for i in index]]
            pt_dst = [[[pt_dst[i * 2], pt_dst[i * 2 + 1]] for i in index]]
            pt_src = torch.Tensor(pt_src).reshape(1, -1, 2)
            pt_dst = torch.Tensor(pt_dst).reshape(1, -1, 2)
            if self.enable_cuda:
                pt_src = pt_src.cuda()
                pt_dst = pt_dst.cuda()
            pt_src_fblr[camera] = pt_src
            pt_dst_fblr[camera] = pt_dst
        return pt_src_fblr, pt_dst_fblr

    def read_image_fblr(self):
        img_fblr = {}
        for camera in self.camera_fblr:
            path = os.path.join(self.single_undist_dir, f'{camera}.jpg')
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))[np.newaxis, :]
            img = torch.from_numpy(img)
            if self.enable_cuda:
                img = img.cuda()
            img_fblr[camera] = img
        return img_fblr

    def warp_perturbed_image_single_src(self, pts_perturb=None, src_fblr=None, bs=None):
        pt_undist_fblr = self.pt_src_fblr
        if pts_perturb is None:
            with open(self.perturb_pts_path, 'r') as f:
                pts_perturb = json.load(f)
        kn_ = self.key_name_pts_perturb
        if src_fblr is None:
            src_fblr = self.read_image_fblr()
        for camera in self.camera_fblr:
            save_dir = f"{self.generate_dir}/{camera}"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(self.delta_num, self.delta_num + self.num_generated_points):
                idx = f'{i:04}'
                # perturbed points on bev
                pt_perturbed = pts_perturb[camera][kn_.perturbed_points_list][idx]
                pt_perturbed = torch.Tensor(pt_perturbed).reshape(1, -1, 2)
                if self.enable_cuda:
                    pt_perturbed = pt_perturbed.cuda()
                pt_undist = pt_undist_fblr[camera]
                H = self.homo_torch_op(pt_undist, pt_perturbed)[0]
                H_inv = torch.inverse(H).unsqueeze(0)
                src = src_fblr[camera]
                dst = self.warp_torch_op[camera](src, H_inv)
                dst = dst.detach().cpu()
                dst = dst.squeeze(0).numpy().transpose((1, 2, 0))
                dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
                sv_path = os.path.join(save_dir, idx)
                cv2.imwrite(sv_path, dst)
        return

    def warp_perturbed_image_multi_src(self, pts_perturb, src_fblr, name_fblr, bs):
        pt_undist_fblr = self.pt_src_fblr
        kn_ = self.key_name_pts_perturb

        # for multiprocessing
        def _save_image_kernel(sv_path, img):
            is_success = cv2.imwrite(sv_path, img)
            return is_success

        def _preprocess(i, j, threads_num, dst, names, save_dir, idx):
            img = dst[i * threads_num + j]
            name = names[i * threads_num + j]
            name = name.replace('.', f'_p{idx}.')
            sv_path = os.path.join(save_dir, name)
            return (sv_path, img)

        def _postprocess(is_success):
            if not is_success:
                raise Exception("imwrite error")

        for camera in self.camera_fblr:
            print(camera)
            names = name_fblr[camera]
            src_cam = src_fblr[camera]
            save_dir = os.path.join(self.generate_dir, camera)
            os.makedirs(save_dir, exist_ok=True)
            for k in range(self.delta_num, self.delta_num + self.num_generated_points):
                idx = f'{k:04}'
                # perturbed points on bev
                pt_perturbed = pts_perturb[camera][kn_.perturbed_points_list][idx]
                pt_perturbed = torch.Tensor(pt_perturbed).reshape(1, -1, 2)
                if self.enable_cuda:
                    pt_perturbed = pt_perturbed.cuda()
                pt_perturbed = pt_perturbed.repeat(bs, 1, 1)
                pt_undist = pt_undist_fblr[camera].repeat(bs, 1, 1)
                H = self.homo_torch_op(pt_undist, pt_perturbed)[0]
                H_inv = torch.inverse(H).unsqueeze(0)
                dst = self.warp_torch_op[camera](src_cam, H_inv)
                dst = dst.transpose(1, 2).transpose(2, 3) # bchw -> bhwc
                dst = dst.detach().cpu().numpy()
                # save
                self.mtp.multi_threads_process(
                    input_values=(self.threads_num, dst, names, save_dir, idx),
                    batch_size=len(dst),
                    threads_num=self.threads_num,
                    func_thread_kernel=_save_image_kernel,
                    func_preprocess=_preprocess,
                    func_postprocess=_postprocess,
                )
        return

    def warp_perturbed_image(
        self, pts_perturb=None, src_fblr=None, name_fblr=None, bs=None
    ):
        warp_params = (pts_perturb, src_fblr, name_fblr, bs)
        if self.src_num_mode == self.src_num_mode_key_name.single:
            self.warp_perturbed_image_single_src(*warp_params)
        elif self.src_num_mode == self.src_num_mode_key_name.multi:
            self.warp_perturbed_image_multi_src(*warp_params)
        else:
            raise ValueError
        return

    def get_grids(self, h, w):
        # 换成 torch
        grids = np.meshgrid(np.arange(w), np.arange(h))
        grids = np.stack(grids, axis=2).astype(np.float32)
        return grids

    def get_fev2undist_remap_table(self, is_bchw=False):
        # 换成torch
        '''output shape
        hwc: (2976, 3968, 2) H W C
        bchw: (B, 2, H, W)
        '''
        undist_center = np.array([self.undist_w / 2, self.undist_h / 2]).reshape(
            1, 1, 2
        )
        dist_center = np.array([self.center_w, self.center_h]).reshape(1, 1, 2)
        f = np.array([self.intrinsic[0][0], self.intrinsic[1][1]]).reshape(1, 1, 2)

        grids = self.get_grids(self.undist_h, self.undist_w)
        grids = grids - undist_center
        grids_norm = grids / f
        r_undist = np.linalg.norm(grids_norm, axis=2)
        angle_undistorted = np.arctan(r_undist)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        r_distort = (
            angle_undistorted
            + self.distort['Opencv_k0'] * angle_undistorted_p3
            + self.distort['Opencv_k1'] * angle_undistorted_p5
            + self.distort['Opencv_k2'] * angle_undistorted_p7
            + self.distort['Opencv_k3'] * angle_undistorted_p9
        )
        scale = r_distort / (r_undist + 0.00001)
        scale = scale[..., np.newaxis]
        grids = grids * scale + dist_center
        grids = grids.astype(np.float32)

        if is_bchw:
            ones = torch.ones_like(1)
        return grids

    def get_fev2bev_remap_table(self, grids):
        return grids

    def shutil_copy(self):
        set_dir = self.dataset_mode_dir
        shutil.copy(
            'dataset/data/detected_points.json',
            f'{set_dir}/detected_points.json',
        )
        shutil.copy('dataset/data/homo.json', f'{set_dir}/homo.json')
        if self.src_num_mode == self.src_num_mode_key_name.single:
            shutil.copytree('dataset/data/bev', f'{set_dir}/bev')
            shutil.copytree('dataset/data/undist', f'{set_dir}/undist')
            shutil.copytree('dataset/data/fev', f'{set_dir}/fev')
        elif self.src_num_mode == self.src_num_mode_key_name.multi:
            pass

    def forward_warp_image(self, src_fblr, pt_src_fblr, pt_dst_fblr, is_save=False):
        # pt_src_fblr, pt_dst_fblr = self.read_points(self.index)
        # src_fblr = self.read_image_fblr()
        dst_fblr = []
        for camera in self.camera_fblr:
            pt_src = pt_src_fblr[camera]
            pt_dst = pt_dst_fblr[camera]
            H = self.homo_torch_op(pt_src, pt_dst)[0]
            H_inv = torch.inverse(H).unsqueeze(0)
            src = src_fblr[camera]
            dst = self.warp_torch_op[camera](src, H_inv)
            dst_fblr.append(dst)
            if is_save:
                dst = dst.detach().cpu()
                dst = dst.squeeze(0).numpy().transpose((1, 2, 0))
                dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
                save_dir = os.path.join(self.base_dir, 'warp', camera)
                os.makedirs(save_dir, exist_ok=True)
                path = f"{save_dir}/0001.jpg"
                cv2.imwrite(path, dst)
        return dst_fblr

    def warp_image_from_grid(self, src, remap_table, is_nchw=False):
        dst_fblr = []
        for cam in self.camera_fblr:
            dst = self.warp_torch_op[cam].warp_image(src, remap_table, is_nchw)
            dst_fblr.append(dst)
        return dst_fblr


class DataMakerCV2:
    def __init__(self, args=None):
        self.base_path = os.path.join(os.getcwd(), "dataset/data")
        self.scale = 0.5
        if args is not None:
            if args.scale is not None:
                self.scale = args.scale
        self.camera_list = ["front", "back", "left", "right"]
        self._init_undistored_parameters()
        self._init_avm_calibrate_paramters()
        self._init_extract_frames_parameters()

    def _init_extract_frames_parameters(self):
        self.dataset_root = '/homo/data/lwb/data'
        self.dataset_name = 'dybev'
        self.f2u_dir_name = 'fev2undist'  # update this
        self.dataset_fev_video_dir_name = 'AVM_record_ocr'
        self.file_record_txt_name = 'image_names.txt'
        self.extracted_fev_dir_name = 'frames'
        self.undist_dir_name = 'frames_undist'

    def _init_avm_calibrate_paramters(self):
        self.calibrate = {
            "shift_w": 310,
            "shift_h": 170,
            "spread_w": 460,
            "spread_h": 740,
            "inn_shift_w": 30,
            "inn_shift_h": 20,
            "rectangle_length": 100,
            "detected_points_json_name": "detected_points.json",
            "detected_points_write_flag": False,
        }
        self.bev_wh_fblr = {
            "front": {"w": 1078, "h": 336},
            "back": {"w": 1078, "h": 336},
            "left": {"w": 1172, "h": 439},
            "right": {"w": 1172, "h": 439},
        }
        self.avm_resolution = {"w": 616, "h": 670, "scale": 1.75}

    def _init_undistored_parameters(self):
        fish = {"scale": self.scale, "width": 1280, "height": 960}
        hardware = {"focal_length": 950, "dx": 3, "dy": 3, "cx": 640, "cy": 480}
        distort = {
            "Opencv_k0": 0.117639891128,
            "Opencv_k1": -0.0264845591715,
            "Opencv_k2": 0.0064761037844,
            "Opencv_k3": -0.0012833025037,
            "undis_scale": 3.1,
        }
        focal_len = hardware["focal_length"]
        self.dx = dx = hardware["dx"] / fish["scale"]
        self.dy = dy = hardware["dy"] / fish["scale"]
        self.fish_width = distort_width = int(fish["width"] * fish["scale"])
        self.fish_height = distort_height = int(fish["height"] * fish["scale"])
        undis_scale = distort["undis_scale"]
        self.distort = distort
        self.center_w = center_w = distort_width / 2
        self.center_h = center_h = distort_height / 2
        self.intrinsic = intrinsic = [
            [focal_len / dx, 0, center_w],
            [0, focal_len / dy, center_h],
            [0, 0, 1],
        ]
        self.intrinsic_undis = intrinsic_undis = [
            [focal_len / dx, 0, center_w * undis_scale],
            [focal_len / dy, 0, center_h * undis_scale],
            [0, 0, 1],
        ]
        self.undist_w = u_w = int(distort_width * undis_scale)
        self.undist_h = u_h = int(distort_height * undis_scale)
        assert (
            u_w == 1984
            and u_h == 1488
            and self.scale == 0.5
            or u_w == 3968
            and u_h == 2976
            and self.scale == 1.0
        )

        self.d = math.sqrt(
            pow(intrinsic_undis[0][2] / intrinsic[0][0], 2)
            + pow(intrinsic_undis[1][2] / intrinsic[1][1], 2)
        )

    def get_grids(self, h, w):
        grids = np.meshgrid(np.arange(w), np.arange(h))
        grids = np.stack(grids, axis=2).astype(np.float32)
        return grids

    def get_correct_table(self):
        t1 = time.time()
        undist_center = np.array([self.undist_w / 2, self.undist_h / 2]).reshape(
            1, 1, 2
        )
        dist_center = np.array([self.center_w, self.center_h]).reshape(1, 1, 2)
        f = np.array([self.intrinsic[0][0], self.intrinsic[1][1]]).reshape(1, 1, 2)

        grids = self.get_grids(self.undist_h, self.undist_w)
        grids = grids - undist_center
        grids_norm = grids / f
        r_undist = np.linalg.norm(grids_norm, axis=2)
        angle_undistorted = np.arctan(r_undist)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        r_distort = (
            angle_undistorted
            + self.distort['Opencv_k0'] * angle_undistorted_p3
            + self.distort['Opencv_k1'] * angle_undistorted_p5
            + self.distort['Opencv_k2'] * angle_undistorted_p7
            + self.distort['Opencv_k3'] * angle_undistorted_p9
        )
        scale = r_distort / (r_undist + 0.00001)
        scale = scale[..., np.newaxis]
        grids = grids * scale + dist_center
        # (2976, 3968, 2) H W C
        t2 = time.time()
        # print('get grids cost time:', round(t2 - t1, 3), 's')
        return grids.astype(np.float32)

    def calc_angle_undistorted(self, r_):
        angle_undistorted = math.atan(r_)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        angle_distorted = (
            angle_undistorted
            + self.distort["Opencv_k0"] * angle_undistorted_p3
            + self.distort["Opencv_k1"] * angle_undistorted_p5
            + self.distort["Opencv_k2"] * angle_undistorted_p7
            + self.distort["Opencv_k3"] * angle_undistorted_p9
        )

        return angle_distorted

    def get_remap_table(self, mode="fev2bev", is_save=False):
        assert mode in [
            "fev2bev",
            "fev2undist",
            "undist2bev",
            'fev2undist2virtual',
        ], AssertionError

        undist_center_w = int(self.undist_w / 2)
        undist_center_h = int(self.undist_h / 2)
        f_dx = self.intrinsic[0][0]
        f_dy = self.intrinsic[1][1]
        homo_fblr = self.get_homo_undist2bev(is_save)
        map_fblr = {}
        for camera in self.camera_list:
            map_x, map_y = [], []
            if mode in ["fev2bev", "undist2bev"]:
                col = self.bev_wh_fblr[camera]["w"]
                row = self.bev_wh_fblr[camera]["h"]
                homo_b2u = np.linalg.inv(homo_fblr[camera])
            elif mode in ['fev2undist', 'fev2undist2virtual']:
                col = self.undist_w  # 1984
                row = self.undist_h  # 1488
                if mode == 'fev2undist2virtual':
                    # homo_u2v = self.get_homo_undist2virtual(is_save)
                    homo_u2v = np.eyes(3)
            for i in range(row):
                row_x, row_y = [], []
                for j in range(col):
                    if mode == "fev2bev":
                        jj, ii = self.matrix_mul_3x3(homo_b2u, j, i)
                    elif mode == "fev2undist":
                        jj, ii = j, i
                    elif mode == 'fev2undist2virtual':
                        # self.matrix_mul_3x3(homo_u2v, j, i)
                        jj, ii = j, i
                    elif mode == "undist2bev":
                        jj, ii = self.matrix_mul_3x3(homo_b2u, j, i)
                        row_x.append(jj)
                        row_y.append(ii)
                        continue
                    x_ = (jj - undist_center_w) / f_dx
                    y_ = (ii - undist_center_h) / f_dy
                    r_ = math.sqrt(pow(x_, 2) + pow(y_, 2)) + 0.00000001
                    angle_distorted = self.calc_angle_undistorted(r_)
                    scale = angle_distorted / r_
                    xx = jj - undist_center_w
                    yy = ii - undist_center_h
                    warp_x = xx * scale + self.center_w
                    warp_y = yy * scale + self.center_h
                    row_x.append(warp_x)
                    row_y.append(warp_y)

                map_x.append(row_x)
                map_y.append(row_y)
            map_x = np.asarray(map_x, dtype=np.float32)
            map_y = np.asarray(map_y, dtype=np.float32)
            map_fblr[camera] = {"x": map_x, "y": map_y}

        return map_fblr

    def random_perturb(self, start):
        """
        adding a random warping for fake pair(MaFa, MbFb) and true pair (Fa, Fa'), since there is an interpolation transformation between the original real pair (Fa, Fa')  [easily
         distinguishable by discriminators]
        start: x y
        """
        Ph, Pw = self.patch_size

        shift = np.random.randint(-self.shift, self.shift, (4, 2))

        src = np.array([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]) + start
        dst = np.copy(src)

        dst[:, 0] = dst[:, 0] + shift[:, 0]
        dst[:, 1] = dst[:, 1] + shift[:, 1]

        H, _ = cv2.findHomography(src, dst)

        return H, shift

    def get_homo_undist2virtual(self, is_save):
        def _slove_homo_from_Rt(R, t, d, n, intrinsic):
            intrinsic_inv = np.linalg.inv(intrinsic)
            n_ = n.reshape(3, 1)
            H = intrinsic * (R + t / d * n_) * intrinsic_inv
            return H

        def _calc_rotation(theta):
            R = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(theta), -math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)],
                ]
            )
            return R

        random_pose_pitch = 5
        random_pose_roll = 5
        theta_pitch = random_pose_pitch / 180 * math.pi
        theta_roll = random_pose_roll / 180 * math.pi
        R_pitch = _calc_rotation(theta_pitch)
        R_roll = _calc_rotation(theta_roll)
        t = np.zeros([3, 1])
        n = np.array([0, 0, 1])
        n *= R_pitch
        d = None
        H_pitch = _slove_homo_from_Rt(R_pitch, t, d, n, intrinsic=None)

        return

    def get_homo_undist2bev(self, shift_func=None, is_save=False):
        pts_path = f"{self.base_path}/detected_points.json"
        with open(pts_path, "r") as f:
            pts = json.load(f)
        bev_fblr_wh = [[1078, 336], [1078, 336], [1172, 439], [1172, 439]]
        homo_fblr = {}
        homo_u2b, homo_b2u = {}, {}
        homo_path = f"{self.base_path}/homo.json"
        for idx, camera in enumerate(self.camera_list):
            # 8点法计算Homography
            pt_det = pts["detected_points"][camera]
            pt_bev = pts["corner_points"][camera]
            pt_det = [[[pt_det[i * 2], pt_det[i * 2 + 1]] for i in range(8)]]
            # ToDo 扰动 det_pts
            pt_det_array = np.asarray(pt_det)
            pt_bev = [[[pt_bev[i * 2], pt_bev[i * 2 + 1]] for i in range(8)]]
            pt_bev_array = np.asarray(pt_bev)
            homo_ransac, _ = cv2.findHomography(pt_det_array, pt_bev_array, 0)
            # 保存 H
            homo_fblr[camera] = homo_ransac
            homo_u2b[camera] = homo_ransac.tolist()
            homo_b2u[camera] = np.linalg.inv(homo_ransac).tolist()
            if is_save:
                img_undist = cv2.imread(f"{self.base_path}/undist/{camera}.jpg")
                img_bev = cv2.warpPerspective(
                    img_undist, homo_ransac, bev_fblr_wh[idx], cv2.INTER_LINEAR
                )
                os.makedirs(f"{self.base_path}/u2b1", exist_ok=True)
                cv2.imwrite(f"{self.base_path}/u2b1/{camera}.png", img_bev)
                map_x, map_y = [], []
                H_b2u = np.linalg.inv(homo_fblr[camera])
                bev_w, bev_h = bev_fblr_wh[idx]
                for i in range(bev_h):
                    row_x, row_y = [], []
                    for j in range(bev_w):
                        jj, ii = self.matrix_mul_3x3(H_b2u, j, i)
                        row_x.append(jj)
                        row_y.append(ii)
                    map_x.append(row_x)
                    map_y.append(row_y)
                map_x = np.asarray(map_x, dtype=np.float32)
                map_y = np.asarray(map_y, dtype=np.float32)
                bev = cv2.remap(
                    img_undist,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                os.makedirs(f"{self.base_path}/u2b2", exist_ok=True)
                cv2.imwrite(f"{self.base_path}/u2b2/{camera}.png", bev)
        homo_js = {"undist2bev": homo_u2b, "bev2undist": homo_b2u}
        with open(homo_path, "w") as f:
            json.dump(homo_js, f, indent=4)
        return homo_fblr

    def matrix_mul_3x3(self, H, j: int, i: int):
        div = H[2][0] * j + H[2][1] * i + H[2][2] * 1
        col_x = (H[0][0] * j + H[0][1] * i + H[0][2] * 1) / div
        row_y = (H[1][0] * j + H[1][1] * i + H[1][2] * 1) / div
        return col_x, row_y

    def read_img_fblr(self, mode="fev"):
        assert mode in ["fev", "undist"], AssertionError
        img_fblr = {}
        for camera in self.camera_list:
            if mode == "fev":
                img_path = f"{self.base_path}/fev/{camera}.png"
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.fish_width, self.fish_height))
                img_fblr[camera] = img
            elif mode == "undist":
                # used for experiment: pytorch operation
                img_path = f"{self.base_path}/undist/{camera}.jpg"
                img = cv2.imread(img_path)
                img_fblr[camera] = img
            assert img.shape, print(f"checkout image path : {img_path}")

        return img_fblr

    def remap_image_cv2(self, src_fblr, map_fblr, is_save=True, save_mode="fev2bev"):
        assert src_fblr is not None and map_fblr is not None
        dst_fblr = {}
        for i, camera in enumerate(self.camera_list):
            src = src_fblr[camera]
            map = map_fblr[camera]
            dst = cv2.remap(
                src,
                map["x"],
                map["y"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            dst_fblr[camera] = dst
            if is_save:
                save_dir = f"{self.base_path}/{save_mode}"
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(f"{save_dir}/{camera}.png", dst)

        return dst_fblr

    def filter_no_texture_frame_kernel(self, image):
        sobel_pixels_threshshold = 5000
        obj_min_pixel_filter = 10
        binary_threshshold = 60
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        gradx = cv2.convertScaleAbs(grad_x)
        grady = cv2.convertScaleAbs(grad_y)
        add_image = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
        _, thresh = cv2.threshold(
            src=add_image,
            thresh=binary_threshshold,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        thresh_bool = thresh > 0
        mask = morphology.remove_small_objects(thresh_bool, obj_min_pixel_filter)
        pix_sum = mask.sum()
        is_filter = False
        if mask.sum() < sobel_pixels_threshshold:
            is_filter = True
        return is_filter, pix_sum

    def warp_from_fev_to_undist(self, grids):
        remap_params = (grids[:, :, 0], grids[:, :, 1], cv2.INTER_LINEAR)
        for camera in self.camera_list:
            img = cv2.imread(f"dataset_maker/fev/{camera}.png")
            t1 = time.time()
            img_undis = cv2.remap(img, *remap_params)
            t2 = time.time()
            cv2.imwrite(f"dataset_maker/undist_1.0/{camera}.jpg", img_undis)
            t3 = time.time()
            # print(camera, "remap cost time", round(t2 - t1, 3))
            # print(camera, "save  cost time", round(t3 - t2, 3))
            # break
        return

    def make_dataset_warp_fev2undist(self, grids):
        '''
        0it [00:00, ?it/s]
        vid: [01/10], vid_name: 20221205135519
        1it [06:55, 415.80s/it]
        vid: [02/10], vid_name: 20221205141455
        2it [20:05, 635.59s/it]
        vid: [03/10], vid_name: 20221205140313
        3it [21:13, 376.71s/it]
        vid: [04/10], vid_name: 20221205135816
        4it [33:44, 524.30s/it]
        vid: [05/10], vid_name: 20221205141246
        5it [47:49, 639.80s/it]
        vid: [06/10], vid_name: 20221205140642
        6it [59:56, 669.64s/it]
        vid: [07/10], vid_name: 20221205140456
        7it [1:12:26, 695.84s/it]
        vid: [08/10], vid_name: 20221205141401
        8it [1:24:29, 704.39s/it]
        vid: [09/10], vid_name: 20221205140548
        9it [1:36:17, 705.56s/it]
        vid: [10/10], vid_name: 20221205135629
        10it [1:48:08, 648.87s/it]
        '''
        remap_params = (grids[:, :, 0], grids[:, :, 1], cv2.INTER_LINEAR)
        base_dir = self.dataset_root
        save_dir = os.path.join(base_dir, self.dataset_name, self.f2u_dir_name)
        undist_sv_dir = os.path.join(save_dir, 'undist')
        cp_fev_sv_dir = os.path.join(save_dir, 'fev')
        videos_dir = os.path.join(
            base_dir,
            self.dataset_fev_video_dir_name,
        )
        vid_names = os.listdir(videos_dir)
        file_record_txt = os.path.join(save_dir, self.file_record_txt_name)
        if os.path.exists(file_record_txt):
            os.remove(file_record_txt)
        for idx, vid in tqdm(enumerate(vid_names)):
            img_nums = 0
            for cam in self.camera_list:
                cam_dir = os.path.join(vid_path, cam)
                img_nums += len(os.listdir(cam_dir))
            prt_str = f'\n  vid: [{idx+1:02}/{len(vid_names)}]'
            prt_str += ', vid_name: {vid}, img_nums: {img_nums}'
            print(prt_str)
            vid_path = os.path.join(videos_dir, vid)
            with tqdm(total=img_nums) as t:
                for cam in self.camera_list:
                    cam_dir = os.path.join(vid_path, cam)
                    img_names = os.listdir(cam_dir)
                    fev_dir_name = self.extracted_fev_dir_name
                    bp_cam_dir = cam_dir.replace(fev_dir_name, self.undist_dir_name)
                    os.makedirs(bp_cam_dir, exist_ok=True)
                    os.makedirs(os.path.join(undist_sv_dir, cam), exist_ok=True)
                    os.makedirs(os.path.join(cp_fev_sv_dir, cam), exist_ok=True)
                    for name in img_names:
                        img_path = os.path.join(cam_dir, name)
                        img_fev = cv2.imread(img_path)
                        # warp
                        img_undist = cv2.remap(img_fev, *remap_params)
                        sv_name = f'{vid}_{cam}_{name}'
                        undist_sv_path = os.path.join(undist_sv_dir, cam, sv_name)
                        cp_fev_sv_path = os.path.join(cp_fev_sv_dir, cam, sv_name)
                        bp_undist_sv_path = os.path.join(bp_cam_dir, name)
                        with open(file_record_txt, 'a+') as f:
                            f.write(f'{sv_name}: {undist_sv_path}\n')
                        cv2.imwrite(undist_sv_path, img_undist)
                        cv2.imwrite(bp_undist_sv_path, img_undist)
                        cv2.imwrite(cp_fev_sv_path, img_fev)
                        t.update()
        '''
        for i in front back left right ;  
        do  
            echo $i; 
        done 
        ll | awk '/front/{print  $9 }'
        ll | awk '/_front_/{print " mv -f " $9 " ./front"}' | sh
        ll | awk '/_back_/{print " mv -f " $9 " ./back"}' | sh
        ll | awk '/_left_/{print " mv -f " $9 " ./left"}' | sh
        ll | awk '/_right/{print " mv -f " $9 " ./right"}' | sh
        '''

        return


class ThreadsOP(threading.Thread):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        '''DIY function'''
        try:
            return self.result
        except Exception:
            return None


class MultiThreadsProcess:
    '''
    func_preprocess:
        input: tuple(i, j, *input_values)
        output: tuple(v1, ...) at least one return value
    func_thread_kernel:
        input: tuple(v1, ...) at least one input value
        output: None or tuple(v1, ...)
    func_postprocess:
        input: None or tuple(v1, ...)
        output: None
    '''

    def __init__(self):
        super().__init__()
        pass

    def multi_threads_process(
        self,
        input_values,
        batch_size,
        threads_num,
        func_thread_kernel,
        func_preprocess,
        func_postprocess=None,
    ):
        # multiprocessing
        if batch_size < threads_num:
            loop_num = 1
            threads_num = max(1, batch_size)
        else:
            assert batch_size % threads_num == 0
            loop_num = int(batch_size / threads_num)
        # pipeline
        result_list = []
        for i in range(loop_num):
            threads = []
            for j in range(threads_num):
                # preprocess
                args = func_preprocess(i, j, *input_values)
                # input
                thr = ThreadsOP(func_thread_kernel, args)
                threads.append(thr)
            # process
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()
            for thr in threads:
                # postprocess
                res = thr.get_result()
                if func_postprocess is not None:
                    func_postprocess(res)
                result_list.append(res)

        return result_list


class HomoTorchOP(nn.Module):
    def __init__(self, method='Axb'):
        super().__init__()
        self.method = method

    def dlt_homo(self, src_pt, dst_pt, method="Axb"):
        """
        :param src_pt: shape=(batch, num, 2)
        :param dst_pt:
        :param method: Axb (Full Rank Decomposition, inv_SVD) = 4 piar points
                    Ax0 (SVD) >= 4 pair points, 4,6,8
        :return: Homography, shape: (batch, 3, 3)
        """
        assert method in ["Ax0", "Axb"]
        assert src_pt.shape[1] >= 4
        assert dst_pt.shape[1] >= 4
        if method == 'Axb':
            assert src_pt.shape[1] == 4
            assert dst_pt.shape[1] == 4
        self.batch_size, self.nums_pt = src_pt.shape[0], src_pt.shape[1]
        xy1 = torch.cat(
            (src_pt, src_pt.new_ones(self.batch_size, self.nums_pt, 1)), dim=-1
        )
        xyu = torch.cat(
            (xy1, xy1.new_zeros((self.batch_size, self.nums_pt, 3))), dim=-1
        )
        xyd = torch.cat(
            (xy1.new_zeros((self.batch_size, self.nums_pt, 3)), xy1), dim=-1
        )
        M1 = torch.cat((xyu, xyd), dim=-1).view(self.batch_size, -1, 6)
        M2 = torch.matmul(dst_pt.view(-1, 2, 1), src_pt.view(-1, 1, 2)).view(
            self.batch_size, -1, 2
        )
        M3 = dst_pt.view(self.batch_size, -1, 1)

        if method == "Ax0":
            A = torch.cat((M1, -M2, -M3), dim=-1)
            U, S, V = torch.svd(A)
            V = V.transpose(-2, -1).conj()
            H = V[:, -1].view(self.batch_size, 3, 3)
            H *= 1 / H[:, -1, -1].view(self.batch_size, 1, 1)
        elif method == "Axb":
            A = torch.cat((M1, -M2), dim=-1)
            B = M3
            A_inv = torch.inverse(A)
            # 矩阵乘 用 gpu算出来结果不对
            # 转cpu
            # 结果用当前device储存
            # mm = torch.matmul(A_inv.cpu(), B.cpu()).to(A)
            # 关闭这个：torch.backends.cuda.matmul.allow_tf32 = False
            mm = torch.matmul(A_inv, B)
            H = torch.cat(
                (
                    mm.view(-1, 8),
                    src_pt.new_ones((self.batch_size, 1)),
                ),
                1,
            ).view(self.batch_size, 3, 3)

        return H

    def forward(self, src_pt, dst_pt):

        H = self.dlt_homo(src_pt, dst_pt, self.method)

        return H


class WarpTorchOP(nn.Module):
    def __init__(self, batch_size, w, h, start=0, enable_cuda=True):
        super().__init__()
        self.h = h
        self.w = w
        self.enable_cuda = enable_cuda
        self.grid = self.get_grid(batch_size, h, w, 0)

    def get_grid(self, batch_size, h, w, start=0):
        """
        this grid same as twice for loop
        start: start point coordinate in an image
        start.shape: (N,1,2), default value: 0
        output shape: (batch, 3, h, w)
        """
        if self.enable_cuda:
            xx = torch.arange(0, w).cuda()
            yy = torch.arange(0, h).cuda()
        else:
            xx = torch.arange(0, w)
            yy = torch.arange(0, h)
        xx = xx.view(1, -1).repeat(h, 1)
        yy = yy.view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)

        if self.enable_cuda:
            ones = torch.ones_like(xx).cuda()
        else:
            ones = torch.ones_like(xx)

        grid = torch.cat((xx, yy, ones), 1).float()

        grid[:, :2, :, :] = grid[:, :2, :, :] + start
        return grid

    def get_flow_vgrid(
        self, H_mat_mul, patch_indices, patch_size_h, patch_size_w, divide=1
    ):
        """
        patch_indices: this is grid
        divide: deblock used for mesh-flow
        output flow and vgrid
        """
        batch_size = H_mat_mul.shape[0]
        small_gap_sz = [patch_size_h // divide, patch_size_w // divide]

        small = 1e-7

        H_mat_pool = H_mat_mul.reshape(
            batch_size, divide, divide, 3, 3
        )  # .transpose(2,1)
        H_mat_pool = H_mat_pool.repeat_interleave(
            small_gap_sz[0], axis=1
        ).repeat_interleave(small_gap_sz[1], axis=2)

        pred_I2_index_warp = patch_indices.permute(0, 2, 3, 1).unsqueeze(4).contiguous()

        pred_I2_index_warp = (
            torch.matmul(H_mat_pool, pred_I2_index_warp)
            .squeeze(-1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # H multiply grid, here is generate remap table
        T_t = pred_I2_index_warp[:, 2:3, ...]
        smallers = 1e-6 * (1.0 - torch.ge(torch.abs(T_t), small).float())
        T_t = T_t + smallers  #
        v1 = pred_I2_index_warp[:, 0:1, ...]
        v2 = pred_I2_index_warp[:, 1:2, ...]
        v1 = v1 / T_t
        v2 = v2 / T_t
        pred_I2_index_warp = torch.cat((v1, v2), 1)
        vgrid = patch_indices[:, :2, ...]

        flow = pred_I2_index_warp - vgrid
        # use vgrid, do not use flow
        return flow, vgrid

    def bilinear_interpolate(self, im, x, y, out_size, enable_cuda=True):
        """
        write as pytorch version, can backward propagation gradient
        """
        # x: x_grid_flat
        # y: y_grid_flat
        # out_size: same as im.size
        # constants
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        # zero = torch.zeros_like([],dtype='int32')
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if enable_cuda and torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1).contiguous()
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def warp_image(self, I, vgrid, train=True):
        """
        I: Img, shape: batch_size, 1, full_h, full_w
        vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
        outsize: (patch_h, patch_w)
        vgrid: H multiply grid
        and then, according to vgrid, fetch data from source image
        """
        C_img = I.shape[1]
        b, c, h, w = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = self.bilinear_interpolate(
            I, x_s_flat, y_s_flat, out_size, self.enable_cuda
        )

        output = input_transformed.reshape([b, h, w, C_img])

        if train:
            output = output.permute(0, 3, 1, 2).contiguous()
        return output

    def forward(self, src, H):
        flow, vgrid = self.get_flow_vgrid(H, self.grid, self.h, self.w, 1)
        dst = self.warp_image(src, vgrid + flow)

        return dst


class VideoToFrames:
    def __init__(
        self,
        args=None,
        base_dir='/home/data/lwb/data',
        data_source_dir_name='AVM_record_ocr',
        save_dir_name='frames',
    ):
        self.videos_dir = os.path.join(base_dir, data_source_dir_name)
        videos_names = []
        for vid_name in os.listdir(self.videos_dir):
            if '2022' in vid_name:
                videos_names.append(vid_name)
        self.videos_names = sorted(videos_names)
        self.camera_list = ['front', 'back', 'left', 'right']
        self.frames_total_dir = os.path.join(self.videos_dir, save_dir_name)
        if os.path.exists(self.frames_total_dir):
            shutil.rmtree(self.frames_total_dir)
        self.img_type = '.jpg'
        self.video_type = '.mp4'

    def extract_frames_kernel(self, video_path, save_dir, img_type, frames_len):
        cap = cv2.VideoCapture(video_path)
        # video_frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        # frames_len = min(frames_len, video_frames_len)
        for k in tqdm(range(frames_len)):
            _, img = cap.read()
            save_path = os.path.join(save_dir, f'{k+1:05d}' + img_type)
            try:
                cv2.imwrite(save_path, img)
            except:
                print(f'image shape:   {img.shape}')
                print(f'imwrite error: {save_path}')
                print(f'frames_length: {frames_len}')
        cap.release()

    def extract_frames(self):
        print(f'videos_dir: {self.videos_dir}\n')
        for idx, vid_name in tqdm(enumerate(self.videos_names)):
            vid_frames_dir = os.path.join(self.frames_total_dir, vid_name)
            os.makedirs(vid_frames_dir, exist_ok=True)
            cameras_frames_len_list = []
            for cam in self.camera_list:
                vid_cam_path = os.path.join(
                    self.videos_dir, vid_name, cam + self.video_type
                )
                cap = cv2.VideoCapture(vid_cam_path)
                frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                cameras_frames_len_list.append(frames_len)
                cap.release()
            frames_len = min(cameras_frames_len_list)
            prt_str = f'\n  vid: [{idx+1:02}/{len(self.videos_names)}]'
            prt_str += f', vid_name: {vid_name}, frames: {frames_len}'
            print(prt_str)
            for cam in self.camera_list:
                frames_sv_dir = os.path.join(vid_frames_dir, cam)
                os.makedirs(frames_sv_dir, exist_ok=True)
                vid_cam_path = os.path.join(
                    self.videos_dir, vid_name, cam + self.video_type
                )
                assert os.path.exists(vid_cam_path), f'vid_cam_path: {vid_cam_path}'
                cap = cv2.VideoCapture(vid_cam_path)
                # extract frames
                print(f'\t\t\t{cam}')
                for k in range(frames_len):
                    _, frame = cap.read()
                    assert len(frame.shape) == 3, f'image shape: {frame.shape}'
                    save_path = os.path.join(
                        frames_sv_dir, f'{k+1:05d}' + self.img_type
                    )
                    # ipdb.set_trace()
                    cv2.imwrite(save_path, frame)
                    if not os.path.exists(save_path):
                        print(f'imwrite error: {save_path}')
                        print(f'frames_length: {frames_len}')
                        raise ValueError
                cap.release()

        return


if __name__ == "__main__":

    # run_mode = 'cv2'
    run_mode = "torch"
    # run_mode = 'fev2undist_cv2'
    # run_mode = 'extract_frames'
    # run_mode = 'fev2bev_torch'

    if run_mode == "cv2":
        datamaker = DataMakerCV2()

        mode = "fev2bev"
        # mode = 'fev2undist'
        # mode = 'undist2bev'

        img_fblr = datamaker.read_img_fblr(mode="fev")
        # img_fblr = datamaker.read_img_fblr(mode='undist')
        map_fblr = datamaker.get_remap_table(mode=mode)
        datamaker.remap_image_cv2(img_fblr, map_fblr, is_save=True, save_mode=mode)

    elif run_mode == 'torch':
        generator = DataMakerTorch(enable_cuda=True)
        # for mode in ['train', 'test']:
        for mode in ['gt_bev']:
            generator._init_mode(mode)
            pts = generator.generate_perturbed_points()
            bs_list = generator.batch_list
            for i in tqdm(range(generator.iteration)):
                src, name = generator.get_src_images(i, bs_list[i])
                generator.warp_perturbed_image(pts, src, name, bs_list[i])
            generator.shutil_copy()

    elif run_mode == 'fev2undist_cv2':
        args = EasyDict(dict(scale=1.0))
        undistort_maker = DataMakerCV2(args)
        grids = undistort_maker.get_correct_table()
        undistort_maker.warp_from_fev_to_undist(grids)
        # undistort_maker.make_dataset_warp_fev2undist(grids)

    elif run_mode == 'extract_frames':
        frames_handle = VideoToFrames()
        frames_handle.extract_frames()

    elif run_mode == 'fev2bev_torch':
        pass
