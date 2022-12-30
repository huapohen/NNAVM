import os
import sys
import cv2
import math
import json
import time
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from itertools import permutations

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
        self.batch_size = batch_size = 1
        self.enable_cuda = enable_cuda
        self.offset_pix_range = 10
        self.perturb_mode = 'random'
        # the idx=0 image don't perturb, and the idx=1 to idx=1000 images are perturbed
        self.train_img_num = 1001
        self.test_img_num = 201
        version = 'v1'
        self.method = 'Axb'
        self.dst_wh_fblr = [(1078, 336), (1078, 336), (1172, 439), (1172, 439)]
        self.camera_fblr = ["front", "back", "left", "right"]
        self.index = [0, 3, 4, 7]
        self.base_dir = base_dir = 'dataset/data'
        self.fev_dir = f'{base_dir}/fev'
        self.undist_dir = f'{base_dir}/undist'
        bev_mode = 'undist2bev'
        # bev_mode = 'fev2bev'
        self.bev_dir = f'{base_dir}/bev/{bev_mode}'
        self.warp_dir = f'{base_dir}/warp'
        self.pts_path = f"{base_dir}/detected_points.json"
        self.dataset_dir = f"{base_dir}/{version}"
        os.makedirs(self.dataset_dir, exist_ok=True)
        shutil.rmtree(self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.homo_torch_op = HomoTorchOP(method=self.method)
        self.warp_torch_op = {
            "front": WarpTorchOP(batch_size, 1078, 336, 0, enable_cuda),
            "back": WarpTorchOP(batch_size, 1078, 336, 0, enable_cuda),
            "left": WarpTorchOP(batch_size, 1172, 439, 0, enable_cuda),
            "right": WarpTorchOP(batch_size, 1172, 439, 0, enable_cuda),
        }

    def init_mode(self, mode='train'):
        if mode == 'train':
            self.num_generated_points = self.train_img_num
            self.delta_num = 0
        else:  # test
            self.num_generated_points = self.test_img_num
            self.delta_num = self.train_img_num
        self.dataset_dir_mode = f"{self.dataset_dir}/{mode}"
        os.makedirs(self.dataset_dir_mode, exist_ok=True)
        self.perturb_pts_path = f"{self.dataset_dir_mode}/perturbed_points.json"
        self.generate_dir = f'{self.dataset_dir_mode}/generate'

    def read_points(self, index=[0, 3, 4, 7]):
        with open(self.pts_path, "r") as f:
            pts = json.load(f)
        pt_src_fblr = {}
        pt_dst_fblr = {}
        if index is None:
            index = self.index
        for camera in self.camera_fblr:
            pt_src = pts["detected_points"][camera]
            pt_dst = pts["corner_points"][camera]
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

    def perturb_func(self, pts, mode='random'):
        assert mode in ['random', 'permutation', 'diffusion', 'learned']
        assert self.offset_pix_range > 0
        _pix = int(self.offset_pix_range)
        pts = np.asarray(pts)
        gen = np.copy(pts)
        offset = np.zeros_like(pts)
        gen_list = {}
        off_list = {}
        if mode == 'random':
            for i in range(self.delta_num, self.delta_num + self.num_generated_points):
                np.random.seed(i)
                offset = np.random.randint(-1 * _pix, _pix, pts.shape)
                if i == self.delta_num:
                    offset = np.zeros_like(pts)
                new_pt = gen + offset
                idx = f'{i:04}'
                gen_list[idx] = new_pt.tolist()
                off_list[idx] = offset.tolist()
        return gen_list, off_list

    def generate_perturbed_points(self, index=[0, 3, 4, 7]):
        with open(self.pts_path, "r") as f:
            pts = json.load(f)
        pts_perturb = {}
        if index is None:
            index = self.index
        for camera in self.camera_fblr:
            # default points: bev
            pt_ori = pts["corner_points"][camera]
            pt_ori = [[pt_ori[i * 2], pt_ori[i * 2 + 1]] for i in index]
            gen_list, offset_list = self.perturb_func(pt_ori, self.perturb_mode)
            pts_perturb[camera] = {
                "original_points": pt_ori,
                "num_pts": self.num_generated_points,
                "perturbed_points_list": gen_list,
                "offset_list": offset_list,
            }
        with open(self.perturb_pts_path, 'w') as f:
            json.dump(pts_perturb, f)

        return pts_perturb

    def warp_perturbed_image(self, pts_perturb=None):
        pt_undist_fblr, pt_bev_fblr = self.read_points(self.index)
        src_fblr = self.read_image_fblr()
        if pts_perturb is None:
            with open(self.perturb_pts_path, 'r') as f:
                pts_perturb = json.load(f)
        for camera in self.camera_fblr:
            for i in range(self.delta_num, self.delta_num + self.num_generated_points):
                idx = f'{i:04}'
                # perturbed points on bev
                pt_perturbed = pts_perturb[camera]['perturbed_points_list'][idx]
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
                save_dir = f"{self.generate_dir}/{camera}"
                os.makedirs(save_dir, exist_ok=True)
                path = f"{save_dir}/{idx}.jpg"
                cv2.imwrite(path, dst)
        return

    def shutil_copy(self):
        shutil.copy(
            'dataset/data/detected_points.json',
            f'{self.dataset_dir_mode}/detected_points.json',
        )
        shutil.copy('dataset/data/homo.json', f'{self.dataset_dir_mode}/homo.json')
        shutil.copytree('dataset/data/bev', f'{self.dataset_dir_mode}/bev')

    def read_image_fblr(self):
        img_fblr = {}
        for camera in self.camera_fblr:
            path = f"{self.undist_dir}/{camera}.jpg"
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))[np.newaxis, :]
            img = torch.from_numpy(img)
            if self.enable_cuda:
                img = img.cuda()
            img_fblr[camera] = img
        return img_fblr

    def warp_image(self):
        pt_src_fblr, pt_dst_fblr = self.read_points(self.index)
        src_fblr = self.read_image_fblr()
        for camera in self.camera_fblr:
            pt_src = pt_src_fblr[camera]
            pt_dst = pt_dst_fblr[camera]
            H = self.homo_torch_op(pt_src, pt_dst)[0]
            H_inv = torch.inverse(H).unsqueeze(0)
            src = src_fblr[camera]
            dst = self.warp_torch_op[camera](src, H_inv)
            dst = dst.detach().cpu()
            dst = dst.squeeze(0).numpy().transpose((1, 2, 0))
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            save_dir = f"{self.warp_dir}/{camera}"
            os.makedirs(save_dir, exist_ok=True)
            path = f"{save_dir}/0001.jpg"
            cv2.imwrite(path, dst)
        return

    def forward(self):

        return self.warp_image()


class DataMakerCV2:
    def __init__(self):
        self.base_path = os.path.join(os.getcwd(), "dataset/data")
        fish = {"scale": 0.5, "width": 1280, "height": 960}
        hardware = {"focal_length": 950, "dx": 3, "dy": 3, "cx": 640, "cy": 480}
        distort = {
            "Opencv_k0": 0.117639891128,
            "Opencv_k1": -0.0264845591715,
            "Opencv_k2": 0.0064761037844,
            "Opencv_k3": -0.0012833025037,
            "undis_scale": 3.1,
        }
        calibrate = {
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
        avm_resolution = {"w": 616, "h": 670, "scale": 1.75}
        self.bev_wh_fblr = {
            "front": {"w": 1078, "h": 336},
            "back": {"w": 1078, "h": 336},
            "left": {"w": 1172, "h": 439},
            "right": {"w": 1172, "h": 439},
        }
        self.camera_list = ["front", "back", "left", "right"]
        focal_len = hardware["focal_length"]
        self.dx = dx = hardware["dx"] / fish["scale"]
        self.dy = dy = hardware["dy"] / fish["scale"]
        self.fish_width = distort_width = int(fish["width"] * fish["scale"])  # 640
        self.fish_height = distort_height = int(fish["height"] * fish["scale"])  # 480
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
        self.undist_w = int(distort_width * undis_scale)  # 1984
        self.undist_h = int(distort_height * undis_scale)  # 1488

        self.num_theta = 1024
        self.d = math.sqrt(
            pow(intrinsic_undis[0][2] / intrinsic[0][0], 2)
            + pow(intrinsic_undis[1][2] / intrinsic[1][1], 2)
        )

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
        assert mode in ["fev2bev", "fev2undist", "undist2bev"], AssertionError

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
            elif mode == "fev2undist":
                col = self.undist_w  # 1984
                row = self.undist_h  # 1488
            for i in range(row):
                row_x, row_y = [], []
                for j in range(col):
                    if mode == "fev2bev":
                        jj, ii = self.matrix_mul_3x3(homo_b2u, j, i)
                    elif mode == "fev2undist":
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
        :return: Homography
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


if __name__ == "__main__":

    # test_mode = 'cv2'
    test_mode = "torch"

    if test_mode == "cv2":
        datamaker = DataMakerCV2()

        mode = "fev2bev"
        # mode = 'fev2undist'
        # mode = 'undist2bev'

        img_fblr = datamaker.read_img_fblr(mode="fev")
        # img_fblr = datamaker.read_img_fblr(mode='undist')
        map_fblr = datamaker.get_remap_table(mode=mode)
        datamaker.remap_image_cv2(img_fblr, map_fblr, is_save=True, save_mode=mode)

    else:
        generator = DataMakerTorch(enable_cuda=True)
        for mode in ['train', 'test']:
            generator.init_mode(mode)
            generator.generate_perturbed_points()
            generator.warp_perturbed_image()
            generator.shutil_copy()
    pass
