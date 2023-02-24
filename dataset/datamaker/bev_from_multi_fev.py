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

from .calibrate_params import CalibrateParameter
from .torch_class_op import RemapTableTorchOP, HomoTorchOP, WarpTorchOP
from util.multi_threads_op import ThreadsOP, MultiThreadsProcess

# at project root directory:
#   python -m dataset.datamaker.bev_from_multi_fev
# at project dataset/datamaker/ directory
#   from calibrate_params import CalibrateParameter
#   but `util` import error
# need add system path or config vscode launch config
# not recommended method: python run.py at root dir

# sys.exit()

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.backends.cuda.matmul.allow_tf32 = False
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class DataMakerTorch(nn.Module):
    '''
    Deep Learning AVM Dataset Maker:
        torch GPU version: calc homo and warp imgs
        multi threads version: read and write images
        support: from undist to bev
        todo: from fev to bev
    Pipeline:
        after class instance initialization, run `def init_dataset_mode_info`
    Application:
        generator = DataMakerTorch(enable_cuda=True)
        # for mode in ['gt_bev']:
        # for mode in ['train', 'test']:
        for mode in ['gt_bev', 'train', 'test']:
            generator.init_dataset_mode_info(mode)
            pts = generator.generate_perturbed_points()
            bs_list = generator.batch_size_list
            for i in tqdm(range(generator.iteration)):
                src, name = generator.get_src_images(i, bs_list[i])
                generator.warp_perturbed_image(pts, src, name, bs_list[i])
            generator.shutil_copy()
    '''

    def __init__(self, enable_cuda=True):
        super().__init__()
        self.dataset_root = '/home/data/lwb/data'
        self.dataset_name = 'dybev'
        self.dataset_version = version = 'v5'
        self.dataset_sv_dir = os.path.join(self.dataset_root, self.dataset_name)
        self.dataset_fev_video_dir_name = 'AVM_record_ocr'
        self.generate_dir_name = 'generate'
        self.perturb_pts_json_name = 'perturbed_points.json'
        self.detected_pts_json_name = 'detected_points.json'
        self.file_record_txt_name = 'image_names.txt'
        self.code_data_dir = code_data_dir = os.path.join('dataset', 'data')
        self.calib_params = EasyDict(CalibrateParameter().__dict__)
        self.__dict__.update(self.calib_params)
        self.threads_num = 32
        self.batch_size = bs = 32
        self.batch_size = bs = self._init_make_batch_divisible(bs, self.threads_num)
        self.drop_last_mismatch_batch = False
        self.mtp = MultiThreadsProcess()
        self.src_img_mode_key_name = EasyDict(undist='undist', fev='fev')
        self.src_img_mode = self.src_img_mode_key_name.fev
        # self.src_img_mode = self.src_img_mode_key_name.undist
        bev_mode = 'fev2bev' if self.src_img_mode == 'fev' else 'undist2bev'
        self.enable_cuda = enable_cuda
        self.offset_pix_range = 15
        self.perturb_mode = 'uniform'  # randint
        # the idx=0 image don't perturb, and the idx=1 to idx=1000 images are perturbed
        self.gt_bev_pertrubed_num = 1
        self.train_pertrubed_num = 10  # ten pictures
        self.test_pertrubed_num = 2
        self.is_split_videos_train_test = True
        self.split_test_ratio = 0.1
        self.num_total_images = -1
        self.dlt_homo_method = 'Axb'
        self.perturbed_points_index = index = [0, 3, 4, 7]  # four corners
        self.camera_fblr = ["front", "back", "left", "right"]
        self.align_fblr = True  # fblr four cameras
        self.perturbed_image_type = 'bev'
        self.perturbed_pipeline = bev_mode
        self.pts_path = os.path.join(code_data_dir, self.detected_pts_json_name)
        self.dataset_dir = os.path.join(self.dataset_sv_dir, version)
        # if os.path.exists(self.dataset_dir):
        #     shutil.rmtree(self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.gt_bev_dir = os.path.join(self.dataset_dir, 'gt_bev')
        self._init_json_key_names()
        self._init_read_points(index)
        self._init_torch_operation(self.batch_size)
        self.assign_sample_num = -1
        self.assign_sample_mode = 'sequence'  # random

    def _init_make_batch_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _init_torch_operation(self, batch_size):
        wh_fblr = self.bev_wh_fblr
        is_gpu = self.enable_cuda
        self.homo_torch_op = HomoTorchOP(method=self.dlt_homo_method)
        self.warp_torch_op = {}
        for x in self.camera_fblr:
            self.warp_torch_op[x] = WarpTorchOP(batch_size, *wh_fblr[x], is_gpu, 0)
        self.remap_table_torch_op = RemapTableTorchOP(self.calib_params, is_gpu)

        return

    def _init_split_train_test_videos(self, dataset_init_mode):
        self.fub_dir_name = 'fev_undist_bev'
        self.source_videos_list = [
            '20221205135519',
            '20221205135629',
            '20221205135816',
            '20221205140313',
            '20221205140456',
            '20221205140548',
            '20221205140642',
            '20221205141246',
            '20221205141401',
            '20221205141455',
        ]
        self.src_img_dir = os.path.join(
            self.dataset_sv_dir, self.fub_dir_name, self.src_img_mode
        )
        assert self.align_fblr == True, 'only support four cameras'
        # all cameras have the same number of images
        self.any_camera = 'front'
        self.cam_img_dir_path = os.path.join(self.src_img_dir, self.any_camera)
        # do this: os.listdir(filter )
        cam_img_name_list = cinl = os.listdir(self.cam_img_dir_path)
        if dataset_init_mode in ['train', 'test'] and self.is_split_videos_train_test:
            vid_num = self.split_test_ratio * len(self.source_videos_list)
            self.test_vid_num = max(1, int(vid_num))
            self.test_vid_names = self.source_videos_list[-self.test_vid_num :]
            self.train_vid_names = self.source_videos_list[: -self.test_vid_num]
            if dataset_init_mode == 'train':
                vid_names = self.train_vid_names
            elif dataset_init_mode == 'test':
                vid_names = self.test_vid_names
            new_cinl = list(filter(lambda x: x.split('_')[0] in vid_names, cinl))
            cam_img_name_list = new_cinl
        return cam_img_name_list

    def _init_src_paths_warp_to_bev(
        self, assign_sample_num, batch_size, cam_img_name_list
    ):
        self.src_img_path_record_txt = os.path.join(
            self.dataset_sv_dir, self.fub_dir_name, self.file_record_txt_name
        )
        cinl = cam_img_name_list
        asm = assign_sample_num
        assert len(cinl) > batch_size
        if self.assign_sample_mode == 'random' and asm > 0:
            cinl = random.shuffle(cinl)
        elif self.assign_sample_mode == 'sequence':
            cinl = sorted(cinl)
        asm = len(cinl) if min(asm, len(cinl)) <= 0 else asm
        self.assign_sample_num = asm = max(asm, batch_size)
        self.cam_img_name_list = cinl[:asm]
        num_img_per_camera = len(self.cam_img_name_list)
        os.path.join(self.gt_bev_dir, 'generate')
        return num_img_per_camera

    def _init_get_perturbed_points_info(self, dataset_init_mode):
        mode = dataset_init_mode
        if mode == 'gt_bev':
            self.num_generated_points = self.gt_bev_pertrubed_num
            self.delta_num = 0
        elif mode in ['train']:
            self.num_generated_points = self.train_pertrubed_num
            self.delta_num = 0
        elif mode == 'test':
            self.num_generated_points = self.test_pertrubed_num
            self.delta_num = self.train_pertrubed_num
        else:
            raise ValueError
        self.dataset_mode_dir = set_dir = os.path.join(self.dataset_dir, mode)
        # rm first
        if os.path.exists(set_dir):
            shutil.rmtree(set_dir)
        os.makedirs(set_dir)
        self.perturb_pts_path = os.path.join(set_dir, self.perturb_pts_json_name)
        self.generate_dir = os.path.join(set_dir, self.generate_dir_name)

    def _init_read_points(self, index=[0, 3, 4, 7]):
        with open(self.pts_path, "r") as f:
            pts = json.load(f)
            kn_ = self.key_name_pts_detect
        pt_src_fblr = {}
        pt_dst_fblr = {}
        if index is None:
            index = self.perturbed_points_index
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
        # cpu or gpu, got it here for the first time
        self.device = pt_src_fblr[camera].device
        self.pt_src_fblr = pt_src_fblr
        self.pt_dst_fblr = pt_dst_fblr
        return

    def _init_batch_and_iteration(self, batch_size, num_img_per_camera):
        size = num_img_per_camera
        batch = min(batch_size, size)
        iteration = int(size / batch)
        batch_size_list = [batch] * iteration
        if size % batch != 0:
            if not self.drop_last_mismatch_batch:
                batch_size_list.append(size % batch)
                iteration += 1
        prt_str = (
            f' \n'
            + f' camera_list: {self.camera_fblr} \n'
            + f' num_img_per_camera: {num_img_per_camera} \n'
            + f' batch_size: {self.batch_size} \n'
            + f' iteration: {iteration} \n'
            + f' threads_num: {self.threads_num} \n'
            + f' gt_bev_pertrubed_num: {self.gt_bev_pertrubed_num} \n'
            + f' train_pertrubed_num: {self.train_pertrubed_num} \n'
            + f' test_pertrubed_num: {self.test_pertrubed_num} \n'
            + f' \n'
        )
        print(prt_str)
        return batch_size_list, iteration

    def _init_json_key_names(self):
        self.key_name_pts_perturb = EasyDict(
            dataset_version="dataset_version",
            perturbed_image_type="perturbed_image_type",
            perturbed_pipeline="perturbed_pipeline",
            dlt_homography_method="dlt_homography_method",
            make_date="make_date",
            random_mode="random_mode",
            gt_bev_pertrubed_num="gt_bev_pertrubed_num",
            train_pertrubed_num="train_pertrubed_num",
            test_pertrubed_num="test_pertrubed_num",
            camera_list="camera_list",
            offset_pixel_range="offset_pixel_range",
            original_points="original_points",
            perturbed_points_list="perturbed_points_list",
            offset_list="offset_list",
        )
        self.key_name_pts_detect = EasyDict(
            detected_points='detected_points',
            corner_points='corner_points',
            homo='homo',
        )
        return

    def init_dataset_mode_info(self, mode='train'):
        assert mode in ['gt_bev', 'train', 'test']
        print(f'\n ---------- init_mode: {mode} ---------- ')
        if mode in ['train', 'test']:
            if not os.path.exists(self.gt_bev_dir):
                print('`gt_bev` dir need to be created first')
                print('sys.exit()')
                sys.exit()
        self.dataset_init_mode = mode
        # source images info
        self.cam_img_name_list = cinl = self._init_split_train_test_videos(mode)
        asn, bs = self.assign_sample_num, self.batch_size
        self.num_img_per_camera = nips = self._init_src_paths_warp_to_bev(asn, bs, cinl)
        self.batch_size_list, self.iteration = self._init_batch_and_iteration(bs, nips)
        # perturbed points info
        self._init_get_perturbed_points_info(mode)

    def get_src_images(self, idx=None, align_fblr=None):

        if align_fblr is None:
            align_fblr = self.align_fblr
        src_fblr = None
        nam_fblr = None

        # for multiprocessing
        def _read_image_kernel(img_path, name):
            img = cv2.imread(img_path)
            if self.src_img_mode == self.src_img_mode_key_name.undist:
                if self.new_scale_for_undist != self.scale_previous_value:  # 1.0 vs 0.5
                    r_scale = int(self.new_scale_for_undist / self.scale_previous_value)
                    wh = tuple([int(x / r_scale) for x in img.shape[:2]][::-1])  # / 2
                    img = cv2.resize(img, wh, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img).unsqueeze(0)
            return img, name

        if align_fblr:
            src_fblr, nam_fblr = {}, {}
            for camera in self.camera_fblr:
                src_fblr[camera] = []
                nam_fblr[camera] = []
            for camera in self.camera_fblr:
                if idx is not None:
                    cam_dir = os.path.join(self.src_img_dir, camera)
                    name_list = self.cam_img_name_list[
                        idx * self.batch_size : (idx + 1) * self.batch_size
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
                            args = (file_path, name)
                            # input
                            thr = ThreadsOP(_read_image_kernel, args)
                            threads.append(thr)
                        # process
                        for thr in threads:
                            thr.start()
                        for thr in threads:
                            thr.join()
                        for thr in threads:
                            # postprocess
                            img, name = thr.get_result()
                            # output
                            src_fblr[camera].append(img)
                            nam_fblr[camera].append(name)
                else:
                    # read all images in one batch at once
                    cam_dir = os.path.join(self.src_img_dir, camera)
                    file_name_list = os.listdir(cam_dir)
                    if len(file_name_list) > 64:
                        raise Exception("Too many images to feed into list")
                    for name in file_name_list:
                        file_path = os.path.join(cam_dir, name)
                        img, _ = _read_image_kernel(file_path, name)
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
            elif random_mode == 'translation':
                # TODO
                pass
            elif random_mode == 'rotation':
                pass
            else:
                raise ValueError("Need to add")
            if i == self.delta_num:
                if self.dataset_init_mode in ['gt_bev', 'train']:
                    offset = np.zeros_like(pts)  # test don't need
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
            kn1_.dlt_homography_method: self.dlt_homo_method,
            kn1_.make_date: time.strftime('%z %Y-%m-%d %H:%M:%S', time.localtime()),
            kn1_.random_mode: self.perturb_mode,
            kn1_.train_pertrubed_num: self.train_pertrubed_num,
            kn1_.test_pertrubed_num: self.test_pertrubed_num,
            kn1_.camera_list: self.camera_fblr,
        }
        print(pts_perturb)
        print(f'dataset_dir:  {self.dataset_dir}')
        print(f'dataset mode: {self.dataset_init_mode} \n')
        if index is None:
            index = self.perturbed_points_index
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

    def warp_perturbed_image(self, pts_perturb, src_fblr, name_fblr):
        pt_undist_fblr = self.pt_src_fblr
        _kn1 = self.key_name_pts_perturb
        _kn2 = self.src_img_mode_key_name

        # for multiprocessing
        def _threads_kernel(sv_path, img):
            is_success = cv2.imwrite(sv_path, img)
            return is_success

        def _preprocess(i, j, threads_num, dst, names, save_dir, idx):
            img = dst[i * threads_num + j]
            name = names[i * threads_num + j]
            name = name.replace('.', f'_p{idx:04}.')
            sv_path = os.path.join(save_dir, name)
            return (sv_path, img)

        def _postprocess(is_success):
            if not is_success:
                raise Exception("imwrite error")
            return

        for camera in self.camera_fblr:
            names = name_fblr[camera]
            src_cam = src_fblr[camera]
            save_dir = os.path.join(self.generate_dir, camera)
            os.makedirs(save_dir, exist_ok=True)
            for k in range(self.delta_num, self.delta_num + self.num_generated_points):
                # perturbed points on bev
                idx = f'{k:04}'  # fixed index format for generating perturbed points
                pt_perturbed = pts_perturb[camera][_kn1.perturbed_points_list][idx]
                pt_perturbed = torch.Tensor(pt_perturbed).reshape(1, -1, 2)
                if self.enable_cuda:
                    pt_perturbed = pt_perturbed.cuda()
                pt_undist = pt_undist_fblr[camera]
                if self.src_img_mode == _kn2.undist:
                    H_u2b = self.homo_torch_op(pt_undist, pt_perturbed)
                    H_b2u = torch.inverse(H_u2b)
                    H_b2u = H_b2u.repeat(len(names), 1, 1)
                    dst = self.warp_torch_op[camera](src_cam, H_b2u, None, 'by_homo')
                elif self.src_img_mode == _kn2.fev:
                    H_u2b = self.homo_torch_op(pt_undist * 2, pt_perturbed)
                    H_b2u = torch.inverse(H_u2b)
                    grids = self.remap_table_torch_op(H_b2u, camera, mode='f2b')
                    grids = grids.repeat(len(names), 1, 1, 1)
                    dst = self.warp_torch_op[camera](src_cam, None, grids, 'by_grid')
                dst = dst.transpose(1, 2).transpose(2, 3)  # bchw -> bhwc
                dst = dst.detach().cpu().numpy()
                # save
                input_values = (self.threads_num, dst, names, save_dir, k)
                self.mtp.multi_threads_process(
                    input_values=input_values,
                    batch_size=len(dst),
                    threads_num=self.threads_num,
                    func_thread_kernel=_threads_kernel,
                    func_preprocess=_preprocess,
                    func_postprocess=_postprocess,
                )
        return

    def shutil_copy(self):
        set_dir = self.dataset_mode_dir
        code_data_dir = self.code_data_dir
        kn_ = self.key_name_pts_detect
        shutil.copy(
            os.path.join(code_data_dir, f'{kn_.detected_points}.json'),
            os.path.join(set_dir, f'{kn_.detected_points}.json'),
        )
        shutil.copy(
            os.path.join(code_data_dir, f'{kn_.homo}.json'),
            os.path.join(set_dir, f'{kn_.homo}.json'),
        )
        if (
            self.dataset_init_mode in ['train', 'test']
            and self.perturbed_image_type == 'bev'
        ):
            # fev or undist
            dst_img_dir = os.path.join(set_dir, self.src_img_mode)
            shutil.copytree(self.src_img_dir, dst_img_dir)
            # bev
            src_bev_dir = os.path.join(self.dataset_dir, 'gt_bev', 'generate')
            dst_bev_dir = os.path.join(set_dir, 'bev')
            shutil.copytree(src_bev_dir, dst_bev_dir)
        else:
            # TODO
            pass

        return


def run_bev_from_multi_fev():
    generator = DataMakerTorch(enable_cuda=True)
    # for mode in ['gt_bev']:
    # for mode in ['test']:
    # for mode in ['gt_bev', 'test']:
    # for mode in ['train', 'test']:
    for mode in ['gt_bev', 'train', 'test']:
        generator.init_dataset_mode_info(mode)
        pts = generator.generate_perturbed_points()
        for i in tqdm(range(generator.iteration)):
            src, name = generator.get_src_images(i)
            generator.warp_perturbed_image(pts, src, name)
        generator.shutil_copy()


if __name__ == "__main__":
    '''
    import other packages, run this .py at other position
    copy the following codes into other .py file on project root directory
    '''
    run_bev_from_multi_fev()
