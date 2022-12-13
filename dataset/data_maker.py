import os
import sys
import cv2
import math
import json
import time
import numpy as np


class DatasetMaker:
    def __init__(self):
        self.base_path = os.path.join(os.getcwd(), 'dataset/data')
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
            'front': {'w': 1078, 'h': 336},
            'back': {'w': 1078, 'h': 336},
            'left': {'w': 1172, 'h': 439},
            'right': {'w': 1172, 'h': 439},
        }
        self.camera_list = ['front', 'back', 'left', 'right']
        focal_len = hardware['focal_length']
        self.dx = dx = hardware["dx"] / fish['scale']
        self.dy = dy = hardware["dy"] / fish['scale']
        self.fish_width = distort_width = int(fish['width'] * fish['scale'])  # 640
        self.fish_height = distort_height = int(fish['height'] * fish['scale'])  # 480
        undis_scale = distort['undis_scale']
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
        # if 0:
        #     self.lut_undis = self.initial_theta_lut()
        # else:
        #     theta_txt_path = f'{self.base_path}/calib/theta.txt'
        #     with open(theta_txt_path, 'r') as f:
        #         lut_undis = f.readlines()
        #     self.lut_undis = [float(ele) for ele in lut_undis]

    def initial_theta_lut(self):
        lut_undis = []
        for i in range(self.num_theta):
            angle_undistorted = i / self.num_theta * math.atan(self.d)
            angle_undistorted_p2 = angle_undistorted * angle_undistorted
            angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
            angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
            angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
            angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
            angle_distorted = (
                angle_undistorted
                + self.distort['Opencv_k0'] * angle_undistorted_p3
                + self.distort['Opencv_k1'] * angle_undistorted_p5
                + self.distort['Opencv_k2'] * angle_undistorted_p7
                + self.distort['Opencv_k3'] * angle_undistorted_p9
            )
            lut_undis.append(angle_distorted)
        theta_txt_path = f'{self.base_path}/theta.txt'
        if os.path.exists(theta_txt_path):
            os.remove(theta_txt_path)
        with open(theta_txt_path, 'a') as f:
            for i, ele in enumerate(lut_undis):
                suf = '\n' if i < len(lut_undis) - 1 else ''
                f.write(str(ele) + suf)

        return lut_undis

    def calc_angle_undistorted(self, r_):
        angle_undistorted = math.atan(r_)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        angle_distorted = (
            angle_undistorted
            + self.distort['Opencv_k0'] * angle_undistorted_p3
            + self.distort['Opencv_k1'] * angle_undistorted_p5
            + self.distort['Opencv_k2'] * angle_undistorted_p7
            + self.distort['Opencv_k3'] * angle_undistorted_p9
        )

        return angle_distorted

    def get_remap_table(self, mode='fev2bev'):
        undist_center_w = int(self.undist_w / 2)
        undist_center_h = int(self.undist_h / 2)
        f_dx = self.intrinsic[0][0]
        f_dy = self.intrinsic[1][1]
        # lut_undis = self.lut_undis
        # num_theta = self.num_theta
        homo_fblr = self.get_homo_undist2bev()
        map_fblr = {}
        for camera in self.camera_list:
            map_x, map_y = [], []
            col = self.undist_w  # 1984
            row = self.undist_h  # 1488
            if mode == 'fev2bev':
                col = self.bev_wh_fblr[camera]['w']
                row = self.bev_wh_fblr[camera]['h']
                homo_b2u = np.linalg.inv(homo_fblr[camera])
            for i in range(row):
                row_x, row_y = [], []
                for j in range(col):
                    if mode == 'fev2bev':
                        jj, ii = self.matrix_mul_3x3(homo_b2u, j, i)
                    else:
                        jj, ii = j, i
                    x_ = (jj - undist_center_w) / f_dx
                    y_ = (ii - undist_center_h) / f_dy
                    r_ = math.sqrt(pow(x_, 2) + pow(y_, 2)) + 0.00000001
                    # idx = int(math.atan(r_) / math.atan(self.d) * (num_theta - 1))
                    # angle_distorted = lut_undis[idx]
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
            map_fblr[camera] = {'x': map_x, 'y': map_y}

        return map_fblr

    def get_homo_undist2bev(self, shift_func=None, is_save=False):
        pts_path = f'{self.base_path}/detected_points.json'
        with open(pts_path, 'r') as f:
            pts = json.load(f)
        bev_fblr_wh = [[1078, 336], [1078, 336], [1172, 439], [1172, 439]]
        homo_fblr = {}
        for idx, camera in enumerate(self.camera_list):
            # 8点法计算Homography
            pt_det = pts['detected_points'][camera]
            pt_bev = pts['corner_points'][camera]
            pt_det = [[[pt_det[i * 2], pt_det[i * 2 + 1]] for i in range(8)]]
            # ToDo 扰动 det_pts
            pt_det_array = np.asarray(pt_det)
            pt_bev = [[[pt_bev[i * 2], pt_bev[i * 2 + 1]] for i in range(8)]]
            pt_bev_array = np.asarray(pt_bev)
            homo_ransac, _ = cv2.findHomography(pt_det_array, pt_bev_array, 0)
            # 保存 H
            homo_fblr[camera] = homo_ransac
            if is_save:
                # img_undist = cv2.imread(f'{self.base_path}/vis/undist{camera[0]}.jpg')
                img_undist = cv2.imread(f'{self.base_path}/undist/{camera}.jpg')
                img_bev = cv2.warpPerspective(
                    img_undist, homo_ransac, bev_fblr_wh[idx], cv2.INTER_LINEAR
                )
                cv2.imwrite(f'{self.base_path}/undist2bev/{camera}.png', img_bev)
                map_x, map_y = [], []
                homo_b2u = np.linalg.inv(homo_fblr[camera])
                bev_w, bev_h = bev_fblr_wh[idx]
                for i in range(bev_h):
                    row_x, row_y = [], []
                    for j in range(bev_w):
                        jj, ii = self.matrix_mul_3x3(homo_b2u, j, i)
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
                cv2.imwrite(f'{self.base_path}/u2b/{camera}.png', bev)

        return homo_fblr

    def matrix_mul_3x3(self, H, j: int, i: int):
        div = H[2][0] * j + H[2][1] * i + H[2][2] * 1
        col_x = (H[0][0] * j + H[0][1] * i + H[0][2] * 1) / div
        row_y = (H[1][0] * j + H[1][1] * i + H[1][2] * 1) / div
        return col_x, row_y

    def read_fev_img(self):
        fev_fblr = {}
        for camera in self.camera_list:
            img_path = f'{self.base_path}/fev/{camera}.png'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.fish_width, self.fish_height))
            fev_fblr[camera] = img

        return fev_fblr

    def read_undist_img(self):
        undist_fblr = {}
        for camera in self.camera_list:
            img_path = f'{self.base_path}/undist/{camera}.jpg'
            img = cv2.imread(img_path)
            undist_fblr[camera] = img

        return undist_fblr

    def warp_img(self, src_fblr, map_fblr, is_save=True, save_mode='fev2bev'):
        assert src_fblr is not None
        dst_fblr = {}
        for i, camera in enumerate(self.camera_list):
            src = src_fblr[camera]
            map = map_fblr[camera]
            dst = cv2.remap(
                src,
                map['x'],
                map['y'],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            dst_fblr[camera] = dst
            if is_save:
                cv2.imwrite(f'{self.base_path}/{save_mode}/{camera}.png', dst)

        return dst_fblr


if __name__ == "__main__":

    datamaker = DatasetMaker()

    mode = 'fev2bev'
    # mode = 'fev2undist'

    img_fblr = datamaker.read_fev_img()
    # img_fblr = datamaker.read_undist_img()
    map_fblr = datamaker.get_remap_table(mode=mode)
    datamaker.warp_img(img_fblr, map_fblr, is_save=True, save_mode=mode)

    pass
