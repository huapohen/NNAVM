import torch
import torch.nn as nn
from easydict import EasyDict
from dataset.datamaker.calibrate_params import CalibrateParameter

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class RemapTableTorchOP(object):
    '''
    two mode: 1. fev2bev; 2. undist2bev
    if H_bev2undist is not None, mode is fev2bev: feb <- undist <- bev
    input  shape:
        H_bev2undist: camera_front bhw=(b,336,1078) b=1
    output shape:
        if u2b: bhwc=(b, 2976, 3968, 2)
        if f2b: bhwc=(b, bev_h, bev_w, 2)
    '''

    def __init__(self, calibrate_parameter, is_gpu=False):
        super().__init__()
        assert isinstance(calibrate_parameter, dict)
        self.__dict__.update(calibrate_parameter)
        self.device = torch.cuda.current_device() if is_gpu else 'cpu'
        # init meshgrid in here whether lead to not transfer gradient ?
        self.remap_table_undist2bev = self.get_undist2bev_remap_table()
        self.grids_b2u_fblr = self._init_bev2undist_grids()

    def undist_grids_to_fev(self, grids):
        dv = self.device
        ele1 = torch.tensor([self.undist_w / 2, self.undist_h / 2], device=dv)
        ele2 = torch.tensor([self.center_w, self.center_h], device=dv)
        ele3 = torch.tensor([self.intrinsic[0][0], self.intrinsic[1][1]], device=dv)
        undist_center, dist_center, f = [
            ele.reshape(1, 1, 1, 2) for ele in [ele1, ele2, ele3]
        ]
        grids = grids - undist_center
        grids_norm = grids / f
        r_undist = torch.linalg.norm(grids_norm, dim=-1)
        angle_undistorted = torch.atan(r_undist)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        coefs = []
        for i in range(4):
            coefs.append(torch.tensor(self.distort[f'Opencv_k{i}'], device=dv))
        r_distort = (
            angle_undistorted
            + coefs[0] * angle_undistorted_p3
            + coefs[1] * angle_undistorted_p5
            + coefs[2] * angle_undistorted_p7
            + coefs[3] * angle_undistorted_p9
        )
        min_value = torch.tensor(1e-6, device=dv)
        scale = torch.div(r_distort, r_undist.clamp(min_value))
        scale = scale.unsqueeze(-1)
        grids = grids * scale + dist_center

        return grids

    def get_undist2bev_remap_table(self):
        h = self.undist_h
        w = self.undist_w
        grid_x, grid_y = torch.meshgrid(  # grid_i, grid_j
            torch.arange(w, device=self.device),  # h
            torch.arange(h, device=self.device),  # w
            indexing='xy',  # 'ij'
        )
        grids = torch.stack([grid_x, grid_y], dim=2)
        grids = grids.unsqueeze(0)

        grids = self.undist_grids_to_fev(grids)

        return grids

    def _init_bev2undist_grids(self):
        grids_b2u_fblr = {}
        for camera, wh in self.bev_wh_fblr.items():
            w, h = wh
            grid_x, grid_y = torch.meshgrid(  # grid_i, grid_j
                torch.arange(w, device=self.device),  # h
                torch.arange(h, device=self.device),  # w
                indexing='xy',  # 'ij'
            )
            grid_v = grid_x.new_ones(*grid_x.shape)
            grids = torch.stack([grid_x, grid_y, grid_v], dim=2)
            grids = grids.reshape((-1, 3)).transpose(1, 0).float()
            grids_b2u_fblr[camera] = grids

        return grids_b2u_fblr

    def get_fev2bev_remap_table(self, H_bev2undist, camera):
        w, h = self.bev_wh_fblr[camera]
        grids = self.grids_b2u_fblr[camera]
        grids = torch.matmul(H_bev2undist, grids)
        grids = grids.transpose(2, 1).reshape((-1, h, w, 3))
        grids = grids / grids[:, :, :, 2:]
        grids = grids[:, :, :, 0:2]

        grids = self.undist_grids_to_fev(grids)

        return grids

    def __call__(self, H_bev2undist=None, camera=None, mode='f2b'):
        assert mode in ['u2b', 'f2b']
        if mode == 'u2b':
            return self.remap_table_undist2bev
        elif mode == 'f2b':
            return self.get_fev2bev_remap_table(H_bev2undist, camera)


class HomoTorchOP(object):
    def __init__(self, method='Axb'):
        super().__init__()
        self.dlt_homo_method = method

    def dlt_homo(self, src_pt, dst_pt, method="Axb"):
        """
        :param src_pt: shape=(batch, num, 2)
        :param dst_pt:
        :param method: Axb (Full Rank Decomposition, inv_SVD) = 4 piar points
                    Ax0 (SVD) >= 4 pair points, 4,6,8
        :return: Homography, shape: (batch, 3, 3)
        """
        method = self.dlt_homo_method
        assert method in ["Ax0", "Axb"]
        assert src_pt.shape[1] >= 4
        assert dst_pt.shape[1] >= 4
        if method == 'Axb':
            assert src_pt.shape[1] == 4
            assert dst_pt.shape[1] == 4
        batch_size, num_pts = src_pt.shape[:2]
        bs = batch_size
        xy1 = torch.cat([src_pt, src_pt.new_ones(bs, num_pts, 1)], dim=-1)
        xyu = torch.cat([xy1, xy1.new_zeros(bs, num_pts, 3)], dim=-1)
        xyd = torch.cat([xy1.new_zeros(bs, num_pts, 3), xy1], dim=-1)
        src_pt = src_pt.view(-1, 1, 2)
        dst_pt = dst_pt.view(-1, 2, 1)
        M1 = torch.cat([xyu, xyd], dim=-1).view(bs, -1, 6)
        M2 = torch.matmul(dst_pt, src_pt).view(bs, -1, 2)
        M3 = dst_pt.view(bs, -1, 1)

        if method == "Ax0":
            A = torch.cat([M1, -M2, -M3], dim=-1)
            U, S, V = torch.svd(A)
            V = V.transpose(-2, -1).conj()
            H = V[:, -1].view(bs, 3, 3)
            H = H * (1 / H[:, -1, -1].view(bs, 1, 1))
        elif method == "Axb":
            A = torch.cat([M1, -M2], dim=-1)
            B = M3
            A_inv = torch.inverse(A)
            # 矩阵乘 用 gpu算出来结果不对
            # 转cpu
            # 结果用当前device储存
            # mm = torch.matmul(A_inv.cpu(), B.cpu()).to(A)
            # 关闭这个：torch.backends.cuda.matmul.allow_tf32 = False
            mm = torch.matmul(A_inv, B)
            H = torch.cat([mm.view(-1, 8), A.new_ones(bs, 1)], 1)
            H = H.view(bs, 3, 3)

        return H

    def __call__(self, src_pt, dst_pt):

        H = self.dlt_homo(src_pt, dst_pt, self.dlt_homo_method)

        return H


class WarpTorchOP(object):
    def __init__(self, batch_size, w, h, enable_cuda=True, start=0):
        super().__init__()
        self.h = h
        self.w = w
        self.enable_cuda = enable_cuda
        # init meshgrid in here whether lead to not transfer gradient ?
        # but, we don't need to save the meshgrid values
        # see nn.Pamameter versus register_buffer
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
        pred_I2_index_warp = torch.cat([v1, v2], 1)
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

    def __call__(self, src=None, H=None, grids=None, mode='by_grid'):
        assert mode in ['by_homo', 'by_grid']
        if mode == 'by_homo':
            grid = self.grid[: src.shape[0]]
            flow, vgrid = self.get_flow_vgrid(H, grid, self.h, self.w, 1)
            grids = vgrid + flow
        dst = self.warp_image(src, grids)
        return dst


def warp_image_f2b(params, bs, homo, fev_fblr):
    calib_param = EasyDict(CalibrateParameter().__dict__)
    remap_table_torch_op = RemapTableTorchOP(calib_param, params.cuda)
    dst = []
    for i, cam in enumerate(params.camera_list):
        h_b2u = homo[i * bs : (i + 1) * bs]
        grids = remap_table_torch_op(h_b2u, cam, mode='f2b')
        grids = grids.permute(0, 3, 1, 2).contiguous()
        warp_torch_op = WarpTorchOP(bs, *params.wh_bev_fblr[cam], params.cuda, 0)
        dst_cam = warp_torch_op(fev_fblr[i], None, grids, 'by_grid')
        dst.append(dst_cam)
    return dst


if __name__ == '__main__':

    '''
    if self.src_img_mode == _kn2.undist:
        H_u2b = self.homo_torch_op(pt_undist, pt_perturbed)
        H_b2u = torch.inverse(H_u2b)
        H_b2u = H_b2u.repeat(len(names), 1, 1)
        dst = self.warp_torch_op[camera](src_cam, H_b2u, None, 'by_homo')
    elif self.src_img_mode == _kn2.fev:
        H_u2b = self.homo_torch_op(pt_undist * 2, pt_perturbed)
        H_b2u = torch.inverse(H_u2b)
        grids = self.remap_table_torch_op(H_b2u, camera, mode='f2b')
        grids = grids.permute(0, 3, 1, 2).contiguous()
        grids = grids.repeat(len(names), 1, 1, 1)
        dst = self.warp_torch_op[camera](src_cam, None, grids, 'by_grid')
    '''
