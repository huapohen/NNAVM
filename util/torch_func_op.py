import cv2
import sys
import torch
import numpy as np
import torch.nn as nn


def dlt_homo(src_pt, dst_pt, method="Axb"):
    """
    :param src_pt: shape=(batch, num, 2)
    :param dst_pt:
    :param method: Axb (Full Rank Decomposition, inv_SVD) = 4 piar points
                Ax0 (SVD) >= 4 pair points, 4,6,8
    :return: Homography, shape=(batch, 3, 3)
    """
    assert method in ["Ax0", "Axb"]
    assert src_pt.shape[1] >= 4
    assert dst_pt.shape[1] >= 4
    if method == 'Axb':
        assert src_pt.shape[1] == 4
        assert dst_pt.shape[1] == 4
    batch_size, nums_pt = src_pt.shape[0], src_pt.shape[1]
    xy1 = torch.cat((src_pt, src_pt.new_ones(batch_size, nums_pt, 1)), dim=-1)
    xyu = torch.cat((xy1, xy1.new_zeros((batch_size, nums_pt, 3))), dim=-1)
    xyd = torch.cat((xy1.new_zeros((batch_size, nums_pt, 3)), xy1), dim=-1)
    M1 = torch.cat((xyu, xyd), dim=-1).view(batch_size, -1, 6)
    M2 = torch.matmul(dst_pt.view(-1, 2, 1), src_pt.view(-1, 1, 2)).view(
        batch_size, -1, 2
    )
    M3 = dst_pt.view(batch_size, -1, 1)

    if method == "Ax0":
        A = torch.cat((M1, -M2, -M3), dim=-1)
        U, S, V = torch.svd(A)
        V = V.transpose(-2, -1).conj()
        H = V[:, -1].view(batch_size, 3, 3)
        H = H * (1 / H[:, -1, -1].view(batch_size, 1, 1))
    elif method == "Axb":
        A = torch.cat((M1, -M2), dim=-1)
        B = M3
        A_inv = torch.inverse(A)
        H = torch.cat(
            (
                torch.matmul(A_inv, B).view(-1, 8),
                src_pt.new_ones((batch_size, 1)),
            ),
            1,
        ).view(batch_size, 3, 3)

    return H


def get_meshgrid(h, w, device):
    grid_x, grid_y = torch.meshgrid(  # grid_i, grid_j
        torch.arange(w, device=device),  # h
        torch.arange(h, device=device),  # w
        indexing='xy',  # 'ij'
    )
    grid_v = grid_x.new_ones(*grid_x.shape)
    grids = torch.stack([grid_x, grid_y, grid_v], dim=2)
    return grids


def get_grid(batch_size, h, w, start=0, enable_cuda=True):
    """
    this grid same as twice for loop
    start: start point coordinate in an image
    start.shape: (N,1,2), default value: 0
    """
    if enable_cuda:
        xx = torch.arange(0, w).cuda()
        yy = torch.arange(0, h).cuda()
    else:
        xx = torch.arange(0, w)
        yy = torch.arange(0, h)
    xx = xx.view(1, -1).repeat(h, 1)
    yy = yy.view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)

    if enable_cuda:
        ones = torch.ones_like(xx).cuda()
    else:
        ones = torch.ones_like(xx)

    grid = torch.cat((xx, yy, ones), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + start
    return grid


def bilinear_interpolate(im, x, y, out_size, enable_cuda=True):
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


def get_flow_vgrid(H_mat_mul, patch_indices, patch_size_h, patch_size_w, divide=1):
    """
    patch_indices: this is grid
    divide: deblock used for mesh-flow
    output flow and vgrid
    """
    batch_size = H_mat_mul.shape[0]
    small_gap_sz = [patch_size_h // divide, patch_size_w // divide]

    small = 1e-7

    H_mat_pool = H_mat_mul.reshape(batch_size, divide, divide, 3, 3)  # .transpose(2,1)
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


def undist_grids_to_fev(self, grids, is_out_bchw=True):
    # input grids shape: b h w c
    dv = grids.device
    ele1 = torch.tensor([self.undist_w / 2, self.undist_h / 2], device=dv)
    ele2 = torch.tensor([self.center_w, self.center_h], device=dv)
    ele3 = torch.tensor([self.intrinsic[0][0], self.intrinsic[1][1]], device=dv)
    undist_center, dist_center, f = [
        ele.reshape(1, 1, 1, 2) for ele in [ele1, ele2, ele3]
    ]
    grids = grids - undist_center
    r_undist = torch.linalg.norm(grids.div(f), dim=-1)
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

    if is_out_bchw:
        grids = grids.permute(0, 3, 1, 2).contiguous()

    return grids


def warp_image(I, vgrid, train=True, enable_cuda=True):
    """
    torch.nn.functional.grid_sample
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
    input_transformed = bilinear_interpolate(
        I, x_s_flat, y_s_flat, out_size, enable_cuda
    )

    output = input_transformed.reshape([b, h, w, C_img])

    if train:
        output = output.permute(0, 3, 1, 2).contiguous()
    return output


def warp_image_to_bev(params, batch_size, homo, src):
    dst = []
    for i, cam in enumerate(params.camera_list):
        w, h = params.wh_bev_fblr[cam]
        grid = get_grid(batch_size, h, w, 0, params.cuda)
        h_this = homo[i * batch_size : (i + 1) * batch_size]
        flow, vgrid = get_flow_vgrid(h_this, grid, h, w, 1)
        grids = vgrid + flow
        if params.src_img_mode == 'fev':
            grids = grids.permute(0, 2, 3, 1).contiguous()
            grids = undist_grids_to_fev(params.calib_param, grids)
        dst_cam = warp_image(src[i], grids)
        dst.append(dst_cam)
    return dst


def unit_test_warp_image():
    '''
    python -m util.torch_func_op
    '''
    import ipdb
    import numpy as np
    from easydict import EasyDict
    from dataset.datamaker.calibrate_params import CalibrateParameter

    calib_param = EasyDict(CalibrateParameter().__dict__)

    dx = 300
    dy = 0
    offset = np.array([dx, dy, dx, dy, dx, dy, dx, dy])
    pt_src = np.array([732, 784, 1236, 781, 282, 946, 1601, 916])
    pt_dst = np.array([309, 216, 769, 216, 309, 316, 769, 316])
    pt_dst += offset
    pt_src = torch.Tensor(pt_src).float().reshape(1, -1, 2).cuda()
    pt_dst = torch.Tensor(pt_dst).float().reshape(1, -1, 2).cuda()
    h, w = 336, 1078
    # ipdb.set_trace()
    undist = cv2.imread('dataset/data/undist/front.jpg')
    undist = torch.tensor(undist).permute(2, 0, 1).contiguous().unsqueeze(0).cuda()
    h_b2u = dlt_homo(pt_dst, pt_src)
    grid = get_grid(1, h, w, 0, True)
    flow, vgrid = get_flow_vgrid(h_b2u, grid, h, w, 1)
    grids = vgrid + flow
    bev_u2b = warp_image(undist, grids)
    bev_u2b = bev_u2b.detach().cpu()[0].permute(1, 2, 0).numpy()
    cv2.imwrite(f'model/vis/bev_u2b.jpg', bev_u2b)
    # ipdb.set_trace()
    fev = cv2.imread('dataset/data/fev/front.png')
    fev = torch.tensor(fev).permute(2, 0, 1).contiguous().unsqueeze(0).cuda()
    h_b2u = dlt_homo(pt_dst, pt_src * 2)
    flow, vgrid = get_flow_vgrid(h_b2u, grid, h, w, 1)
    grids = vgrid + flow
    grids = grids.permute(0, 2, 3, 1).contiguous()
    grids = undist_grids_to_fev(calib_param, grids)
    bev_f2b = warp_image(fev, grids)
    bev_f2b = bev_f2b.detach().cpu()[0].permute(1, 2, 0).numpy()
    cv2.imwrite(f'model/vis/bev_f2b.jpg', bev_f2b)
    # ipdb.set_trace()


if __name__ == '__main__':

    unit_test_warp_image()
