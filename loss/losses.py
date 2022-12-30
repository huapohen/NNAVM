import torch
import torch.nn.functional as F
import numpy as np
from dataset.data_maker import *
import warnings

warnings.filterwarnings("ignore")


def dlt_homo(src_pt, dst_pt, method="Axb"):
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
        H *= 1 / H[:, -1, -1].view(batch_size, 1, 1)
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


def get_coords(params):
    setname = params.train_data_ratio[0][0]
    path = f'{params.data_dir}/{setname}/train/detected_points.json'
    with open(path, 'r') as f:
        pts = json.load(f)
    src_coords, dst_coords = [], []
    for camera in ['front', 'back', 'left', 'right']:
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
    # return src_coords, dst_coords
    return dst_coords


def get_origin_undist():
    undist_fblr = []
    for camera in ['front', 'back', 'left', 'right']:
        img = cv2.imread(f'./dataset/data/undist/{camera}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = img.cuda()
        undist_fblr.append(img)
    return undist_fblr


def get_bev_ori(batch_size):
    bev_ori_fblr = []
    for camera in ['front', 'back', 'left', 'right']:
        img = cv2.imread(f'./dataset/data/bev/undist2bev/{camera}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        img = img.cuda()
        bev_ori_fblr.append(img)
    return bev_ori_fblr


def warp_image(I, vgrid, train=True, enable_cuda=True):
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
    input_transformed = bilinear_interpolate(
        I, x_s_flat, y_s_flat, out_size, enable_cuda
    )

    output = input_transformed.reshape([b, h, w, C_img])

    if train:
        output = output.permute(0, 3, 1, 2).contiguous()
    return output


def get_warp_image(bs, homo, src):
    wh_fblr = [(1078, 336), (1078, 336), (1172, 439), (1172, 439)]
    dst = []
    for i in range(4):
        w, h = wh_fblr[i]
        grid = get_grid(bs, h, w, 0)
        h_this = homo[i::4]
        flow, vgrid = get_flow_vgrid(h_this, grid, h, w, 1)
        # src_this = src[i::4].unsqueeze(0).repeat(bs, 1, 1, 1)
        src_this = torch.stack(src[i::4])
        dst_this = warp_image(src_this, vgrid + flow)
        dst.append(dst_this)
    return dst


def visualize(param, pred, gt, k=0):
    # [(1078, 336), (1078, 336), (1172, 439), (1172, 439)]
    sv_dir = f'{param.model_dir}/vis'
    os.makedirs(sv_dir, exist_ok=True)
    imgs = []
    for i in range(4):
        i1 = pred[i][k].detach().cpu().numpy()
        i2 = gt[i][k].detach().cpu().numpy()
        i3 = np.concatenate([i1, i2], axis=1)
        i4 = i3.transpose((1, 2, 0))
        imgs.append(i4)
    fb = np.concatenate([imgs[0], imgs[1]], axis=0)
    lr = np.concatenate([imgs[2], imgs[3]], axis=0)
    z1 = np.zeros([336 * 4, 1172 - 1078, 3])
    z2 = np.zeros([4 * (439 - 336), 1172, 3])
    fb = np.concatenate([fb, z1], axis=1)
    fb = np.concatenate([fb, z2], axis=0)
    all = np.concatenate([fb, lr], axis=1)
    cv2.imwrite(f'{sv_dir}/fblr.jpg', all)
    return


def compute_losses(param, delta, labels, bev_coords, bev_ori_fblr):
    delta = delta * param.max_shift_pixels
    delta = delta.view(delta.shape[0], 4, 2)
    bev_coords = bev_coords.repeat(param.train_batch_size, 1, 1)
    bev_coords_pred = bev_coords + delta
    homo = dlt_homo(bev_coords, bev_coords_pred, 'Axb')

    bev_warped_fblr = get_warp_image(param.train_batch_size, homo, labels)

    visualize(
        param,
        bev_warped_fblr,
        bev_ori_fblr,
    )
    losses = 0
    for i in range(4):
        loss = F.l1_loss(bev_warped_fblr[i], bev_ori_fblr[i].float())
        losses += loss.mean()
        # break
    return {"total": losses}
