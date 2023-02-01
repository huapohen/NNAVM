import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util.torch_func_op import dlt_homo


class HomoWarpLayer(nn.Module):
    def __init__(self, out_w, out_h):
        super(HomoWarpLayer, self).__init__()
        self.out_w = out_w
        self.out_h = out_h
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, out_w), torch.arange(0, out_h), indexing='xy'
        )
        meshgrid_xy = torch.cat([grid_x[:, :, None], grid_y[:, :, None]], dim=2)
        meshgrid_xy = meshgrid_xy.reshape(1, -1, 2).float()
        mesh_ones = torch.ones(1, meshgrid_xy.shape[1], 1)
        meshgrid_xy = torch.cat([meshgrid_xy, mesh_ones], 2)
        meshgrid_xy = meshgrid_xy.permute(0, 2, 1)
        self.register_buffer("meshgrid_xy", meshgrid_xy)

    def forward(self, input, src_pts, dst_pts):
        """
        grid_sample input format: [-1, 1]
        :param input: input images (n, c, h, w)
        :param src_pts: src_pts (batch, num, 2)
        :param dst_pts: dst_pts (batch, num, 2)
        :return: warpd images
        """

        H_inv = dlt_homo(dst_pts, src_pts)
        warp_grids = H_inv.matmul(self.meshgrid_xy)  # bs, xyz, h*w
        warp_grids = warp_grids.permute(0, 2, 1)  # bs, h*w, xyz
        warp_grids = warp_grids / warp_grids[:, :, 2:]  # /z, normalilzation
        warp_grids = warp_grids.reshape(-1, self.out_h, self.out_w, 3)
        warp_grids[:, :, :, 0] = warp_grids[:, :, :, 0] / (self.out_w - 1) * 2 - 1
        warp_grids[:, :, :, 1] = warp_grids[:, :, :, 1] / (self.out_h - 1) * 2 - 1
        warp_img = F.grid_sample(input, warp_grids[:, :, :, :2], mode='bilinear')

        return warp_img


if __name__ == '__main__':
    import ipdb

    ipdb.set_trace()

    img = cv2.imread("test.jpg")
    input = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)[np.newaxis])

    h, w, _ = img.shape
    m = 0
    src_pts = np.array(
        [m, m, w - m, m, w - m, h - m, m, h - m], dtype=np.float32
    ).reshape(1, 4, 2)
    dst_pts = np.array(
        [
            m + 10,
            m + 10,
            w - m - 10,
            m + 10,
            w - m - 30,
            h - m - 50,
            m + 40,
            h - m - 30,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    src_pts = torch.from_numpy(src_pts)
    dst_pts = torch.from_numpy(dst_pts)

    homo_layer = HomoWarpLayer(img.shape[1], img.shape[0])
    warp_img = homo_layer(input, src_pts, dst_pts)

    # save
    warp_img = torch.clamp(warp_img, 0, 255)
    warp_img = warp_img[0].numpy()
    warp_img = warp_img.transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite('warp_test.jpg', warp_img)
