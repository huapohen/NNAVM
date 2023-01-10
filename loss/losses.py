import torch
import torch.nn.functional as F


def loss_supervised(params, data):
    offset = data['offset']
    offset_pred = data['offset_pred']
    batch_size = int(data['image'].shape[0] / len(params.camera_list))
    losses = []
    for i, cam in enumerate(params.camera_list):
        pred = offset_pred[i : (i + 1) * batch_size]
        gt = offset[i : (i + 1) * batch_size]
        loss = F.mse_loss(pred, gt)
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses).mean()
    return losses


def loss_unsupervised(params, data):
    losses = []
    for i, cam in enumerate(params.camera_list):
        pred = data['bev_pred'][i]
        gt = data['bev_origin'][i]
        loss = F.l1_loss(pred, gt.float())
        losses.append(loss.unsqueeze(0))
    losses = torch.cat(losses).mean()
    return losses


def compute_losses(params, data):

    if params.model_train_type == 'supervised':
        losses = loss_supervised(params, data)
    elif params.model_train_type == 'unsupervised':
        losses = loss_unsupervised(params, data)
    else:
        raise ValueError

    return {"total_loss": losses}



if __name__ == "__main__":
    
    import ipdb
    import numpy as np
    from util.torch_op import dlt_homo

    ipdb.set_trace()
    src_pts = np.array([10, 10, 100, 10, 10, 50, 100, 50], dtype=np.float32).reshape(
        1, 4, 2
    )
    dst_pts = np.array(
        [10, 10 - 5, 100, 10 - 8, 10 + 5, 50, 100, 50 + 5], dtype=np.float32
    ).reshape(1, 4, 2)
    src_pts = torch.from_numpy(src_pts)
    dst_pts = torch.from_numpy(dst_pts)
    H = dlt_homo(src_pts, dst_pts)
    pts_ones = torch.ones(1, 4, 1)
    src_pts = torch.cat([src_pts, pts_ones], 2).permute(0, 2, 1)
    dstp_pts = H.matmul(src_pts)
    dstp_pts = dstp_pts.permute(0, 2, 1)
    dstp_pts = dstp_pts / dstp_pts[:, :, 2:]
    print(dstp_pts)
    print(dstp_pts[:, :, :2] - dst_pts)
