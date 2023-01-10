import os
import sys
import cv2
import ipdb
import torch
import shutil
import skimage
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def plot_indicator(params, pt_this, pt_gt, img_this, img_gt, mode):
    # not round 0.999* -> 1, reserve 0.999

    # psnr ssim
    # skimage.__version__
    # skimage.measure.compare_psnr
    mse = np.mean(np.square(img_this / 255.0 - img_gt / 255.0))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    # psnr = 10 * np.log10(1.0 / mse)
    # mse = np.mean(np.square(img_this - img_gt))
    # psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    # psnr = 10 * np.log10(255.0**2 / mse)
    # psnr = compare_psnr(img_this, img_gt, data_range=255)
    # psnr = compare_psnr(img_this / 255.0, img_gt / 255.0, data_range=1)
    ssim = compare_ssim(
        img_this, img_gt, data_range=img_this.max() - img_this.min(), multichannel=True
    )
    psnr = f'psnr: {psnr:.3f}'
    ssim = f'ssim: {ssim:.3f}'

    # pix_err (pixel_dist)
    x2 = np.square(pt_this - pt_gt)
    y2 = np.square(pt_this - pt_gt)
    pix_err = np.mean(np.sqrt(x2 + y2))
    pix = f' pix: {pix_err:.3f}'

    # homing_point_ratio
    hpr = pix_err / params.max_shift_pixels
    hpr = 1.0 - min(hpr, 1.0)
    hpr = f' hpr: {hpr:.3f}'

    color = (192, 192, 192)
    for i, ele in enumerate([psnr, ssim, pix, hpr]):
        cv2.putText(
            img_this, ele, (460, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

    return img_this


def plot_pt(params, img, pt_ori, pt_pred, pt_pert, img_gt, mode='gt'):
    color = [[0, 0, 255], [0, 255, 0], [0, 255, 255]]
    for i, pt in enumerate([pt_ori, pt_pred, pt_pert]):
        if mode == 'pred' and i == 2:
            break
        elif mode == 'pert' and i == 1:
            continue
        for k in range(len(pt_ori)):
            x, y = tuple(np.int32(pt)[k])
            cv2.circle(img, (x, y), 3, color[i], -1)
    if mode == 'pred':
        img = plot_indicator(params, pt_pred, pt_ori, img, img_gt, mode)
    elif mode == 'pert':
        img = plot_indicator(params, pt_pert, pt_ori, img, img_gt, mode)
    cv2.putText(img, mode, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    if mode == 'gt' and params.visualize_mode == 'train':
        idx = params.current_iter
        epoch = params.current_epoch
        cv2.putText(
            img,
            f'iter: [{idx:04}/{params.iter_max:04}]',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (192, 192, 192),
            2,
        )
        cv2.putText(
            img,
            f'epoch: [{epoch:2}/{params.num_epochs:2}]',
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (192, 192, 192),
            2,
        )
    return img


def plot_pt_v2(img, pt_ori, pt_pert):
    color = [[0, 0, 255], [0, 255, 255]]
    for i, pt in enumerate([pt_ori, pt_pert]):
        for k in range(len(pt_ori)):
            x, y = tuple(np.int32(pt)[k])
            cv2.circle(img, (x, y), 3, color[i], -1)
    return img


def visualize_supervised(params, data):
    '''
    num = 1 for training, every iteration_frequence, plot
    num = batch_size for evaluate
    '''
    idx = params.current_iter
    batch_size = int(data['image'].shape[0] / len(params.camera_list))
    if params.visualize_mode == 'train':
        num = 1
        if params.train_vis_iter_frequence <= 0:
            return
        if idx != 1 and idx % params.train_vis_iter_frequence != 0:
            return
    else:  # test evaluate
        num = batch_size

    homo = data['homo']
    undist = data['undist']
    bev_ori = data['bev_origin']
    bev_pred = data['bev_pred']
    bev_pert = data['bev_perturbed']
    pts_ori = data['coords_bev_ori'].detach().cpu().numpy()
    pts_pred = data['coords_bev_ori_pred'].detach().cpu().numpy()
    pts_pert = data['coords_bev_perturbed'].detach().cpu().numpy()
    name = data['name']
    set_name = data['path'][0][0].split(os.sep)[-2]
    set_sv_dir = os.path.join(params.model_dir, set_name)
    if idx == 1:
        os.makedirs(set_sv_dir, exist_ok=True)
        shutil.rmtree(set_sv_dir)
    for i, cam in enumerate(params.camera_list):
        sv_dir = os.path.join(set_sv_dir, cam)
        os.makedirs(sv_dir, exist_ok=True)
        for j in range(num):
            undist_cv2 = undist[i][j].detach().cpu().numpy().transpose((1, 2, 0))
            h_this = homo[i : (i + 1) * batch_size][j]
            homo_u2b_cv2 = h_this.detach().cpu().numpy()
            wh = tuple(params.wh_bev_fblr[cam])
            img_cv2 = cv2.warpPerspective(
                undist_cv2, homo_u2b_cv2, wh, cv2.INTER_LINEAR
            )
            img_cv2 = img_cv2[:, :, np.newaxis]
            img_torch = bev_pred[i][j].detach().cpu().numpy()
            img_torch = img_torch.transpose((1, 2, 0)).astype(np.uint8)
            img_pert = bev_pert[i][j].numpy().transpose((1, 2, 0))
            img_gt = bev_ori[i][j].numpy().transpose((1, 2, 0))
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
            img_torch = cv2.cvtColor(img_torch, cv2.COLOR_GRAY2BGR)
            img_pert = cv2.cvtColor(img_pert, cv2.COLOR_GRAY2BGR)
            pt_ori = pts_ori[i : (i + 1) * batch_size][j]
            pt_pred = pts_pred[i : (i + 1) * batch_size][j]
            pt_pert = pts_pert[i : (i + 1) * batch_size][j]
            img_gt = plot_pt(params, img_gt, pt_ori, pt_pred, pt_pert, img_gt, 'gt')
            img_cv2 = plot_pt(params, img_cv2, pt_ori, pt_pred, pt_pert, img_gt, 'pred')
            img_torch = plot_pt(
                params, img_torch, pt_ori, pt_pred, pt_pert, img_gt, 'pred'
            )
            img_pert = plot_pt(
                params, img_pert, pt_ori, pt_pred, pt_pert, img_gt, 'pert'
            )
            img_cv2 = np.concatenate([img_pert, img_cv2, img_gt], axis=0)
            img_torch = np.concatenate([img_pert, img_torch, img_gt], axis=0)
            cv2.imwrite(os.path.join(sv_dir, f'{name[i][j]}_cv2.jpg'), img_cv2)
            cv2.imwrite(os.path.join(sv_dir, f'{name[i][j]}_torch.jpg'), img_torch)
        cv2.imwrite(os.path.join(params.model_dir, f"vis_{cam}.jpg"), all)
    return


def visualize_unsupervised_inference(params, data):
    return


def visualize_unsupervised_kernel(params, data):
    '''
    num = 1 for training, every iteration_frequence, plot
    num = batch_size for evaluate
    '''
    idx = params.current_iter
    batch_size = int(data['image'].shape[0] / len(params.camera_list))
    if params.visualize_mode == 'train':
        num = 1
        if params.train_vis_iter_frequence <= 0:
            return
        if idx != 1 and idx % params.train_vis_iter_frequence != 0:
            return
    else:  # test evaluate
        num = batch_size

    sv_dir = f'{params.model_dir}/vis_{params.visualize_mode}'
    os.makedirs(sv_dir, exist_ok=True)
    wh_fblr = list(params.wh_bev_fblr.values())
    pts_ori = data['coords_bev_ori'].detach().cpu().numpy()
    pts_pred = data['coords_bev_ori_pred'].detach().cpu().numpy()
    pts_pert = data['coords_bev_perturbed'].detach().cpu().numpy()
    imgs = []

    for i, cam in enumerate(params.camera_list):
        for j in range(num):
            pt_ori = pts_ori[i : (i + 1) * batch_size][j]
            pt_pred = pts_pred[i : (i + 1) * batch_size][j]
            pt_pert = pts_pert[i : (i + 1) * batch_size][j]
            img_pd = data['bev_pred'][i][j].detach().cpu().numpy()
            img_gt = data['bev_origin'][i][j].detach().cpu().numpy()
            img_pt = data['bev_perturbed'][i][j].detach().cpu().numpy()
            img_pd = img_pd.transpose((1, 2, 0))
            img_gt = img_gt.transpose((1, 2, 0))
            img_pt = img_pt.transpose((1, 2, 0))
            img_pd = cv2.cvtColor(img_pd, cv2.COLOR_GRAY2BGR)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
            img_pt = cv2.cvtColor(img_pt, cv2.COLOR_GRAY2BGR)
            img_gt = plot_pt(params, img_gt, pt_ori, pt_pred, pt_pert, img_gt, 'gt')
            img_pd = plot_pt(params, img_pd, pt_ori, pt_pred, pt_pert, img_gt, 'pred')
            img_pt = plot_pt(params, img_pt, pt_ori, pt_pred, pt_pert, img_gt, 'pert')
            pt_pd_gt = np.concatenate([img_pt, img_pd, img_gt], axis=0)
            imgs.append(pt_pd_gt)
        # Concatenate 4 cameras
        cams = params.camera_list
        if len(cams) == 1:
            all = imgs[0]
        elif len(cams) == 4:
            fb = np.concatenate([imgs[0], imgs[1]], axis=0)
            lr = np.concatenate([imgs[2], imgs[3]], axis=0)
            # z1 = np.zeros([336 * 4, 1172 - 1078, 3])
            # z2 = np.zeros([4 * (439 - 336), 1172, 3])
            z1 = np.zeros([wh_fblr[0][1] * 6, wh_fblr[2][0] - wh_fblr[0][0], 3])
            z2 = np.zeros([6 * (wh_fblr[2][1] - wh_fblr[0][1]), wh_fblr[2][0], 3])
            fb = np.concatenate([fb, z1], axis=1)
            fb = np.concatenate([fb, z2], axis=0)
            all = np.concatenate([fb, lr], axis=1)
        elif len({'front', 'back'}.intersection(cams)) == 2:
            all = np.concatenate([imgs[0], imgs[1]], axis=0)
        elif len({'left', 'right'}.intersection(cams)) == 2:
            all = np.concatenate([imgs[0], imgs[1]], axis=0)
        elif len(cams) == 2 and cams[0] in ['front', 'back']:
            fb, lr = imgs[0], imgs[1]
            # z1 = np.zeros([336 * 2, 1172 - 1078, 3])
            # z2 = np.zeros([2 * (439 - 336), 1172, 3])
            z1 = np.zeros([wh_fblr[0][1] * 3, wh_fblr[2][0] - wh_fblr[0][0], 3])
            z2 = np.zeros([3 * (wh_fblr[2][1] - wh_fblr[0][1]), wh_fblr[2][0], 3])
            fb = np.concatenate([fb, z1], axis=1)
            fb = np.concatenate([fb, z2], axis=0)
            all = np.concatenate([fb, lr], axis=1)
        else:
            raise ValueError("not support 3 cameras")
        # Save result
        if num == 1:
            name = f'{idx:04}.jpg'
        else:  # num = bs
            name = f"{data['name'][i][j]}.jpg"
        cv2.imwrite(os.path.join(sv_dir, name), all)
    cv2.imwrite(os.path.join(params.model_dir, "vis.jpg"), all)

    return


def visualize_unsupervised(params, data):
    vis_mode = params.visualize_mode
    assert vis_mode in ['evaluate', 'train', 'inference']
    if vis_mode == 'inference':
        visualize_unsupervised_inference(params, data)
        return
    else:  # train, evaluate
        visualize_unsupervised_kernel(params, data)

    return


def visulize_results(params, data):
    mode = params.model_train_type
    if mode == 'supervised':
        visualize_supervised(params, data)
    elif mode == 'unsupervised':
        visualize_unsupervised(params, data)
    else:
        raise ValueError

    return
