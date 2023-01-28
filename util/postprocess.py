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


def plot_indicator(params, pt_this, pt_gt, img_this, img_gt):
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


def plot_iter_info(params, img):
    str_info = [
        f'iter: [{params.current_iter:04}/{params.iter_max:04}]',
        f'epoch: [{params.current_epoch:2}/{params.num_epochs:2}]',
    ]
    txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    for i, ele in enumerate(str_info):
        cv2.putText(img, ele, (460, (i + 2) * 50), *txt_info)
    return img


def plot_pt(params, img, pt_ori, pt_pred, pt_pert, img_gt, mode='gt'):
    '''suitable for unsupervise and supervise'''
    # plot indicator first, otherwise `plot_op` will changes the picture
    if mode == 'pred':
        img = plot_indicator(params, pt_pred, pt_ori, img, img_gt)
    elif mode == 'pert':
        img = plot_indicator(params, pt_pert, pt_ori, img, img_gt)
    color = [[0, 0, 255], [0, 255, 0], [0, 255, 255]]
    for i, pt in enumerate([pt_ori, pt_pred, pt_pert]):
        if mode == 'pred' and i == 2:
            break
        elif mode == 'pert' and i == 1:
            continue
        for k in range(len(pt_ori)):
            x, y = tuple(np.int32(pt)[k])
            cv2.circle(img, (x, y), 3, color[i], -1)
    cv2.putText(img, mode, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    if mode == 'bev_origin' and params.visualize_mode == 'train':
        img = plot_iter_info(params, img)
    return img


def plot_pt_unsupervised_kernel(params, pts, bevs_cam):
    color = {
        'bev_perturbed': (0, 255, 255),
        'bev_perturbed_pred': (0, 128, 255),
        'bev_origin_pred': (0, 255, 0),
        'bev_origin': (0, 0, 255),
    }
    color_name = {
        'bev_perturbed': 'yellow',
        'bev_perturbed_pred': 'orange',
        'bev_origin_pred': 'green',
        'bev_origin': 'red',
    }

    def _plot_point(pt, name, img):
        for k in range(len(pt)):
            x, y = tuple(np.int32(pt)[k])
            cv2.circle(img, (x, y), 3, color[name], -1)
        return img

    pt_ori = pts['coords_bev_origin']
    img_ori = bevs_cam['bev_origin']
    imgs_cam = []
    for name, img in bevs_cam.items():
        pt = pts[f'coords_{name}']
        # plot indicator first, otherwise `plot_op` will changes the picture
        if name != 'bev_origin':
            img = plot_indicator(params, pt, pt_ori, img, img_ori)
        elif params.visualize_mode == 'train':
            img = plot_iter_info(params, img)
        img = _plot_point(pt_ori, 'bev_origin', img)
        img = _plot_point(pt, name, img)
        if name == 'bev_perturbed':
            n2 = 'bev_perturbed_pred'
            img = _plot_point(pts[f'coords_{n2}'], n2, img)
        elif name == 'bev_perturbed_pred':
            n1 = 'bev_perturbed'
            img = _plot_point(pts[f'coords_{n1}'], n1, img)
        elif name == 'bev_origin_pred':
            pass
        elif name == 'bev_origin':
            for ele in ['bev_perturbed', 'bev_perturbed_pred', 'bev_origin_pred']:
                img = _plot_point(pts[f'coords_{ele}'], ele, img)
        txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
        cv2.putText(img, name[4:], (50, 50), *txt_info)
        cv2.putText(img, color_name[name], (50, 100), *txt_info)
        imgs_cam.append(img)

    return imgs_cam


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

    homo = data['homo_u2b']
    undist = data['undist']
    bev_origin = data['bev_origin']
    bev_origin_pred = data['bev_origin_pred']
    bev_pert = data['bev_perturbed']
    pts_ori = data['coords_bev_origin'].detach().cpu().numpy()
    pts_pred = data['coords_bev_origin_pred'].detach().cpu().numpy()
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
            h_this = homo[i * batch_size : (i + 1) * batch_size][j]
            homo_u2b_cv2 = h_this.detach().cpu().numpy()
            wh = tuple(params.wh_bev_fblr[cam])
            img_cv2 = cv2.warpPerspective(
                undist_cv2, homo_u2b_cv2, wh, cv2.INTER_LINEAR
            )
            img_cv2 = img_cv2[:, :, np.newaxis]
            img_torch = bev_origin_pred[i][j].detach().cpu().numpy()
            img_torch = img_torch.transpose((1, 2, 0)).astype(np.uint8)
            img_pert = bev_pert[i][j].numpy().transpose((1, 2, 0))
            img_gt = bev_origin[i][j].numpy().transpose((1, 2, 0))
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
            img_torch = cv2.cvtColor(img_torch, cv2.COLOR_GRAY2BGR)
            img_pert = cv2.cvtColor(img_pert, cv2.COLOR_GRAY2BGR)
            pt_ori = pts_ori[i * batch_size : (i + 1) * batch_size][j]
            pt_pred = pts_pred[i * batch_size : (i + 1) * batch_size][j]
            pt_pert = pts_pert[i * batch_size : (i + 1) * batch_size][j]
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


def stack_multi_cams_imgs(params, imgs, bev_name_list):
    cams = params.camera_list
    wh_fblr = list(params.wh_bev_fblr.values())
    k = len(bev_name_list)
    if len(cams) == 1:
        imgs_all = imgs[0]
    elif len(cams) == 4:
        fb = np.concatenate([imgs[0], imgs[1]], axis=0)
        lr = np.concatenate([imgs[2], imgs[3]], axis=0)
        # z1 = np.zeros([336 * 4, 1172 - 1078, 3])
        # z2 = np.zeros([4 * (439 - 336), 1172, 3])
        z1 = np.zeros([wh_fblr[0][1] * (k * 2), wh_fblr[2][0] - wh_fblr[0][0], 3])
        z2 = np.zeros([(k * 2) * (wh_fblr[2][1] - wh_fblr[0][1]), wh_fblr[2][0], 3])
        fb = np.concatenate([fb, z1], axis=1)
        fb = np.concatenate([fb, z2], axis=0)
        imgs_all = np.concatenate([fb, lr], axis=1)
    elif len({'front', 'back'}.intersection(cams)) == 2:
        imgs_all = np.concatenate([imgs[0], imgs[1]], axis=0)
    elif len({'left', 'right'}.intersection(cams)) == 2:
        imgs_all = np.concatenate([imgs[0], imgs[1]], axis=0)
    elif len(cams) == 2 and cams[0] in ['front', 'back']:
        fb, lr = imgs[0], imgs[1]
        # z1 = np.zeros([336 * 2, 1172 - 1078, 3])
        # z2 = np.zeros([2 * (439 - 336), 1172, 3])
        z1 = np.zeros([wh_fblr[0][1] * k, wh_fblr[2][0] - wh_fblr[0][0], 3])
        z2 = np.zeros([k * (wh_fblr[2][1] - wh_fblr[0][1]), wh_fblr[2][0], 3])
        fb = np.concatenate([fb, z1], axis=1)
        fb = np.concatenate([fb, z2], axis=0)
        imgs_all = np.concatenate([fb, lr], axis=1)
    else:
        raise ValueError("not support 3 cameras")

    return imgs_all


def visualize_unsupervised_kernel(params, data):
    '''
    num = 1 for training, every iteration_frequence, plot
    num = batch_size for evaluate
    # have used dataset supervised info for visualization
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

    pt_name_list = [
        'coords_bev_perturbed',  # supervised info
        'coords_bev_perturbed_pred',
        'coords_bev_origin_pred',  # supervised info
        'coords_bev_origin',
    ]
    pts = {}
    for name in pt_name_list:
        pts[name] = data[name].detach().cpu().numpy()
    bev_name_list = [
        'bev_perturbed',
        'bev_perturbed_pred',
        'bev_origin_pred',
        'bev_origin',  # put it at the end for drawing the training infos
    ]
    bevs = {}
    for name in bev_name_list:
        bevs[name] = data[name]

    imgs = []
    for i, cam in enumerate(params.camera_list):
        for j in range(num):
            pts_cam = {}
            for k, v in pts.items():
                pts_cam[k] = v[i * batch_size : (i + 1) * batch_size][j]
            bevs_cam = {}
            for k, v in bevs.items():
                bev = bevs[k][i][j].detach().cpu().numpy().transpose((1, 2, 0))
                bevs_cam[k] = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
            imgs_cam = plot_pt_unsupervised_kernel(params, pts_cam, bevs_cam)
            stack = np.concatenate(imgs_cam, axis=0)
            imgs.append(stack)
    # Concatenate 4 or multi cameras
    img_stack_all = stack_multi_cams_imgs(params, imgs, bev_name_list)
    # Save result
    if num == 1:
        name = f'{params.current_iter:04}.jpg'
    else:  # num = bs
        name = f"{data['name'][i][j]}.jpg"
    sv_dir = f'{params.model_dir}/vis_{params.visualize_mode}'
    os.makedirs(sv_dir, exist_ok=True)
    cv2.imwrite(os.path.join(sv_dir, name), img_stack_all)
    cv2.imwrite(os.path.join(params.model_dir, "vis.jpg"), img_stack_all)

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
