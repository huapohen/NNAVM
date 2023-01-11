'''
    second stage net util test
'''
import os
import cv2
import shutil
import numpy as np


# front camera
img_undist = cv2.imread("dataset/data/undist/front.jpg")
bev_wh = (1078, 336)
sv_dir = 'model/vis'
if os.path.exists(sv_dir):
    shutil.rmtree(sv_dir)
os.makedirs(sv_dir, exist_ok=True)

undist_pts = np.array([732, 784, 1236, 781, 282, 946, 1601, 916]).reshape(-1, 4, 2)
bev_pts_ori = np.array([309, 216, 769, 216, 309, 316, 769, 316]).reshape(-1, 4, 2)


for pert_value in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

    # 1. 扰动
    offset_pert = np.random.randint(-pert_value, pert_value, bev_pts_ori.shape)
    bev_pts_pert = bev_pts_ori + offset_pert
    H_u2p_pert, _ = cv2.findHomography(undist_pts, bev_pts_pert, 0)
    img_bev_pert = cv2.warpPerspective(img_undist, H_u2p_pert, bev_wh, cv2.INTER_LINEAR)

    # 2. 训练 和 测试，非推理
    # scale = 20
    scale = 1
    offset_pred = offset_pert.copy() * scale
    bev_pts_pert_pred = bev_pts_ori + offset_pred
    H_p2b_pred, _ = cv2.findHomography(bev_pts_pert_pred, bev_pts_ori, 0)
    H_u2p_pred, _ = cv2.findHomography(undist_pts, bev_pts_pert_pred, 0)  # 训练
    H_u2b_pred = H_p2b_pred @ H_u2p_pert  # u to p to b , 从右往左，左连乘矩阵
    # H_u2b_pred = H_p2b_pred @ H_u2p_pred  # 此方法失效，扰动作为了中间变量，被抵消了
    img_bev_pred = cv2.warpPerspective(img_undist, H_u2b_pred, bev_wh, cv2.INTER_LINEAR)
    # H_u2p_ori, _ = cv2.findHomography(undist_pts, bev_pts_ori, 0) # 推理
    # H_u2b_pred = H_p2b_pred @ H_u2p_ori # 推理
    '''
    分析原因在于，用的 undist 到 bev扰动图，这个要对齐 所用的图，输入的图。
        鸟瞰图用的 扰动后的图，这样算的 undist到bev的H；只不过 鸟瞰图的点是扰动后的点。
    推理时本身是扰动后的图，鸟瞰图的点是正常的原始的点。
    而如果用 bev预测的扰动图，反向找点就回到 bev_ori了，行不通了。
    '''

    # 3. 真实
    H_u2b_ori, _ = cv2.findHomography(undist_pts, bev_pts_ori, 0)
    img_bev_ori = cv2.warpPerspective(img_undist, H_u2b_ori, bev_wh, cv2.INTER_LINEAR)

    # 4. bev_pert 直接 warp to bev_ori_pred
    img_bev_directly = cv2.warpPerspective(
        img_bev_pert, H_p2b_pred, bev_wh, cv2.INTER_LINEAR
    )

    # 5. mask
    bev_mask = cv2.warpPerspective(
        np.ones_like(img_bev_pred), H_p2b_pred, bev_wh, cv2.INTER_LINEAR
    )

    # 6. mask bev_ori
    bev_ori_mask = bev_mask * img_bev_ori

    # 7. image-level supervised
    bev_pts_ori_pred = bev_pts_pert - offset_pred
    H_u2b_pred, _ = cv2.findHomography(undist_pts, bev_pts_ori_pred, 0)
    bev_pred_supervised = cv2.warpPerspective(
        img_undist, H_u2b_pred, bev_wh, cv2.INTER_LINEAR
    )
    '''
        和 2. 结果不一样; 
        2. 是用确定的 bev_ori图（pert点固定，但pred在变），
            此处是用确定的 bev_pts_pert点（对应确定的图）
        确定的 pts_bev_pert 使得， u2p2b是连贯的；适合训练，但无法推理，且是监督
        确定的  pts_bev_ori 使得， u2p2b是断开的；训练有歧义，但可以推理，训练和推理也不一致
                                  因为，u2p->p'2b，p!=p'（固定！=预测）；中间的图是断开的。不一样
                                  推理却用的 u2b+p'2b这个u2b的图都是扰动的，但H固定
                                  如果网络学得很好，offset_pred完美预测，则差异没有。
    '''

    # 8. bev_pert_pred
    bev_pert_pred = cv2.warpPerspective(
        img_undist, H_u2p_pred, bev_wh, cv2.INTER_LINEAR
    )

    pts_diff = bev_pts_ori_pred - bev_pts_ori
    off_diff = offset_pred - offset_pert
    print('\n\tpoints diff is: ', pts_diff.sum())
    print('\n\toffset diff is: ', off_diff.sum())

    img_merge = np.concatenate(
        [
            img_bev_pert,
            img_bev_pred,
            img_bev_ori,
            img_bev_directly,
            bev_mask * 128,
            bev_ori_mask,
            bev_pred_supervised,
            bev_pert_pred,
        ],
        axis=0,
    )
    # cv2.imwrite(f"{sv_dir}/net_util_test_{pert_value:02}.jpg", img_merge)
    cv2.imwrite(f"{sv_dir}/{scale}_{pert_value:02}.jpg", img_merge)

    ''''
    核心是offset准

    '''
    print()
