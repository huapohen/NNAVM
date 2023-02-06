'''
unit test for net second stage pipeline
'''

__all__ = [
    "unit_test_second_stage_cv2",
    "unit_test_second_stage_torch",
    "unit_test_warp_image_torch",
]


def unit_test_second_stage_cv2():
    '''
    second stage net unit test
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
        img_bev_pert = cv2.warpPerspective(
            img_undist, H_u2p_pert, bev_wh, cv2.INTER_LINEAR
        )
        # cv2.imwrite(f'model/vis/{pert_value:02}.jpg', img_bev_pert)
        # continue

        # 2. 训练 和 测试，非推理
        # scale = 20
        scale = 1  # 100% right, no error
        offset_pred = offset_pert.copy() * scale
        bev_pts_pert_pred = bev_pts_ori + offset_pred
        H_p2b_pred, _ = cv2.findHomography(bev_pts_pert_pred, bev_pts_ori, 0)
        H_u2p_pred, _ = cv2.findHomography(undist_pts, bev_pts_pert_pred, 0)  # 训练
        H_u2b_pred = H_p2b_pred @ H_u2p_pert  # u to p to b , 从右往左，左连乘矩阵
        # H_u2b_pred = H_p2b_pred @ H_u2p_pred  # 此方法失效，扰动作为了中间变量，被抵消了
        img_bev_pred = cv2.warpPerspective(
            img_undist, H_u2b_pred, bev_wh, cv2.INTER_LINEAR
        )
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
        img_bev_ori = cv2.warpPerspective(
            img_undist, H_u2b_ori, bev_wh, cv2.INTER_LINEAR
        )

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

        # 9. for training and test:
        #   undist -> bev_pert, and bev_pert_pred -> bev_ori
        #   unaligned situation, less error
        ''' 应该采用 # 7. image-level supervised，因为数据集都是有监督的，除非你不用数据集 '''
        offset_pert_pred_err = offset_pert * 0.8  # 20% error
        bev_pts_pert_pred_err = bev_pts_ori + offset_pert_pred_err
        H_p2b_pred_err, _ = cv2.findHomography(bev_pts_pert_pred_err, bev_pts_ori, 0)
        H_u2b_pred_err = H_p2b_pred_err @ H_u2p_pert
        img_bev_pred_err = cv2.warpPerspective(
            img_undist, H_u2b_pred_err, bev_wh, cv2.INTER_LINEAR
        )

        # 10. for inference:
        #   undist -> bev_ori(in fact, perturbed), and bev_pert_pred -> bev_ori
        offset_pert_pred_infact = offset_pert * 0.75  # 25% error
        bev_pts_pert_pred_infact = bev_pts_ori + offset_pert_pred_infact
        H_p2b_pred_infact, _ = cv2.findHomography(
            bev_pts_pert_pred_infact, bev_pts_ori, 0
        )
        bev_pts_infact = bev_pts_ori + offset_pert * 0.6
        H_u2b_infact, _ = cv2.findHomography(undist_pts, bev_pts_infact, 0)
        H_u2b_pred_infact = H_p2b_pred_infact @ H_u2b_infact
        img_bev_pred_infact = cv2.warpPerspective(
            img_undist, H_u2b_pred_infact, bev_wh, cv2.INTER_LINEAR
        )

        # 11. vs # 9 for training and test: choose # 7
        offset_pert_pred_err = offset_pert * 0.8  # 20% error
        bev_pts_ori_pred_err = bev_pts_pert - offset_pert_pred_err
        H_u2b_pred_err, _ = cv2.findHomography(undist_pts, bev_pts_ori_pred_err, 0)
        img_bev_pred_err_supervised = cv2.warpPerspective(
            img_undist, H_u2b_pred_err, bev_wh, cv2.INTER_LINEAR
        )

        pts_diff = bev_pts_ori_pred - bev_pts_ori
        off_diff = offset_pred - offset_pert
        # print('\n\tpoints diff is: ', pts_diff.sum())
        # print('\n\toffset diff is: ', off_diff.sum())

        imgs_all = {
            '1. pert': img_bev_pert,
            '2. pred': img_bev_pred,
            '3. ori': img_bev_ori,
            '4. directly': img_bev_directly,
            '5. mask': bev_mask * 128,
            '6. ori_mask': bev_ori_mask,
            '7. pred_supervised': bev_pred_supervised,
            '8. pert_pred': bev_pert_pred,
            '9. pred_err': img_bev_pred_err,
            '10. pred_infact': img_bev_pred_infact,
            '11. pred_err_supervised': img_bev_pred_err_supervised,
        }

        def _plot_id(s, img):
            cv2.putText(
                img, s, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (192, 192, 192), 3
            )
            return img

        imgs_all = [_plot_id(k, v) for k, v in imgs_all.items()]

        img_merge = np.concatenate(imgs_all, axis=0)
        # cv2.imwrite(f"{sv_dir}/net_util_test_{pert_value:02}.jpg", img_merge)
        cv2.imwrite(f"{sv_dir}/{scale}_{pert_value:02}.jpg", img_merge)

        ''''
        核心是offset准

        '''
        # print()


def unit_test_first_stage_torch(data):
    prt_str = (
        f"\n input images shape: {data['image'].shape} \n"
        + f"output offset shape: {data['offset_pred'].shape} \n"
        + f" input offset e.g.:\n{data['offset'][0].detach().cpu().numpy()} \n"
        + f"output offset e.g.:\n{data['offset_pred'][0].detach().cpu().numpy()} \n"
    )
    print(prt_str)


def unit_test_second_stage_torch(data):
    import os
    import cv2
    import numpy as np

    pts_docs = '''
    offset_err = offset_pred - offset_gt
    when offset_err is small, 
        the `bev_perturbed_pred` is close to `bev_perturbed`(bev's pert_gt),
        and the `bev_origin_pred` is close to `bev_origin`(bev's ori_gt).
    when offset_err is big:
        the `bev_perturbed_pred` is close to `bev_origin`, 
        and the `bev_origin_pred` is close to `bev_perturbed`(input image).
    '''
    p_pd = data['offset_pred'].detach().cpu().numpy()
    p_gt = data['offset'].detach().cpu().numpy()
    pm = [
        np.mean(np.sqrt(np.square(p[:, 0]) + np.square(p[:, 1])))
        for p in [p_pd, p_gt, p_pd - p_gt]
    ]
    pts_info = (
        f"    {'offset_pd:':10} {pm[0]:6.3f} \n"
        + f"    {'offset_gt:':10} {pm[1]:6.3f} \n"
        + f"    {'offset_err:':11} {pm[2]:6.3f} \n"
    )
    print(pts_docs)
    print(pts_info)

    item_dic = {
        'pert': 'bev_perturbed',
        'pert_pred': 'bev_perturbed_pred',
        'ori_pred': 'bev_origin_pred',
        'ori': 'bev_origin',
        'ori_pred_supervised': 'bev_origin_pred_supervised',
    }
    img_list = []
    for k, v in item_dic.items():
        img = data[v][0][0].detach().cpu().numpy().transpose((1, 2, 0))
        path = os.path.join('model', 'vis', f"{v}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, img)
        cv2.putText(img, k, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (192, 192, 192), 2)
        img_list.append(img)
    img_merge = np.concatenate(img_list, axis=0)
    sv_path = os.path.join('model', 'vis', 'bev_all ####.jpg')
    cv2.imwrite(sv_path, img_merge)

    print("save all unit test results in `model/vis/` \n")


def unit_test_warp_image_torch():
    from util.torch_func_op import unit_test_warp_image

    unit_test_warp_image()


if __name__ == '__main__':

    unit_test_second_stage_cv2()
    unit_test_second_stage_torch()
    unit_test_warp_image_torch()
