import torch
from util.torch_op import dlt_homo, get_warp_image


def fetch_net(params):
    '''first stage'''
    if params.model_type == "yolo":
        from model.yolo import get_model
    elif params.model_type == 'light':
        from model.light import get_model
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))

    return get_model(params)


def second_stage(params, data):

    if params.nn_output_do_tanh:
        data['offset_pred'] = data['offset_pred'].tanh() * params.max_shift_pixels

    data['coords_bev_perturbed_pred'] = data["coords_bev_ori"] + data['offset_pred']
    data['H_bev_pt2gt'] = dlt_homo(
        data["coords_bev_perturbed_pred"], data['coords_bev_ori']
    )

    if params.bev_mask_mode:
        bs = int(data['image'].shape[0] / len(params.camera_list))
        data['bev_pred'] = get_warp_image(
            params, bs, data['H_bev_pt2gt'], data['bev_perturbed']
        )
        mask = [torch.ones_like(img) for img in data['bev_perturbed']]
        data['mask'] = get_warp_image(params, bs, data['H_bev_pt2gt'], mask)
        data['bev_ori'] = [
            x[0] * x[1] for x in list(zip(data['bev_ori'], data['mask']))
        ]

        return data

    if not params.inference_mode:
        # train & test: the input fev and undist is ground thruth, correct images
        data['H_undist2bev'] = dlt_homo(
            data["coords_undist"], data['coords_bev_perturbed_pred']
        )
        data['homo'] = data['H_undist2bev'] @ data['H_bev_pt2gt']

        # used for visualization
        data['coords_bev_ori_pred'] = data["coords_bev_perturbed"] - data['offset_pred']
        if params.second_stage_image_supervised:
            data['homo'] = dlt_homo(data["coords_undist"], data['coords_bev_ori_pred'])
    else:
        # Inference: the input fish-eye view is distorted, also as the undistored image
        data['H_undist2bev'] = dlt_homo(data["coords_undist"], data['coords_bev_ori'])
        # (1) undist_real = undist_pert @ H_u2b -> bev_pert
        # (2) bev_pert @ H_pt2gt -> bev_pred
        # (1)+(2): undist_pert @ (H_u2b @ H_pt2gt = Homo) -> bev_pred
        data['homo'] = data['H_undist2bev'] @ data['H_bev_pt2gt']

    # for train & test: undist_gt to bev_pert
    # for inference:    undist_?  to bev_pert
    # because undist_? in real world, maybe correct or error (same as perturbed)
    # add some right images ?

    bs = int(data['image'].shape[0] / len(params.camera_list))
    data['bev_pred'] = get_warp_image(params, bs, data['homo'], data['undist'])

    return data
