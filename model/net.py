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

    bs = int(data['image'].shape[0] / len(params.camera_list))

    if params.nn_output_do_tanh:
        data['offset_pred'] = data['offset_pred'].tanh() * params.max_shift_pixels

    # data['offset_pred'] = data['offset']
    data['coords_bev_perturbed_pred'] = data["coords_bev_origin"] + data['offset_pred']
    data['H_bev_pert_pred_to_origin'] = dlt_homo(
        data["coords_bev_perturbed_pred"], data['coords_bev_origin']
    )

    if params.bev_mask_mode:
        bs = int(data['image'].shape[0] / len(params.camera_list))
        data['bev_origin_pred'] = get_warp_image(
            params, bs, data['H_bev_pert_pred_to_origin'], data['bev_perturbed']
        )
        mask = [torch.ones_like(img) for img in data['bev_perturbed']]
        data['mask_bev_origin_pred'] = get_warp_image(
            params, bs, data['H_bev_pert_pred_to_origin'], mask
        )
        data['bev_origin_masked'] = [
            x[0] * x[1]
            for x in list(zip(data['bev_origin'], data['mask_bev_origin_pred']))
        ]

        return data

    if params.train_eval_inference in ['train', 'eval']:
        # train & test:
        #   the input fev and undist is ground thruth, correct images
        data['H_undist_to_bev_pert'] = dlt_homo(
            data["coords_undist"], data['coords_bev_perturbed']
        )
        # undist -> bev perturbed -> bev pred
        data['homo_u2b'] = (
            data['H_bev_pert_pred_to_origin'] @ data['H_undist_to_bev_pert']
        )

        # used for visualization
        data['H_undist_to_bev_pert_pred'] = dlt_homo(
            data["coords_undist"], data['coords_bev_perturbed_pred']
        )
        data['bev_perturbed_pred'] = get_warp_image(
            params, bs, data['H_undist_to_bev_pert_pred'], data['undist']
        )
        data['coords_bev_origin_pred'] = (
            data["coords_bev_perturbed"] - data['offset_pred']
        )

        if params.second_stage_image_supervised:
            # In inference, input doesn't have coords_bev_perturbed, which is supervised info.
            data['homo_u2b'] = dlt_homo(
                data["coords_undist"], data['coords_bev_origin_pred']
            )

    elif params.train_eval_inference == 'inference':
        # Inference:
        #   the input fish-eye view has two situations: distorted or correct,
        #   so the undistord images may be perturbed.
        #   For example, a car drives over a speed bump.
        #   Similarly, the car drove through the flat ground without perturbed.
        data['H_undist_to_bev_origin'] = dlt_homo(
            data["coords_undist"], data['coords_bev_origin']
        )
        # (1) H_u2? @ undist_real-> bev_real
        # (2) H_?2b @ bev_real -> bev_origin_pred
        # (1)+(2): (H_?2b @ H_u2?= Homo) @ undist_real -> bev_origin_pred
        data['homo_u2b'] = (
            data['H_bev_pert_pred_to_origin'] @ data['H_undist_to_bev_origin']
        )
    else:
        raise ValueError

    # for train & test: undist_gt to bev_pert to bev_origin_pred
    # for inference:    undist_?  to bev_? to bev_origin_pred
    # because undist_? in real world, maybe correct or error (same as perturbed)
    # add some right images as negtive samples

    data['bev_origin_pred'] = get_warp_image(
        params, bs, data['homo_u2b'], data['undist']
    )

    return data
