import torch
from easydict import EasyDict
from dataset.datamaker.calibrate_params import CalibrateParameter
from util.torch_func_op import dlt_homo, warp_image_to_bev


def warp_head(params, data):
    bs = int(data['image'].shape[0] / len(params.camera_list))
    params.calib_param = EasyDict(CalibrateParameter().__dict__)

    data['coords_bev_perturbed_pred'] = data["coords_bev_origin"] + data['offset_pred']

    data['H_bev_pert_pred_to_origin'] = dlt_homo(
        data["coords_bev_perturbed_pred"], data['coords_bev_origin']
    )

    if params.bev_mask_mode:
        data['bev_origin_pred'] = warp_image_to_bev(
            params, bs, data['H_bev_pert_pred_to_origin'], data['bev_perturbed']
        )
        mask = [torch.ones_like(img) for img in data['bev_perturbed']]
        data['mask_bev_origin_pred'] = warp_image_to_bev(
            params, bs, data['homo_b2u_pert_pred'], mask
        )

        data['bev_origin_masked'] = [
            x[0] * x[1]
            for x in list(zip(data['bev_origin'], data['mask_bev_origin_pred']))
        ]

        return data

    if params.train_eval_inference in ['train', 'eval']:
        # train & test:
        #   the input fev and undist is ground thruth, correct images
        data['homo_u2b_pert'] = dlt_homo(
            data["coords_undist"], data['coords_bev_perturbed']
        )
        # undist -> bev perturbed -> bev pred
        data['homo_u2b'] = data['H_bev_pert_pred_to_origin'] @ data['homo_u2b_pert']
        data['homo_b2u'] = torch.inverse(data['homo_u2b'])

        data['homo_b2u_pert_pred'] = dlt_homo(
            data['coords_bev_perturbed_pred'], data["coords_undist"]
        )

        data['bev_perturbed_pred'] = warp_image_to_bev(
            params, bs, data['homo_b2u_pert_pred'], data[params.src_img_mode]
        )

        data['coords_bev_origin_pred'] = (
            data["coords_bev_perturbed"] - data['offset_pred']
        )

        if params.second_stage_image_supervised:
            # In inference, input doesn't have coords_bev_perturbed, which is supervised info.
            data['homo_b2u'] = dlt_homo(
                data['coords_bev_origin_pred'], data["coords_undist"]
            )

        # for versus
        data['homo_b2u_supervised'] = dlt_homo(
            data['coords_bev_origin_pred'], data["coords_undist"]
        )
        data['bev_origin_pred_supervised'] = warp_image_to_bev(
            params, bs, data['homo_b2u_supervised'], data[params.src_img_mode]
        )

    elif params.train_eval_inference == 'inference':
        # Inference:
        #   the input fish-eye view has two situations: distorted or correct,
        #   so the undistord images may be perturbed.
        #   For example, a car drives over a speed bump.
        #   Similarly, the car drove through the flat ground without perturbed.
        data['homo_u2b_origin'] = dlt_homo(
            data["coords_undist"], data['coords_bev_origin']
        )
        # (1) H_u2? @ undist_real-> bev_real
        # (2) H_?2b @ bev_real -> bev_origin_pred
        # (1)+(2): (H_?2b @ H_u2?= Homo) @ undist_real -> bev_origin_pred
        data['homo_u2b'] = data['H_bev_pert_pred_to_origin'] @ data['homo_u2b_origin']
        data['homo_b2u'] = torch.inverse(data['homo_u2b'])
    else:
        raise ValueError

    # for train & test: undist_gt to bev_pert to bev_origin_pred
    # for inference:    undist_?  to bev_? to bev_origin_pred
    # because undist_? in real world, maybe correct or error (same as perturbed)
    # add some right images as negtive samples

    data['bev_origin_pred'] = warp_image_to_bev(
        params, bs, data['homo_b2u'], data[params.src_img_mode]
    )

    return data
