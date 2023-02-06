from model.warp_head import warp_head
from util.postprocess import backbone_output_activation


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

    data = backbone_output_activation(params, data)
    data = warp_head(params, data)

    if params.is_unit_test_model:
        import sys
        import model.unit_test as ut

        ut.unit_test_first_stage_torch(data)
        ut.unit_test_second_stage_cv2()
        ut.unit_test_second_stage_torch(data)
        ut.unit_test_warp_image_torch()
        sys.exit()

    return data
