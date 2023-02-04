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

    return data
