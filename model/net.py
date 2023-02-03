from model.warp_head import warp_head


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

    # get bev
    data = warp_head(params, data)

    return data
