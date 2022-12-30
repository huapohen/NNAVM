"""Defines the detector network structure."""


def fetch_net(params):

    if params.model_type == "yolo":
        from model.yolo import get_model
    elif params.model_type == 'light':
        from model.light import get_model
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return get_model(params)
