"""Defines the detector network structure."""


def fetch_net(params):

    if params.net_type == "eeavm":
        from model.yolo import get_model
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return get_model(params)
